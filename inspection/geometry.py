"""Planar registration in metres and radians. Poses map local points into their parent frame."""

import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
from scipy.sparse import eye, lil_matrix
from scipy.sparse.linalg import splu


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def rot(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def transform(points, pose):
    return points @ rot(pose[2]).T + pose[:2]


def relative(a, b):
    return np.r_[(b[:2] - a[:2]) @ rot(a[2]), wrap(b[2] - a[2])]


def compose(a, b):
    return np.r_[transform(b[:2], a), wrap(a[2] + b[2])]


def voxel(points, size):
    if len(points) == 0:
        return points
    cells = np.floor(points / size).astype("int32")
    # Encode signed XY in lexicographic order. The first return in each cell is preserved.
    if cells.shape[1] == 2:
        keys = (cells[:, 0].astype("int64") << 32) | (
            cells[:, 1].astype("int64") + 2**31
        )
        _, idx = np.unique(keys, return_index=True)
    else:
        _, idx = np.unique(cells, axis=0, return_index=True)
    return points[idx]


def register(source, target, seed, radius=0.60, iterations=12):
    """Trimmed point-to-point ICP. Scores include all source points, including outliers."""
    pose = np.asarray(seed).copy()
    if (
        len(source) < 12
        or (target.n if isinstance(target, cKDTree) else len(target)) < 16
    ):
        return pose, 0.0, np.inf
    tree = target if isinstance(target, cKDTree) else cKDTree(target)
    for i in range(iterations):
        points = transform(source, pose)
        d, idx = tree.query(points)
        inside = d < radius
        if inside.sum() < 12:
            break
        limit = max(0.065, np.quantile(d[inside], 0.82))
        mask = inside & (d <= limit)
        a = points[mask]
        b = tree.data[idx[mask]]
        if len(a) < 8:
            break
        ac, bc = a.mean(0), b.mean(0)
        u, _, v = np.linalg.svd((a - ac).T @ (b - bc))
        correction = v.T @ u.T
        if np.linalg.det(correction) < 0:
            v[-1] *= -1
            correction = v.T @ u.T
        yaw = np.arctan2(correction[1, 0], correction[0, 0])
        delta = np.r_[bc - correction @ ac, yaw]
        new = compose(delta, pose)
        step = relative(pose, new)
        pose = new
        if np.linalg.norm(step[:2]) < 0.002 and abs(step[2]) < 0.001:
            break
    d = tree.query(transform(source, pose))[0]
    score = float((d < 0.21).mean())
    return pose, score, float(np.median(d))


def pose_graph(initial, edges):
    # Each edge carries an independently gated observation and inverse standard deviations.
    if len(initial) < 2:
        return initial, np.full(len(initial), 0.04)
    aa = np.array([e[0] for e in edges], dtype=int)
    bb = np.array([e[1] for e in edges], dtype=int)
    observed = np.array([e[2] for e in edges])
    weights = np.array([e[3] for e in edges])

    def residual(x):
        poses = np.vstack([initial[0], x.reshape(-1, 3)])
        a, b = poses[aa], poses[bb]
        d = b[:, :2] - a[:, :2]
        cs, sn = np.cos(a[:, 2]), np.sin(a[:, 2])
        err = np.column_stack(
            [
                cs * d[:, 0] + sn * d[:, 1],
                -sn * d[:, 0] + cs * d[:, 1],
                wrap(b[:, 2] - a[:, 2] - observed[:, 2]),
            ]
        )
        err[:, :2] -= observed[:, :2]
        return (err * weights).ravel()

    jac = lil_matrix((3 * len(edges), 3 * (len(initial) - 1)), dtype=int)
    for k, (i, j, _, _) in enumerate(edges):
        for n in (i, j):
            if n:
                jac[3 * k : 3 * k + 3, 3 * (n - 1) : 3 * n] = 1
    fit = least_squares(
        residual,
        initial[1:].ravel(),
        jac_sparsity=jac.tocsr(),
        loss="soft_l1",
        f_scale=1.5,
        max_nfev=35,
    )
    # Marginal XY uncertainty with the initial frame fixed. Weights are engineering noise assumptions,
    # so this is internal precision rather than a calibrated absolute position error.
    jacobian = fit.jac.tocsr()
    hessian = (
        jacobian.T @ jacobian + eye(jacobian.shape[1], format="csr") * 1e-8
    ).tocsc()
    factor = splu(hessian)
    chosen = np.unique(
        np.linspace(1, len(initial) - 1, min(len(initial) - 1, 150), dtype=int)
    )
    variance = []
    for part in np.array_split(chosen, max(1, (len(chosen) + 15) // 16)):
        diag_ids = np.column_stack([3 * (part - 1), 3 * (part - 1) + 1]).ravel()
        basis = np.zeros((hessian.shape[0], len(diag_ids)))
        basis[diag_ids, np.arange(len(diag_ids))] = 1
        columns = factor.solve(basis)
        variance.extend(
            np.maximum(0, columns[diag_ids, np.arange(len(diag_ids))])
            .reshape(-1, 2)
            .sum(axis=1)
        )
    sigma = np.sqrt(
        0.04**2
        + np.interp(np.arange(len(initial)), np.r_[0, chosen], np.r_[0, variance])
    )
    return np.vstack([initial[0], fit.x.reshape(-1, 3)]), sigma

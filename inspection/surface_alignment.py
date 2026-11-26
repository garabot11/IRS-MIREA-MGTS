"""Bounded corrections of measured voxel centroids using nearby repeated observations."""

import numpy as np
from scipy.spatial import cKDTree

ALGORITHM = "local_surface_fit_v1"


def _surface_steps(points, neighbors, mask, size):
    """Fit broad planes, round vertical sections and unresolved narrow post axes."""
    count = mask.sum(axis=1)
    weights = mask / np.maximum(count[:, None], 1)
    # Relative coordinates keep covariance and the circle fit well conditioned.
    rel = (neighbors - points[:, None, :]) / size
    center = np.einsum("nk,nkj->nj", weights, rel)
    local = rel - center[:, None, :]
    cov = (local * weights[:, :, None]).transpose(0, 2, 1) @ local
    values, axes = np.linalg.eigh(cov)
    normal, axis = axes[:, :, 0], axes[:, :, 2]
    height = rel[:, :, 2]
    above_below = (mask & (height < -1)).any(1) & (mask & (height > 1)).any(1)
    offsets = np.einsum("nkj,nj->nk", local, normal)
    offsets_sorted = np.sort(np.where(mask, offsets, np.inf), axis=1)
    # Two distinct parallel faces must not be pulled onto their mid-plane.
    cut = np.arange(1, mask.shape[1])
    interior = (cut >= count[:, None] * 0.2) & (cut <= count[:, None] * 0.8)
    finite = np.where(np.isfinite(offsets_sorted), offsets_sorted, 0)
    split = (interior & (np.diff(finite, axis=1) > 0.29)).any(axis=1)
    # Clutter near a wall can spoil an ordinary least-squares plane. Trim only a
    # minority of neighbours and refit. Keep balanced, separated faces distinct.
    median_index = np.maximum(count - 1, 0) // 2
    seed = finite[np.arange(len(points)), median_index]
    inliers = mask & (np.abs(offsets - seed[:, None]) <= 0.48)
    support = inliers.sum(1)
    plane_weights = inliers / np.maximum(support[:, None], 1)
    patch_center = np.einsum("nk,nkj->nj", plane_weights, rel)
    patch = rel - patch_center[:, None, :]
    patch_cov = (patch * plane_weights[:, :, None]).transpose(0, 2, 1) @ patch
    variances, patch_axes = np.linalg.eigh(patch_cov)
    patch_normal = patch_axes[:, :, 0]
    plane = (
        (support >= 12)
        & (support >= 0.68 * count)
        & (variances[:, 1] > 0.65)
        & (variances[:, 0] < 0.048 * variances[:, 1])
        & (variances[:, 0] < 0.09)
        & (np.abs((patch_normal * normal).sum(1)) > 0.97)
        & ~split
    )
    normal = patch_normal
    distance = np.einsum("ij,ij->i", patch_center, normal)
    plane &= np.abs(distance) <= 0.48
    steps = np.zeros_like(center)
    kinds = np.zeros(len(points), np.uint8)
    steps[plane] = distance[plane, None] * normal[plane]
    kinds[plane] = 1

    # For a circular column require support around its centre, not a short arc or a corner.
    xy = local[:, :, :2]
    z2 = np.sum(xy * xy, axis=2)
    rhs = 0.5 * np.einsum("nk,nk,nkj->nj", weights, z2, xy)
    a, b, d = cov[:, 0, 0], cov[:, 0, 1], cov[:, 1, 1]
    det = a * d - b * b
    conditioned = det > 0.06 * (a + d) ** 2
    denom = np.where(conditioned, det, 1)
    circle = (
        np.column_stack((d * rhs[:, 0] - b * rhs[:, 1], a * rhs[:, 1] - b * rhs[:, 0]))
        / denom[:, None]
    )
    radial = xy - circle[:, None, :]
    radii = np.linalg.norm(radial, axis=2)
    radius = np.sum(weights * radii, axis=1)
    axis_xy = center[:, :2] + circle
    from_axis = -axis_xy
    length = np.linalg.norm(from_axis, axis=1)
    far_side = np.einsum("nkj,nj->nk", radial, from_axis) < -0.35 * radius[:, None] ** 2
    cylinder = (
        (~plane)
        & (count >= 18)
        & conditioned
        & above_below
        & (radius >= 1.15)
        & (radius * size <= 1.0)
        & ((far_side & mask).sum(1) >= 3)
        & (np.max(np.where(mask, np.abs(radii - radius[:, None]), 0), axis=1) < 0.30)
        & (np.abs(length - radius) < 0.4)
    )
    steps[cylinder, :2] = (
        from_axis[cylinder] * (radius[cylinder] / length[cylinder] - 1)[:, None]
    )
    kinds[cylinder] = 2

    # Single-file returns along a post can support an axis, but cannot measure a diameter.
    # Multiple returns at the same height may be distinct faces of a wider object.
    levels = np.sort(np.where(mask, height, 1e6), axis=1)
    single_file = ~((np.diff(levels, axis=1) < 0.5) & (cut < count[:, None])).any(1)
    line = (
        (kinds == 0)
        & (count >= 7)
        & above_below
        & single_file
        & (np.abs(axis[:, 2]) > 0.96)
        & (values[:, :2].sum(1) < 0.10)
    )
    lateral = center - axis * np.einsum("ij,ij->i", center, axis)[:, None]
    line &= np.linalg.norm(lateral, axis=1) < 0.45
    steps[line] = lateral[line]
    kinds[line] = 3
    return steps * size, kinds


def align_surfaces(xyz, visits, size, origin):
    """Return candidates in the original cells. Counts and outer bounds stay reproducible."""
    corrected = xyz.copy()
    kinds = np.zeros(len(xyz), np.uint8)
    anchors = xyz[visits >= 2]
    if len(anchors) < 7:
        return corrected, kinds
    tree = cKDTree(anchors)
    k = min(48, len(anchors))
    radius = min(6.0 * size, 1.5)
    lo, hi = xyz.min(0), xyz.max(0)
    for first in range(0, len(xyz), 4096):
        last = min(first + 4096, len(xyz))
        points = xyz[first:last].astype(float)
        dist, ids = tree.query(points, k=k, distance_upper_bound=radius)
        neighbors = anchors[np.minimum(ids, len(anchors) - 1)]
        steps, labels = _surface_steps(points, neighbors, np.isfinite(dist), size)
        # Scale the whole correction vector to avoid bending it at a voxel face.
        norm = np.linalg.norm(steps, axis=1)
        scale = np.minimum(
            1.0, (0.9999 * min(0.35 * size, 0.04)) / np.maximum(norm, 1e-12)
        )
        lower = origin + np.floor((points - origin) / size) * size
        boundary = np.where(steps > 0, lower + size, lower)
        fraction = np.divide(
            boundary - points, steps, out=np.ones_like(steps), where=steps != 0
        )
        scale = np.minimum(scale, np.maximum(0.0, fraction.min(1)) * 0.9999)
        candidate = np.clip(points + scale[:, None] * steps, lo, hi).astype(xyz.dtype)
        candidate = np.where((points == lo) | (points == hi), points, candidate).astype(
            xyz.dtype
        )
        same_cell = (
            np.floor((candidate.astype(float) - origin) / size)
            == np.floor((points - origin) / size)
        ).all(1)
        moved = same_cell & (np.linalg.norm(candidate - points, axis=1) > 0.001 * size)
        corrected[first:last][moved] = candidate[moved]
        kinds[first:last][moved] = labels[moved]
    return corrected, kinds

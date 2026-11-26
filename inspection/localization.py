"""Cross-session planar registration to a fixed occupancy map without a shared odometry origin."""

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from .mapping import occupied_grid
from .geometry import compose, register, relative, rot, transform, voxel, wrap


def register_recording(cells, origin, res, scans, poses, pairs=()):
    # Split by consecutive stretches; only one half contributes to generating hypotheses.
    distance = np.r_[
        0, np.cumsum(np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1))
    ]
    mask = (np.floor(distance / 1.7).astype(int) % 2) == 0
    test_ids = []
    visited = set()
    for i in np.flatnonzero(~mask):
        loc = tuple(np.floor(poses[i, :2] / 1.2).astype(int))
        if loc not in visited:
            test_ids.append(i)
            visited.add(loc)
    fallback = dict(
        status="unverified",
        score=0.0,
        validation_sites=len(test_ids),
        alternative_score=None,
        reason="Недостаточно независимых участков для проверки регистрации.",
    )
    if len(test_ids) < 3:
        return poses, fallback

    def stable_points(take):
        indices = np.flatnonzero(take)
        cells_b, origin_b, res_b = occupied_grid(
            [scans[i] for i in indices], poses[indices], res
        )
        iy, ix = np.where(cells_b == 100)
        return origin_b + (np.column_stack([ix, iy]) + 0.5) * res_b

    train = voxel(stable_points(mask), 0.15)
    heldout = stable_points(~mask)
    if len(heldout) < 25:
        return poses, fallback
    repeatable = cKDTree(heldout)
    y, x = np.where(cells == 100)
    ref = origin + (np.column_stack([x, y]) + 0.5) * res
    if len(ref) < 30 or len(train) < 40:
        return poses, fallback
    center = np.mean(train, axis=0)
    offsets = train - center
    scale = max(res, 0.25)
    step = np.ceil(np.array(cells.shape[::-1]) * res / scale).astype(int) + 1
    occupied = np.zeros(step[::-1], np.uint8)
    coords = np.floor((ref - origin) / scale).astype(int)
    occupied[coords[:, 1], coords[:, 0]] = 1
    field = np.exp(
        -0.5 * (distance_transform_edt(1 - occupied) * scale / 0.22) ** 2
    ).astype(np.float32)
    radius = int(np.ceil(np.linalg.norm(offsets, axis=1).max() / scale)) + 1
    if radius > 900:
        return poses, {
            **fallback,
            "reason": "Протяжённость записи превышает предел поиска гипотез.",
        }
    proposals = []
    for yaw in np.arange(-np.pi, np.pi, np.deg2rad(10.0)):
        delta = np.rint((offsets @ rot(yaw).T) / scale).astype(int)
        shape = np.zeros((2 * radius + 1,) * 2, np.float32)
        shape[delta[:, 1] + radius, delta[:, 0] + radius] = 1
        response = cv2.filter2D(
            field, -1, shape / shape.sum(), borderType=cv2.BORDER_CONSTANT
        )
        for _ in range(2):
            iy, ix = np.unravel_index(np.argmax(response), response.shape)
            score = response[iy, ix]
            xy = origin + (np.array([ix, iy]) + 0.5) * scale
            proposals.append((float(score), np.r_[xy - center @ rot(yaw).T, yaw]))
            cv2.circle(response, (ix, iy), max(2, int(1.4 / scale)), -1, -1)
    for _, current, reference in pairs:
        # Position + heading provides a hypothesis only. Laser registration and independent evidence decide acceptance.
        seed = compose(reference, relative(current, np.zeros(3)))
        proposals.append((1.01, seed))
    tree = cKDTree(ref)
    candidates = []
    # Preserve alternate bearings as well as the highest-scoring match.
    for _, seed in sorted(proposals, key=lambda item: -item[0])[
        : max(32, len(pairs) + 32)
    ]:
        fit, score, err = register(train, tree, seed, radius=0.62, iterations=18)
        centered = transform(center, fit)
        if any(
            np.linalg.norm(centered - transform(center, other)) < 0.65
            and abs(wrap(fit[2] - other[2])) < 0.16
            for _, other, _ in candidates
        ):
            continue
        candidates.append((score, fit, err))
    candidates.sort(key=lambda item: -item[0])
    if not candidates:
        return poses, fallback

    def validate(fit):
        scores = []
        for i in test_ids:
            pts = transform(scans[i].hits, poses[i])
            pts = pts[repeatable.query(pts)[0] < 0.14]
            if len(pts) < 10:
                continue
            dd = tree.query(transform(pts, fit))[0]
            scores.append(float(np.mean(dd < 0.29)))
        return (float(np.mean(scores)) if scores else 0.0), sum(
            s > 0.52 for s in scores
        )

    # Choose using training scores only, then test on withheld stretches of the second recording.
    def image_support(fit):
        locations = set()
        for _, p, q in pairs:
            point = compose(fit, p)
            if (
                np.linalg.norm(point[:2] - q[:2]) < 1.7
                and abs(wrap(point[2] - q[2])) < 0.52
            ):
                locations.add(tuple(np.floor(q[:2] / 1.8).astype(int)))
        return len(locations)

    if pairs:
        candidates.sort(
            key=lambda item: -(item[0] + min(image_support(item[1]), 6) * 0.026)
        )
    train_score, best, err = candidates[0]
    valid, sites = validate(best)
    alt = max(
        (min(score, validate(fit)[0]) for score, fit, _ in candidates[1:]), default=0.0
    )
    success = (
        train_score >= 0.56
        and valid >= 0.58
        and sites >= 3
        and sites / len(test_ids) >= 0.55
        and min(train_score, valid) - alt >= 0.045
    )
    visual_sites = image_support(best)
    if visual_sites >= 3 and train_score >= 0.48 and valid >= 0.54 and sites >= 3:
        # Alternative transforms must also explain the independent, spatially distributed camera places.
        rivals = [(score, image_support(fit)) for score, fit, _ in candidates[1:]]
        success = success or not any(
            s >= train_score - 0.045 and n >= visual_sites - 1 for s, n in rivals
        )
    status = "localized" if success else "unverified"
    diagnostics = dict(
        status=status,
        score=round(valid, 4),
        train_score=round(train_score, 4),
        validation_sites=len(test_ids),
        supported_sites=sites,
        visual_matches=len(pairs),
        visual_sites=visual_sites,
        alternative_score=round(alt, 4),
        median_residual_m=round(float(err), 4),
        transform_xy_yaw=best.tolist(),
        reason="Геометрия подтверждена независимыми участками."
        if success
        else "Недостаточное совпадение или неоднозначная геометрия.",
    )
    aligned = np.array([compose(best, p) for p in poses])
    if success:
        # Track against the fixed reference after global initialization. Reject large or unsupported jumps.
        delta = np.zeros(3)
        for i, guess in enumerate(aligned.copy()):
            estimate = compose(guess, delta)
            fit, s, med = register(
                scans[i].hits, tree, estimate, radius=0.36, iterations=6
            )
            correction = relative(guess, fit)
            if (
                s > 0.57
                and med < 0.17
                and np.linalg.norm(correction[:2]) < 0.35
                and abs(correction[2]) < 0.10
            ):
                delta = 0.75 * delta + 0.25 * correction
            else:
                delta *= 0.9
            aligned[i] = compose(guess, delta)
    return aligned, diagnostics

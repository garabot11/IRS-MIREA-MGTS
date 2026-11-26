from scipy.ndimage import distance_transform_edt
import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .geometry import compose, pose_graph, register, relative, transform, voxel, wrap


def prepare_poses(data):
    scans = data["scans"]
    times = np.array([s.t for s in scans])
    odom = data["odom"]
    initial = np.zeros((len(times), 3))
    zrp = np.zeros((len(times), 3))
    if len(odom) >= 2:
        t = odom[:, 0]
        euler = Rotation.from_quat(odom[:, 4:8]).as_euler("xyz")
        for i, series in enumerate([odom[:, 1], odom[:, 2], np.unwrap(euler[:, 2])]):
            initial[:, i] = np.interp(times, t, series)
        for i, series in enumerate(
            [odom[:, 3], np.unwrap(euler[:, 0]), np.unwrap(euler[:, 1])]
        ):
            zrp[:, i] = np.interp(times, t, series)
        nearest = np.searchsorted(t, times).clip(1, len(t) - 1)
        age = np.minimum(abs(t[nearest] - times), abs(t[nearest - 1] - times))
        if np.max(age) > 0.5:
            data["warnings"].append(
                "В одометрии есть пропуски более 0,5 с. Позы в промежутках интерполированы."
            )
    else:
        for i in range(1, len(times)):
            delta, score, _ = register(
                scans[i].hits, scans[i - 1].hits, np.zeros(3), radius=0.8
            )
            if score > 0.45:
                initial[i] = compose(initial[i - 1], delta)
            else:
                initial[i] = initial[i - 1]
    return times, initial, zrp


def submap(scans, initial, keys, k, before=1):
    pts = []
    # Pool nearby revolutions even on stationary sections where keyframes are far apart in time.
    idx = keys[k]
    for j in range(max(0, idx - 4), min(len(scans), idx + 5)):
        pts.append(transform(scans[j].hits, relative(initial[idx], initial[j])))
    return voxel(np.vstack(pts), 0.095)


def parallel_surfaces(points):
    """Flag a scan whose dominant local normals constrain only one translation direction."""
    if len(points) < 20:
        return False
    points = points[:: max(1, len(points) // 320)]
    neighbors = points[cKDTree(points).query(points, k=6)[1]]
    centered = neighbors - neighbors.mean(axis=1, keepdims=True)
    cov = np.swapaxes(centered, 1, 2) @ centered
    vals, vecs = np.linalg.eigh(cov)
    reliable = (vals[:, 0] < 0.08 * vals[:, 1]) & (vals[:, 1] > 0.006)
    if reliable.sum() < max(15, 0.6 * len(points)):
        return False
    normals = vecs[reliable, :, 0]
    spectrum = np.linalg.eigvalsh(normals.T @ normals)
    return bool(spectrum[0] < 0.055 * spectrum[1])


def loop_hypothesis(source, target, seed):
    """Search a bounded translation/yaw neighborhood before ICP when odometry has drifted."""
    base, score, err = register(source, target, seed)
    if score > 0.86 and err < 0.075:
        return base, score, err
    res = 0.14
    low = target.min(0) - 0.42
    dims = np.ceil((target.max(0) - low + 0.42) / res).astype(int) + 1
    raster = np.zeros(dims[::-1], np.uint8)
    xy = np.floor((target - low) / res).astype(int)
    raster[xy[:, 1], xy[:, 0]] = 1
    field = np.exp(-0.5 * (res * distance_transform_edt(1 - raster) / 0.14) ** 2)
    offsets = np.array([(x, y) for x in range(-9, 10, 2) for y in range(-9, 10, 2)])
    rough = []
    probe = source[:: max(1, len(source) // 220)]
    for dyaw in (-0.24, -0.12, 0.0, 0.12, 0.24):
        rotated = seed + np.array([0, 0, dyaw])
        cells = np.floor((transform(probe, rotated) - low) / res).astype(int)
        uv = cells[None, :, :] + offsets[:, None, :]
        x, y = uv[:, :, 0], uv[:, :, 1]
        valid = (x >= 0) & (y >= 0) & (x < dims[0]) & (y < dims[1])
        scores = (
            field[np.clip(y, 0, dims[1] - 1), np.clip(x, 0, dims[0] - 1)] * valid
        ).mean(axis=1)
        i = scores.argmax()
        rough.append((scores[i], rotated + np.r_[offsets[i] * res, 0.0]))
    for _, guess in sorted(rough, key=lambda k: -k[0])[:2]:
        trial, s, e = register(source, target, guess)
        if s - 0.48 * e > score - 0.48 * err:
            base, score, err = trial, s, e
    return base, score, err


def optimize(data, progress):
    scans = data["scans"]
    times, prior, zrp = prepare_poses(data)
    keys = [0]
    for i in range(1, len(scans)):
        diff = relative(prior[keys[-1]], prior[i])
        if (
            np.linalg.norm(diff[:2]) > 0.32
            or abs(diff[2]) > 0.19
            or times[i] - times[keys[-1]] > 2.6
        ):
            keys.append(i)
    if keys[-1] != len(scans) - 1:
        keys.append(len(scans) - 1)
    key = prior[keys]
    edges = []
    accepted = 0
    weak = 0
    loops = []
    rejected = 0
    imu_edges = 0
    weak_keys = np.zeros(len(keys), bool)
    degenerate_edges = 0
    imu = data["imu"]
    arclength = np.r_[0, np.cumsum(np.linalg.norm(np.diff(key[:, :2], axis=0), axis=1))]
    imu_integral = None
    gyro_bias = None
    if len(imu) > 1:
        # A stationary interval provides zero angular-rate observations independent of IMU orientation.
        speed = np.linalg.norm(np.diff(prior[:, :2], axis=0), axis=1) / np.maximum(
            0.01, np.diff(times)
        )
        turn = np.abs(wrap(np.diff(prior[:, 2]))) / np.maximum(0.01, np.diff(times))
        stationary = (speed < 0.025) & (turn < 0.035)
        which = np.searchsorted(times[1:], imu[:, 0]).clip(0, len(stationary) - 1)
        quiet = stationary[which] & (np.abs(imu[:, 1]) < 0.10)
        rates = imu[:, 1].copy()
        if len(data["odom"]) and np.sum(quiet) >= 30:
            gyro_bias = float(np.median(rates[quiet]))
            rates -= gyro_bias
        dt = np.diff(imu[:, 0])
        imu_integral = np.r_[
            0,
            np.cumsum(
                np.where((dt > 0) & (dt < 0.25), dt, 0) * (rates[1:] + rates[:-1]) / 2
            ),
        ]
    for j in range(1, len(keys)):
        a, b = keys[j - 1], keys[j]
        delta = relative(prior[a], prior[b])
        still = np.linalg.norm(delta[:2]) < 0.025 and abs(delta[2]) < 0.025
        edges.append(
            (
                j - 1,
                j,
                delta,
                [38.0, 38.0, 45.0] if still and len(data["odom"]) else [9.0, 9.0, 20.0],
            )
        )
        if (
            imu_integral is not None
            and imu[0, 0] <= times[a] + 0.12
            and imu[-1, 0] >= times[b] - 0.12
        ):
            dt_yaw = np.diff(np.interp([times[a], times[b]], imu[:, 0], imu_integral))[
                0
            ]
            if abs(wrap(dt_yaw - delta[2])) < 0.10:
                edges.append((j - 1, j, np.r_[0.0, 0.0, dt_yaw], [0.0, 0.0, 7.0]))
                imu_edges += 1
        prev = voxel(scans[a].hits, 0.09)
        src = voxel(scans[b].hits, 0.09)
        degenerate = parallel_surfaces(prev)
        degenerate_edges += int(degenerate)
        pose, score, err = register(src, prev, delta)
        diff = relative(delta, pose)
        if (
            not degenerate
            and score > 0.57
            and err < 0.16
            and np.linalg.norm(diff[:2]) < 0.32
            and abs(diff[2]) < 0.13
        ):
            edges.append((j - 1, j, pose, [4.0, 4.0, 10.0]))
            accepted += 1
        else:
            weak += 1
            weak_keys[j] = True
        if (j % 3 and j != len(keys) - 1) or j < 8:
            continue
        cand = np.where(
            (times[b] - times[np.array(keys[:j])] > 22)
            & (arclength[j] - arclength[:j] > 6)
        )[0]
        if not len(cand):
            continue
        dist = np.linalg.norm(key[cand, :2] - key[j, :2], axis=1)
        for k in cand[np.argsort(dist)[:3]]:
            rel = relative(key[k], key[j])
            if np.linalg.norm(rel[:2]) > 2.3:
                continue
            earlier = submap(scans, prior, keys, k)
            current = submap(scans, prior, keys, j)
            fit, overlap, err = loop_hypothesis(current[::2], earlier, rel)
            # Held-out points and reciprocal support reduce false closures along parallel walls.
            check = cKDTree(earlier).query(transform(current[1::2], fit))[0]
            reverse = cKDTree(transform(current, fit)).query(earlier)[0]
            shift = relative(rel, fit)
            spread = np.linalg.eigvalsh(np.cov(earlier.T))
            if (
                overlap > 0.68
                and np.mean(check < 0.23) > 0.7
                and np.mean(reverse < 0.26) > 0.60
                and err < 0.15
                and np.linalg.norm(shift[:2])
                < (1.85 if overlap > 0.83 and err < 0.11 else 0.85)
                and abs(shift[2]) < 0.34
                and spread[0] > max(0.035, 0.025 * spread[1])
                and not parallel_surfaces(earlier)
            ):
                edges.append((int(k), j, fit, [7.0, 7.0, 14.0]))
                loops.append([int(k), j])
                break
            rejected += 1
    progress(0.64, "Оптимизация графа поз и замыканий")
    refined, sigma = pose_graph(key, edges)
    # Interpolate the correction separately from robot motion to preserve measured odometry between keys.
    corr = np.array(
        [compose(new, relative(old, np.zeros(3))) for old, new in zip(key, refined)]
    )
    corr[:, 2] = np.unwrap(corr[:, 2])
    continuous = np.column_stack(
        [np.interp(times, times[keys], corr[:, d]) for d in range(3)]
    )
    final = np.array([compose(c, p) for c, p in zip(continuous, prior)])
    diag = dict(
        keyframes=len(keys),
        parallel_surface_edges=degenerate_edges,
        scan_edges=accepted,
        weak_scan_edges=weak,
        loop_closures=len(loops),
        rejected_loop_candidates=rejected,
        imu_edges=imu_edges,
        raw_distance_m=float(
            np.linalg.norm(np.diff(prior[:, :2], axis=0), axis=1).sum()
        ),
        raw_end_gap_m=float(np.linalg.norm(prior[-1, :2] - prior[0, :2])),
        max_graph_correction_m=float(
            np.linalg.norm(refined[:, :2] - key[:, :2], axis=1).max()
        ),
    )
    right = np.searchsorted(times[keys], times).clip(0, len(keys) - 1)
    left = np.maximum(0, right - 1)
    nearest = np.where(
        np.abs(times - times[np.asarray(keys)[left]])
        < np.abs(times - times[np.asarray(keys)[right]]),
        left,
        right,
    )
    uncertainty = np.interp(times, times[keys], sigma)
    diag["drift"] = dict(
        gyro_bias_rad_s=gyro_bias,
        peak_uncertainty_m=float(uncertainty.max()),
        description="Внутренняя плановая неопределённость графа (корень суммы дисперсий X и Y). Оценка относится к внутреннему графу прохода до регистрации с внешней картой. Шум связей задан моделью; систематическая ошибка без внешнего эталона неизвестна.",
        samples=[
            dict(
                t=round(float(max(0, t)), 5),
                uncertainty_m=round(float(u), 4),
                weak=bool(weak_keys[k]),
            )
            for t, u, k in zip(times, uncertainty, nearest)
        ],
    )
    return times, final, zrp, diag


def _grid_extent(observations, res):
    low = np.min(
        [
            np.min(np.vstack((hits, clear, base)), axis=0)
            for hits, clear, base in observations
        ],
        axis=0,
    )
    high = np.max(
        [
            np.max(np.vstack((hits, clear, base)), axis=0)
            for hits, clear, base in observations
        ],
        axis=0,
    )
    origin = np.floor((low - 0.65) / res) * res
    dims = np.ceil((high + 0.65 - origin) / res).astype(int)
    if int(dims[0]) * int(dims[1]) > 16_000_000:
        raise ValueError(
            "Размер карты превышает 16 млн ячеек. Проверьте единицы и выбросы одометрии."
        )
    return origin, dims


def _filter_grid_observations(observations, scans, res, diagnostics):
    # Repeated hits support a thin post even if its neighbours are all empty.
    # Use the ordinary hit/free inverse sensor model to distinguish a persistent
    # obstacle from a few elevated samples that share its horizontal bearing.
    all_rays = [
        (hits, np.vstack((hits, clear)), base) for hits, clear, base in observations
    ]
    preliminary, origin, _ = _integrate_grid(all_rays, res)
    height, width = preliminary.shape
    witnesses = np.zeros((height, width), np.uint8)
    last_seen = np.full((height, width), -1, np.int32)
    windows = np.asarray([int(np.floor((s.t - scans[0].t) / 0.5)) for s in scans])
    index_by_scan = []
    for (hits, _, _), window in zip(observations, windows):
        ij = np.floor((hits - origin) / res).astype(int)
        index = ij[:, 1] * width + ij[:, 0]
        index_by_scan.append(index)
        cells = np.unique(index)
        new = cells[last_seen.flat[cells] != window]
        witnesses.flat[new] = np.minimum(2, witnesses.flat[new] + 1)
        last_seen.flat[new] = window
    count = cv2.filter2D(
        (witnesses > 0).astype(np.uint8),
        -1,
        np.ones((3, 3), np.uint8),
        borderType=cv2.BORDER_CONSTANT,
    )
    supported = (witnesses >= 2) | (count >= 2)
    filtered = []
    removed = blocked = 0

    def bearings(endpoints, base):
        delta = endpoints - base
        bins = (
            np.floor(np.degrees(np.arctan2(delta[:, 1], delta[:, 0])) + 180).astype(int)
            % 360
        )
        return bins, np.linalg.norm(delta, axis=1)

    for (hits, clear, base), index in zip(observations, index_by_scan):
        keep = supported.flat[index]
        removed += int((~keep).sum())
        bins, ranges = bearings(hits, base)
        nearest = np.full(360, np.inf)
        blocking = preliminary.flat[index] == 100
        np.minimum.at(nearest, bins[blocking], ranges[blocking])
        # A return from farther away (often the floor or wall beyond a table)
        # can locate a surface. It cannot prove the floor behind a nearer post is free.
        has_sightline = ranges <= nearest[bins] + 1e-7
        ray_hits = hits[keep & has_sightline]
        free_bins, free_ranges = bearings(clear, base)
        visible = free_ranges <= nearest[free_bins] + 1e-7
        blocked += int((~visible).sum()) + int(keep.sum()) - len(ray_hits)
        filtered.append(
            (
                hits[keep & (has_sightline | blocking)],
                np.vstack((ray_hits, clear[visible])),
                base,
            )
        )
    if diagnostics is not None:
        diagnostics.update(
            algorithm="supported_endpoints_first_return_v1",
            unsupported_hit_returns=removed,
            occluded_clear_rays=blocked,
            angular_bin_deg=1,
            support_window_s=0.5,
        )
    return filtered


def occupied_grid(scans, poses, res=0.1, *, diagnostics=None):
    observations = []
    for s, p in zip(scans, poses):
        observations.append((transform(s.hits, p), transform(s.clear, p), p[:2]))
    filtered = _filter_grid_observations(observations, scans, res, diagnostics)
    return _integrate_grid(filtered, res)


def _integrate_grid(observations, res):
    origin, (width, height) = _grid_extent(observations, res)
    evidence = np.zeros((height, width), np.float32)
    seen = np.zeros((height, width), np.uint16)

    def uv(arr):
        return np.floor((arr - origin) / res).astype(int)

    for hits, free, base in observations:
        h, f, b = uv(hits), uv(free), uv(base)
        if not len(f) and not len(h):
            continue
        extent = np.vstack((h, f, b))
        low = extent.min(0)
        high = extent.max(0)
        ox, oy = low
        x, y = high
        footprint = np.zeros((y - oy + 1, x - ox + 1), np.uint8)
        if len(f):
            rays = np.empty((len(f), 2, 2), dtype=np.int32)
            rays[:, 0] = b - low
            rays[:, 1] = f - low
            cv2.polylines(footprint, rays, False, 1)
        patch = evidence[oy : y + 1, ox : x + 1]
        patch -= 0.43 * footprint
        if len(h):
            occupied_idx = np.unique(h[:, 1] * width + h[:, 0])
            seen.flat[occupied_idx] = np.minimum(seen.flat[occupied_idx] + 1, 2)
            evidence.flat[occupied_idx] += 1.7
        np.clip(patch, -12, 12, out=patch)
    cells = np.full((height, width), -1, np.int8)
    cells[evidence <= -0.5] = 0
    cells[(evidence >= 0.7) & (seen >= 2)] = 100
    return cells, origin, res

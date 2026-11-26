"""Measured XYZ surfaces in map coordinates, independent of the 2D occupancy representation."""

from pathlib import Path
import threading

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .recording import open_recording
from .map_filter import filter_supported_voxels, normalize_map_filter
from .surface_alignment import ALGORITHM as ALIGNMENT_ALGORITHM, align_surfaces
from .surface_camera import calibrated_views, preserve_image_edges
from .sensors import cloud_points, read_static_transforms, static_transform, stamp

from .voxel_cloud import DEFAULT_VOXEL_M, cloud_records, ply_header

COMPUTE_LOCK = threading.RLock()
MAX_CELLS = 2_400_000
MAX_WORK_BYTES = 1536 * 1024**2
MAX_PREVIEW = 100_000
WINDOW_S = 0.5
MAX_SCAN_POINTS = 24_000
MAX_RANGE_M = 80.0
OFFSET = 1 << 20


class CoarserGridRequired(Exception):
    pass


def _reduce(keys, xyz, counts=None, views=None):
    """Sort once. Each result contains point sums, return counts, and distinct survey windows."""
    order = np.argsort(keys, kind="stable")
    ordered = keys[order]
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    points = np.add.reduceat(xyz[order], starts, axis=0)
    if counts is None:
        counts = np.diff(np.r_[starts, len(keys)])
        visits = np.ones(len(starts), np.int32)
    else:
        counts = np.add.reduceat(counts[order], starts)
        visits = np.add.reduceat(views[order], starts)
    return ordered[starts], points, counts, visits


class Survey:
    def __init__(self, origin, res, max_cells=None):
        self.max_cells = MAX_CELLS if max_cells is None else max_cells
        self.origin = origin
        self.res = res
        self.window = None
        self.parts = []
        self.chunk = []
        self.chunk_size = 0
        self.state = None
        self.windows = 0
        self.points = 0

    def add(self, xyz, t):
        window = int(np.floor(t / WINDOW_S))
        if self.window != window:
            self.finish_window()
            self.window = window
        self.parts.append(xyz)
        self.points += len(xyz)

    def finish_window(self):
        if not self.parts:
            return
        xyz = np.concatenate(self.parts)
        self.parts.clear()
        cells = np.floor((xyz - self.origin) / self.res).astype(np.int64) + OFFSET
        if np.any(cells < 0) or np.any(cells >= 2 * OFFSET):
            raise CoarserGridRequired
        keys = (cells[:, 0] << 42) | (cells[:, 1] << 21) | cells[:, 2]
        entry = _reduce(keys, xyz)
        self.chunk.append(entry)
        self.chunk_size += len(entry[0])
        self.windows += 1
        if self.chunk_size >= 120_000:
            self.merge()

    def merge(self):
        if not self.chunk:
            return
        if self.state is not None:
            self.chunk.append(self.state)
        if sum(len(part[0]) for part in self.chunk) * 192 > MAX_WORK_BYTES:
            raise MemoryError("Превышен бюджет памяти измеренной карты")
        self.state = _reduce(*(np.concatenate(column) for column in zip(*self.chunk)))
        self.chunk.clear()
        self.chunk_size = 0
        if len(self.state[0]) > self.max_cells:
            raise CoarserGridRequired

    def result(self):
        self.finish_window()
        self.merge()
        if self.state is None:
            return None
        _, sums, counts, views = self.state
        return (sums / counts[:, None]).astype("<f4"), np.minimum(views, 65535).astype(
            "<u2"
        )


def _pose_at(t, times, xyz, quaternions):
    j = int(np.searchsorted(times, t, side="right"))
    if j == 0:
        return xyz[0], quaternions[0]
    if j == len(times):
        return xyz[-1], quaternions[-1]
    w = (t - times[j - 1]) / (times[j] - times[j - 1])
    q0, q1 = quaternions[j - 1 : j + 1]
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot
    if dot < 0.9995:
        angle = np.arccos(np.clip(dot, -1.0, 1.0))
        q = (np.sin((1.0 - w) * angle) * q0 + np.sin(w * angle) * q1) / np.sin(angle)
    else:
        q = (1.0 - w) * q0 + w * q1
    return (1.0 - w) * xyz[j - 1] + w * xyz[j], q


def _scan_scene(reader, selected, graph, base, trajectory, res, progress, max_cells):
    times = trajectory[:, 0] - reader.start_time * 1e-9
    gap = max(1.0, 3.0 * np.median(np.diff(times)))
    xyz, quats = trajectory[:, 1:4], trajectory[:, 4:8]
    cache = {}
    survey = Survey(xyz[0], res, max_cells)
    stats = dict(
        scans=0, scans_used=0, skipped_by_time=0, sampled_scans=0, returns_read=0
    )
    next_update = -1
    last_stamp = -np.inf
    for c, tick, raw in reader.messages(connections=selected):
        m = reader.deserialize(raw, c.msgtype)
        frame = m.header.frame_id.strip("/")
        if frame not in cache:
            tf = static_transform(graph, frame, base)
            if tf is None:
                raise ValueError(
                    f"Нет статической привязки {frame} → {base} для 3D-карты."
                )
            cache[frame] = tf
        log = (tick - reader.start_time) * 1e-9
        t = stamp(m, log, reader.start_time * 1e-9)
        stats["scans"] += 1
        # Limit the input rate per interval and reject backwards header jumps before counting visits.
        if t - last_stamp < 1 / 30:
            stats["skipped_by_time"] += 1
            continue
        last_stamp = t
        if log >= next_update:
            progress(
                min(1.0, log / max(0.01, times[-1])),
                "Совмещение XYZ с траекторией и объединение вокселей",
            )
            next_update = log + 4.0
        j = np.searchsorted(times, t)
        if (
            t < times[0] - 0.5
            or t > times[-1] + 0.5
            or (0 < j < len(times) and times[j] - times[j - 1] > gap)
        ):
            stats["skipped_by_time"] += 1
            continue
        points = cloud_points(m)
        stats["returns_read"] += len(points)
        if len(points) > MAX_SCAN_POINTS:
            stride = int(np.ceil(len(points) / MAX_SCAN_POINTS))
            points = points[stats["scans"] % stride :: stride]
            stats["sampled_scans"] += 1
        # Reject near-body returns and out-of-range spikes in the sensor frame.
        norm2 = np.einsum("ij,ij->i", points, points)
        points = points[(norm2 > 0.16**2) & (norm2 <= MAX_RANGE_M**2)]
        if not len(points):
            continue
        pos, q = _pose_at(t, times, xyz, quats)
        rb = Rotation.from_quat(q).as_matrix()
        tf = cache[frame]
        survey.add(points @ (rb @ tf[:3, :3]).T + (pos + rb @ tf[:3, 3]), t)
        stats["scans_used"] += 1
    result = survey.result()
    return result, dict(
        stats, integrated_returns=survey.points, observation_windows=survey.windows
    )


def _save_surface(out, xyz, visits):
    z = xyz[:, 2]
    low, high = np.percentile(z, [1, 99])
    quant = np.asarray(
        np.clip((z - low) / max(0.001, high - low), 0, 1) * 255, dtype=np.uint8
    )
    rgb = (
        cv2.applyColorMap(quant.reshape(-1, 1), cv2.COLORMAP_VIRIDIS)
        .reshape(-1, 3)[:, ::-1]
        .copy()
    )
    cloud = cloud_records(xyz, rgb, visits)
    with (out / "map3d.ply").open("wb") as stream:
        stream.write(
            ply_header(
                len(cloud),
                "units metres; color from height; observations per 0.5 s interval",
            )
        )
        cloud.tofile(stream)
    idx = np.linspace(0, len(xyz) - 1, min(len(xyz), MAX_PREVIEW), dtype=np.int64)
    np.savez_compressed(
        out / "map3d.npz", xyz=xyz[idx], rgb=rgb[idx], observations=visits[idx]
    )
    return len(idx)


def unavailable(reason, topic=None):
    return dict(status="unavailable", reason=reason, source_topic=topic, files={})


def build_map3d(
    source,
    result,
    trajectory_dir,
    out=None,
    progress=None,
    voxel_size_m=DEFAULT_VOXEL_M,
    *,
    exact=False,
    max_cells=None,
    bind_source=False,
):
    """Build a measured layer. New jobs also bind their recording to the publication."""
    with COMPUTE_LOCK:
        if bind_source:
            from .voxel_levels import recording_members, recording_sha, signature

            members = recording_members(Path(source))
            before = signature(members)
        try:
            info = _build_map3d(
                source,
                result,
                trajectory_dir,
                out,
                progress,
                voxel_size_m,
                exact,
                MAX_CELLS if max_cells is None else max_cells,
            )
        except MemoryError:
            info = dict(
                unavailable("Превышен бюджет памяти. Выберите более крупный шаг."),
                code="resource_limit",
            )
        if bind_source and info["status"] == "ready":
            info["source_manifest_sha256"] = recording_sha(members)
            if signature(members) != before:
                info = dict(
                    unavailable("Исходная запись изменилась во время обработки."),
                    code="inputs_changed",
                )
        if info["status"] != "ready" and exact:
            info.update(requested_voxel_size_m=voxel_size_m, voxel_size_m=None)
            info.setdefault("code", "unsupported_3d")
        return info


def _build_map3d(
    source, result, trajectory_dir, out, progress, voxel_size_m, exact, max_cells
):
    """Build assets from the selected 3D channel using the *published* time-stamped trajectory."""
    out = Path(out or trajectory_dir)
    progress = progress or (lambda *_: None)
    lidar = next((c for c in result["channels"] if c.get("role") == "lidar"), {})
    topic = lidar.get("topic")
    if lidar.get("type") != "sensor_msgs/msg/PointCloud2":
        return unavailable(
            "Выбранный лидар предоставляет плоский LaserScan. Для объёмной карты нужен канал PointCloud2 с измеренными XYZ.",
            topic,
        )
    if (
        result.get("mode") == "localization"
        and result["diagnostics"].get("registration", {}).get("status") != "localized"
    ):
        return unavailable(
            "Совмещение прохода с картой не подтверждено. Можно построить собственную карту этой записи.",
            topic,
        )
    trajectory = np.loadtxt(Path(trajectory_dir) / "trajectory.tum", ndmin=2)
    if (
        len(trajectory) < 2
        or trajectory.shape[1] != 8
        or not np.isfinite(trajectory).all()
        or np.any(np.diff(trajectory[:, 0]) <= 0)
        or np.any(np.linalg.norm(trajectory[:, 4:], axis=1) < 1e-6)
    ):
        return unavailable(
            "Для привязки объёмных измерений требуется корректная траектория с монотонными метками времени.",
            topic,
        )
    base = result["diagnostics"]["base_frame"].strip("/")
    previous = result.get("map3d", {})
    # Archived publications keep their geometry when a finer level is replayed.
    align = previous.get("status") != "ready" or (
        previous.get("alignment", {}).get("algorithm") == ALIGNMENT_ALGORITHM
    )
    with open_recording(source) as reader:
        selected = [
            c
            for c in reader.connections
            if c.topic == topic and c.msgtype == lidar["type"]
        ]
        if not selected:
            return unavailable("В исходной записи не найден выбранный 3D-лидар.", topic)
        graph = read_static_transforms(reader)
        res = voxel_size_m
        for _ in range(7):
            try:
                cloud, stats = _scan_scene(
                    reader, selected, graph, base, trajectory, res, progress, max_cells
                )
                break
            except CoarserGridRequired:
                if exact:
                    return dict(
                        unavailable(
                            "Запрошенный шаг превышает допустимое число ячеек или диапазон координат карты.",
                            topic,
                        ),
                        code="exact_level_limit",
                    )
                res *= 2.0
            except ValueError as exc:
                return unavailable(str(exc), topic)
        else:
            return unavailable(
                "Область съёмки превышает допустимую ёмкость карты.", topic
            )
        if align and cloud is not None:
            views, camera_status = calibrated_views(
                reader, result, trajectory_dir, trajectory, graph, base, _pose_at
            )
    if cloud is None:
        return unavailable(
            "Нет конечных точек XYZ в диапазоне дальности и времени траектории.", topic
        )
    xyz, visits = cloud
    # Published jobs without filter metadata must still replay their original PLY byte for byte.
    legacy_replay = (
        result.get("map3d", {}).get("status") == "ready" and "map_filter" not in result
    )
    if not legacy_replay:
        settings = normalize_map_filter(result.get("map_filter"))
        keep, stats["filter"] = filter_supported_voxels(xyz, visits, res, **settings)
        if not keep.any():
            return dict(
                unavailable(
                    "Нет вокселей с пространственной или повторной поддержкой.", topic
                ),
                code="insufficient_support",
                **stats,
            )
        xyz, visits = xyz[keep], visits[keep]
    files = dict(ply="map3d.ply", preview="map3d.npz")
    if align:
        progress(.92, "Выравнивание измеренных поверхностей и проверка кромок")
        candidate, kinds = align_surfaces(xyz, visits, res, trajectory[0, 1:4])
        protected = preserve_image_edges(xyz, candidate, kinds, views, res)
        rows = np.flatnonzero(kinds)
        shifts = np.linalg.norm(candidate[rows] - xyz[rows], axis=1)
        files["alignment"] = "map3d-alignment.npz"
        np.savez_compressed(out / files["alignment"], index=rows.astype("<i4"),
                            measured_xyz=xyz[rows], kind=kinds[rows])
        stats["alignment"] = dict(
            algorithm=ALIGNMENT_ALGORITHM, moved_points=len(rows),
            planar_points=int(np.sum(kinds == 1)), cylinder_points=int(np.sum(kinds == 2)),
            slender_post_points=int(np.sum(kinds == 3)),
            max_shift_m=float(shifts.max(initial=0)), limit_m=min(.35 * res, .04),
            camera=dict(status=camera_status, views=len(views), protected_points=protected),
        )
        xyz = candidate
    preview = _save_surface(out, xyz, visits)
    pose_source = result["diagnostics"].get("drift", {}).get("prior_source")
    return dict(
        status="ready",
        frame=result.get("frame", "map"),
        source_topic=topic,
        points=len(xyz),
        preview_points=preview,
        repeated_points=int(np.count_nonzero(visits >= 2)),
        voxel_size_m=res,
        requested_voxel_size_m=voxel_size_m,
        window_s=WINDOW_S,
        max_range_m=MAX_RANGE_M,
        bounds=dict(min=xyz.min(0).tolist(), max=xyz.max(0).tolist()),
        pose_model="planar_scan_odometry"
        if pose_source == "scan_odometry"
        else "planar_slam_with_recorded_attitude",
        color="height",
        files=files,
        **stats,
    )

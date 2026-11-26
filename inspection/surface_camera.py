"""Use synchronized raw optical images to preserve measured edges during voxel alignment."""

from hashlib import sha256
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .sensors import static_transform


def _intrinsics(info, size):
    k = np.asarray(info.k, dtype=float).reshape(3, 3)
    d = np.asarray(info.d, dtype=float)
    w, h = size
    roi = info.roi
    if (
        (w, h) != (info.width, info.height)
        or min(w, h) <= 0
        or info.binning_x > 1
        or info.binning_y > 1
        or roi.x_offset
        or roi.y_offset
        or roi.width not in (0, w)
        or roi.height not in (0, h)
        or info.distortion_model not in ("plumb_bob", "rational_polynomial")
        or len(d) not in (0, 4, 5, 8)
        or not np.isfinite(k).all()
        or not np.isfinite(d).all()
        or min(k[0, 0], k[1, 1]) <= 0
        or not np.allclose(k[2], [0, 0, 1])
    ):
        return None
    return k, d


def calibrated_views(reader, result, directory, trajectory, graph, base, pose_at):
    camera = next((c for c in result["channels"] if c.get("role") == "camera"), {})
    frames = result.get("camera", {}).get("frames", [])
    if not camera or not frames:
        return [], "no_frames"
    topic = camera["topic"].removesuffix("/compressed")
    info_topic = topic.rpartition("/")[0] + "/camera_info"
    connections = [
        c
        for c in reader.connections
        if c.msgtype == "sensor_msgs/msg/CameraInfo" and c.topic == info_topic
    ]
    if not connections:
        return [], "no_camera_info"
    if topic.rsplit("/", 1)[-1] not in (
        "image",
        "image_raw",
        "image_color",
        "image_mono",
    ):
        return [], "unsupported_image_geometry"
    timed = [
        f
        for f in frames
        if f.get("optical_frame")
        and f.get("sensor_size")
        and f.get("stamp_ns", 0) > 0
        and abs(f.get("received_ns", 0) - f["stamp_ns"]) <= 150_000_000
        and abs((f["stamp_ns"] - reader.start_time) * 1e-9 - f["t"]) <= 0.002
    ]
    if not timed:
        return [], "no_acquisition_times"
    indices = np.linspace(0, len(timed) - 1, min(16, len(timed)), dtype=int)
    chosen = [timed[i] for i in indices]
    stamps = np.array([f["stamp_ns"] for f in chosen], dtype=np.int64)
    best_age = np.full(len(chosen), 50_000_001, dtype=np.int64)
    models = [None] * len(chosen)
    for c, _, raw in reader.messages(connections=connections):
        info = reader.deserialize(raw, c.msgtype)
        t = info.header.stamp
        age = np.abs(stamps - (t.sec * 10**9 + t.nanosec))
        for i in np.flatnonzero(age < best_age):
            if chosen[i]["optical_frame"] == info.header.frame_id.strip("/"):
                models[i] = _intrinsics(info, chosen[i]["sensor_size"])
                best_age[i] = age[i]
    views = []
    for frame, model in zip(chosen, models):
        if model is None:
            continue
        tf = static_transform(graph, frame["optical_frame"], base)
        t = frame["stamp_ns"] * 1e-9
        j = np.searchsorted(trajectory[:, 0], t)
        if tf is None or t < trajectory[0, 0] or j == len(trajectory):
            continue
        if (
            j
            and trajectory[j, 0] != t
            and trajectory[j, 0] - trajectory[j - 1, 0] > 1.0
        ):
            continue
        xyz, quat = pose_at(t, trajectory[:, 0], trajectory[:, 1:4], trajectory[:, 4:])
        body_rot = Rotation.from_quat(quat).as_matrix()
        center = xyz + body_rot @ tf[:3, 3]
        if any(np.linalg.norm(center - v["center"]) < 0.10 for v in views):
            continue
        path = Path(directory) / frame["file"]
        if not path.is_file():
            continue
        data = path.read_bytes()
        if not data or (
            frame.get("image_sha256")
            and sha256(data).hexdigest() != frame["image_sha256"]
        ):
            continue
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if (
            image is None
            or abs(
                image.shape[0] * frame["sensor_size"][0] / frame["sensor_size"][1]
                - image.shape[1]
            )
            > 2
        ):
            continue
        k, d = model
        scale = np.array(image.shape[::-1]) / frame["sensor_size"]
        k = k.copy()
        k[:2] *= scale[:, None]
        k[:2, 2] += (scale - 1) / 2
        edges = cv2.Canny(cv2.GaussianBlur(image, (3, 3), 0), 60, 140)
        distances = cv2.distanceTransform(
            255 - edges, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )
        views.append(
            dict(
                center=center,
                rotation=body_rot @ tf[:3, :3],
                k=k,
                d=d,
                distances=distances,
            )
        )
    return views, "calibrated" if len(views) >= 2 else "insufficient_calibrated_views"


def _pixels(xyz, view):
    camera = (xyz - view["center"]) @ view["rotation"]
    z = camera[:, 2]
    height, width = view["distances"].shape
    # Cull behind-camera and extreme off-axis points before applying lens distortion.
    visible = (z > 0.2) & (np.max(np.abs(camera[:, :2]), axis=1) < 4 * z)
    ids = np.flatnonzero(visible)
    if len(ids):
        projected = cv2.projectPoints(
            camera[ids], np.zeros(3), np.zeros(3), view["k"], view["d"]
        )[0].reshape(-1, 2)
        inside = ((projected >= 0) & (projected < [width - 0.5, height - 0.5])).all(1)
        return ids[inside], np.rint(projected[inside]).astype(int), z
    return ids, np.empty((0, 2), int), z


def preserve_image_edges(xyz, candidate, kinds, views, size):
    """A visible edge in two separated views can veto a lidar-only displacement."""
    moved = np.flatnonzero(kinds)
    votes = np.zeros(len(moved), np.uint8)
    if len(views) < 2 or not len(moved):
        return 0
    for view in views:
        distances = view["distances"]
        height, width = distances.shape
        # The depth buffer uses all retained surfaces, including those which stayed fixed.
        depth = np.full(((height + 7) // 8, (width + 7) // 8), np.inf, np.float32)
        for first in range(0, len(xyz), 65536):
            ids, uv, z = _pixels(xyz[first : first + 65536], view)
            np.minimum.at(depth, (uv[:, 1] // 8, uv[:, 0] // 8), z[ids])
        depth = cv2.erode(depth, np.ones((3, 3), np.uint8))
        for first in range(0, len(moved), 65536):
            rows = moved[first : first + 65536]
            old, uv, z = _pixels(xyz[rows], view)
            new, dest, _ = _pixels(candidate[rows], view)
            common, a, b = np.intersect1d(
                old, new, assume_unique=True, return_indices=True
            )
            x, y = uv[a].T
            nx, ny = dest[b].T
            near = z[common] <= depth[y // 8, x // 8] + min(size, 0.12)
            lost_edge = (distances[y, x] <= 1.5) & (
                distances[ny, nx] > distances[y, x] + 1.0
            )
            votes[first + common[near & lost_edge]] += 1
    protected = moved[votes >= 2]
    candidate[protected] = xyz[protected]
    kinds[protected] = 0
    return len(protected)

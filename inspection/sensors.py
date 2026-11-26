"""Read ROS 2 MCAP/SQLite and standard ROS recordings through their message definitions."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .channels import choose_channels
from .recording import open_recording
from .perception import Camera
from .geometry import relative, transform, voxel


@dataclass
class Scan:
    t: float
    hits: np.ndarray
    clear: np.ndarray


def quat(q):
    return [q.x, q.y, q.z, q.w]


def stamp(m, log_time, epoch):
    if not hasattr(m, "header"):
        return log_time
    s = m.header.stamp
    t = s.sec + s.nanosec * 1e-9 - epoch
    # Header stamps with a different epoch cannot be interpolated against bag time.
    return t if abs(t - log_time) < 3 else log_time


def cloud_points(msg):
    fields = {f.name: f for f in msg.fields}
    formats = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}
    if not all(n in fields for n in ("x", "y", "z")):
        raise ValueError("В PointCloud2 отсутствуют координаты x, y, z")
    xyz = ["x", "y", "z"]
    dtype = np.dtype(
        dict(
            names=xyz,
            formats=[
                (">" if msg.is_bigendian else "<") + formats[fields[n].datatype]
                for n in xyz
            ],
            offsets=[fields[n].offset for n in xyz],
            itemsize=msg.point_step,
        )
    )
    data = np.ndarray(
        (msg.height, msg.width),
        dtype=dtype,
        buffer=msg.data,
        strides=(msg.row_step, msg.point_step),
    )
    points = np.column_stack([data[n].ravel() for n in xyz])
    return points[np.isfinite(points).all(axis=1)]


def static_transform(graph, source, target):
    source, target = source.strip("/"), target.strip("/")
    stack = [(source, np.eye(4))]
    seen = set()
    while stack:
        f, tf = stack.pop()
        if f == target:
            return tf
        if f in seen:
            continue
        seen.add(f)
        for (child, parent), mat in graph.items():
            if f == child:
                stack.append((parent, mat @ tf))
    return None


def read_static_transforms(reader):
    graph = {}
    connections = [
        c for c in reader.connections if c.topic.rstrip("/").endswith("/tf_static")
    ]
    if not connections:
        return graph
    for c, _, raw in reader.messages(connections=connections):
        for tf in reader.deserialize(raw, c.msgtype).transforms:
            v = tf.transform.translation
            mat = np.eye(4)
            mat[:3, :3] = Rotation.from_quat(quat(tf.transform.rotation)).as_matrix()
            mat[:3, 3] = [v.x, v.y, v.z]
            child, parent = tf.child_frame_id.strip("/"), tf.header.frame_id.strip("/")
            graph[child, parent] = mat
            graph[parent, child] = np.linalg.inv(mat)
    return graph


def first_body_frame(reader, names):
    odometry = [
        c
        for c in reader.connections
        if names.get(c.id) == "odometry" and c.msgtype == "nav_msgs/msg/Odometry"
    ]
    if odometry:
        for c, _, raw in reader.messages(connections=odometry):
            frame = reader.deserialize(raw, c.msgtype).child_frame_id.strip("/")
            if frame:
                return frame
    return None


def reduce_scan(points, planar):
    # Keep one nearest obstacle and one floor observation per bearing. Floor returns reveal free space.
    r = np.linalg.norm(points[:, :2], axis=1)
    good = (r > 0.18) & (r < 25) & (points[:, 2] > -1.5) & (points[:, 2] < 2.25)
    p = points[good]
    r = r[good]
    occupied = (
        np.ones(len(p), dtype=bool) if planar else (p[:, 2] > -0.17) & (p[:, 2] < 1.65)
    )
    bins = (
        np.floor((np.arctan2(p[:, 1], p[:, 0]) + np.pi) * 180 / np.pi).astype(int) % 360
    )
    # Retain the height-separated endpoints observed in each direction for registration.
    # Only floor returns may certify free space. A ceiling return above an obstacle cannot do so.
    hits = voxel(p[occupied, :2], 0.065)
    order = np.argsort(r)
    lo = order[p[order, 2] <= -0.17][::-1] if not planar else np.array([], dtype=int)
    _, far = np.unique(bins[lo], return_index=True)
    return hits, p[lo[far], :2]


def accumulate(scans, odom):
    # The non-repetitive Go2 scanner samples different narrow sectors on successive revolutions.
    # Bring each observation into its bundle's body frame using the measured, timestamped motion.
    if len(odom) < 2 or len(scans) < 3:
        return scans
    ts = np.array([s.t for s in scans])
    euler = Rotation.from_quat(odom[:, 4:8]).as_euler("xyz")
    poses = np.column_stack(
        [
            np.interp(ts, odom[:, 0], v)
            for v in (odom[:, 1], odom[:, 2], np.unwrap(euler[:, 2]))
        ]
    )
    result = []
    i = 0
    while i < len(scans):
        stop = np.searchsorted(ts, ts[i] + 0.44, side="right")
        j = (i + stop - 1) // 2
        hits = []
        free = []
        for k in range(i, stop):
            delta = relative(poses[j], poses[k])
            hits.append(transform(scans[k].hits, delta))
            free.append(transform(scans[k].clear, delta))
        result.append(
            Scan(ts[j], voxel(np.vstack(hits), 0.075), voxel(np.vstack(free), 0.14))
        )
        i = stop
    return result


def read(source, out, progress, topics=None):
    source = Path(source)
    cam = Camera(out)
    scans = []
    odom = []
    imu = []
    base = None
    transforms = {}
    last = {}
    warnings = []
    with open_recording(source) as reader, cam:
        epoch = reader.start_time * 1e-9
        duration = (reader.end_time - reader.start_time) * 1e-9
        cc = reader.connections
        names, channels = choose_channels(cc, topics)
        if "lidar" not in names.values():
            raise ValueError(
                "В записи нет PointCloud2 или LaserScan для построения карты"
            )
        tf_graph = read_static_transforms(reader)
        base = first_body_frame(reader, names)
        if not base and any("base_link" in edge for edge in tf_graph):
            base = "base_link"
        update = -1
        for c, tick, buf in reader.messages(
            connections=[c for c in cc if c.id in names and names[c.id] != "static_tf"]
        ):
            role = names[c.id]
            log = (tick - reader.start_time) * 1e-9
            if log > update:
                progress(
                    0.02 + 0.48 * log / max(0.001, duration),
                    "Декодирование датчиков и обработка камеры",
                )
                update = log + 2
            interval = {"odometry": 0.05, "imu": 0.02}.get(role)
            if interval and log - last.get(role, -100) < interval:
                continue
            if role == "camera" and c.msgtype == "unitree_go/msg/Go2FrontVideoData":
                cam.go2(buf, log)
                continue
            m = reader.deserialize(buf, c.msgtype)
            last[role] = log
            t = stamp(m, log, epoch)
            if role == "camera":
                cam.standard(m, t, tick)
            elif role == "odometry":
                if hasattr(m, "child_frame_id") and m.child_frame_id:
                    if base != m.child_frame_id.strip("/"):
                        raise ValueError(
                            "Запись меняет систему координат корпуса. Выберите однородный канал одометрии."
                        )
                p = m.pose.pose if hasattr(m.pose, "pose") else m.pose
                q = quat(p.orientation)
                values = [p.position.x, p.position.y, p.position.z, *q]
                if np.isfinite(values).all() and np.linalg.norm(q) > 0.2:
                    odom.append((t, *values))
            elif role == "imu":
                v = m.angular_velocity
                q = quat(m.orientation)
                if np.linalg.norm(q) > 0.5 and m.orientation_covariance[0] != -1:
                    up = Rotation.from_quat(q).inv().apply([0, 0, 1])
                else:
                    acc = m.linear_acceleration
                    up = np.array([acc.x, acc.y, acc.z])
                    up /= max(np.linalg.norm(up), 1e-6)
                rate = np.dot(up, [v.x, v.y, v.z])
                if np.isfinite(rate):
                    imu.append((t, rate))
            elif role == "lidar":
                frame = m.header.frame_id.strip("/")
                base = base or frame
                if frame not in transforms:
                    mat = static_transform(tf_graph, frame, base)
                    if mat is None:
                        raise ValueError(
                            f"Требуется статическая калибровка TF из {frame} в {base}"
                        )
                    transforms[frame] = mat
                planar = c.msgtype.endswith("LaserScan")
                if planar:
                    ranges = np.array(m.ranges)
                    angles = m.angle_min + np.arange(len(ranges)) * m.angle_increment
                    ok = (
                        np.isfinite(ranges)
                        & (ranges >= m.range_min)
                        & (ranges <= m.range_max)
                    )
                    points = np.column_stack(
                        [
                            np.cos(angles[ok]) * ranges[ok],
                            np.sin(angles[ok]) * ranges[ok],
                            np.zeros(ok.sum()),
                        ]
                    )
                else:
                    points = cloud_points(m)
                points = points.astype("float32", copy=False)
                mat = transforms[frame]
                if frame != base:
                    points = points @ mat[:3, :3].T + mat[:3, 3]
                hits, free = reduce_scan(points, planar)
                if len(hits) >= 8:
                    scans.append(Scan(t, hits, free))
    scans.sort(key=lambda scan: scan.t)
    if len(scans) < 3:
        raise ValueError("Недостаточно пригодных сканирований лидара")
    odom = np.array(sorted(odom))
    scans = accumulate(scans, odom)
    if not odom.size:
        warnings.append(
            "Одометрия платформы отсутствует, начальное приближение вычисляется по последовательным сканам."
        )
    if not imu:
        warnings.append("Нет пригодных данных IMU.")
    if not cam.frames:
        warnings.append("Декодируемые кадры камеры отсутствуют.")
    if cam.frames and cam.clock.startswith("bag"):
        warnings.append(
            "Камера Go2 привязана по времени приёма пакета; точная задержка камеры не калибрована."
        )
    return dict(
        epoch=epoch,
        duration=duration,
        scans=scans,
        odom=odom,
        imu=np.array(sorted(imu)),
        base=base,
        camera=cam.report(),
        channels=channels,
        warnings=warnings,
    )

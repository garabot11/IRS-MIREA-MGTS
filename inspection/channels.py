"""ROS recording metadata and sensor selection shared by preview and analysis."""

import sqlite3

from rosbags.highlevel import AnyReaderError
from rosbags.rosbag2 import ReaderError
from .recording import open_recording

# Type order is the fallback when a rig has no known preferred topic name.
ROLES = {
    "lidar": ("sensor_msgs/msg/PointCloud2", "sensor_msgs/msg/LaserScan"),
    "odometry": (
        "nav_msgs/msg/Odometry",
        "geometry_msgs/msg/PoseStamped",
        "geometry_msgs/msg/PoseWithCovarianceStamped",
    ),
    "imu": ("sensor_msgs/msg/Imu",),
    "camera": (
        "unitree_go/msg/Go2FrontVideoData",
        "sensor_msgs/msg/CompressedImage",
        "sensor_msgs/msg/Image",
    ),
}
PREFERRED = {
    "lidar": ("/utlidar/cloud_base", "/points", "/scan"),
    "odometry": ("/utlidar/robot_odom", "/odom"),
    "imu": ("/utlidar/imu", "/imu/data"),
    "camera": (
        "/frontvideostream",
        "/camera/image_raw/compressed",
        "/camera/image_raw",
    ),
}


def choose_channels(connections, topics=None):
    topics = topics or {}
    if unknown := topics.keys() - ROLES.keys():
        raise ValueError(f"Неизвестная роль датчика: {', '.join(sorted(unknown))}")
    names = {}
    for role, types in ROLES.items():
        preferred = PREFERRED[role]
        choices = [c for c in connections if c.msgtype in types and c.msgcount != 0]
        if role in topics:
            choices = [c for c in choices if c.topic == topics[role]]
            if not choices:
                raise ValueError(
                    f"Для роли {role} в записи нет сообщений подходящего типа в {topics[role]}"
                )
        if choices:

            def priority(c):
                return (
                    types.index(c.msgtype) if role in ("lidar", "odometry") else 0,
                    preferred.index(c.topic)
                    if c.topic in preferred
                    else len(preferred),
                    types.index(c.msgtype),
                    -c.msgcount,
                    c.topic,
                )

            names[min(choices, key=priority).id] = role
    for c in connections:
        if c.msgtype == "tf2_msgs/msg/TFMessage" and c.topic.rstrip("/").endswith(
            "/tf_static"
        ):
            names[c.id] = "static_tf"
    channels = [
        dict(
            topic=c.topic,
            type=c.msgtype,
            count=c.msgcount,
            role=names.get(c.id, "auxiliary"),
        )
        for c in connections
    ]
    return names, channels


def capabilities(source):
    try:
        with open_recording(source) as reader:
            _, channels = choose_channels(reader.connections)
            duration = reader.duration * 1e-9
    except (
        AnyReaderError,
        ReaderError,
        sqlite3.Error,
        OSError,
        ValueError,
        UnicodeError,
    ) as exc:
        raise ValueError(
            "Не удалось открыть индекс записи ROS. Проверьте формат и целостность файла."
        ) from exc
    candidates = {
        role: sorted(
            {c["topic"] for c in channels if c["type"] in types and c["count"] != 0}
        )
        for role, types in ROLES.items()
    }
    selected = {c["role"]: c["topic"] for c in channels if c["role"] in ROLES}
    lidar = next((c for c in channels if c["role"] == "lidar"), None)
    xyz = bool(lidar and lidar["type"] == ROLES["lidar"][0])
    notes = []
    if not lidar:
        notes.append("Нет PointCloud2 или LaserScan для построения карты.")
    elif not xyz:
        notes.append(
            "LaserScan измеряет плоское сечение. Для объёмного облака нужны XYZ."
        )
    if "odometry" not in selected:
        notes.append("Одометрия отсутствует. Траектория будет оценена по сканам.")
    if "camera" not in selected:
        notes.append("Канал поддерживаемой камеры отсутствует.")
    if any(len(options) > 1 for options in candidates.values()):
        notes.append(
            "Обнаружено несколько каналов одной роли. Проверьте выбор датчика."
        )
    return dict(
        duration=duration,
        channels=channels,
        selected=selected,
        candidates=candidates,
        mapping=bool(lidar),
        pointcloud=xyz,
        notes=notes,
    )

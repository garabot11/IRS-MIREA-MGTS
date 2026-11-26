"""Recover the Go2 sensor interface and a bank of measured ray directions from the two MCAPs."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import rosbag2_py
from nav_msgs.msg import Odometry
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Imu, PointCloud2

RAW = "/utlidar/cloud"
BASE = "/utlidar/cloud_base"
IMU = "/utlidar/imu"
ODOM = "/utlidar/robot_odom"
CLOUD_LAYOUT = [
    ("x", 0, 7),
    ("y", 4, 7),
    ("z", 8, 7),
    ("intensity", 16, 7),
    ("ring", 20, 4),
    ("time", 24, 7),
]
RECORD = np.dtype(
    dict(
        names=["xyz", "intensity", "ring", "time"],
        formats=[("<f4", 3), "<f4", "<u2", "<f4"],
        offsets=[0, 16, 20, 24],
        itemsize=32,
    )
)


def distribution(values):
    return dict(
        zip(
            ("min", "p05", "median", "p95", "max"),
            np.percentile(values, [0, 5, 50, 95, 100]).tolist(),
        )
    )


def points(msg):
    if (
        [(f.name, f.offset, f.datatype) for f in msg.fields] != CLOUD_LAYOUT
        or any(f.count != 1 for f in msg.fields)
        or msg.point_step != 32
        or msg.height != 1
        or msg.is_bigendian
        or msg.row_step != msg.width * 32
    ):
        raise ValueError("Unexpected PointCloud2 layout")
    return np.frombuffer(msg.data, RECORD)


def rigid_transform(raw, base):
    raw, base = raw.astype(float), base.astype(float)
    u, _, vh = np.linalg.svd((raw - raw.mean(0)).T @ (base - base.mean(0)))
    sign = np.sign(np.linalg.det(vh.T @ u.T))
    rotation = vh.T @ np.diag([1.0, 1.0, sign]) @ u.T
    return rotation, base.mean(0) - rotation @ raw.mean(0)


def inspect(directory, transform, bank):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(directory), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=[RAW, BASE, ODOM, IMU]))
    times, frames, waiting = defaultdict(list), defaultdict(set), {}
    widths, raw_widths, matches, max_dt, residuals, ranges, rings = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    scan_stamps = []
    bank_indices = []
    while reader.has_next():
        topic, payload, _ = reader.read_next()
        cls = Imu if topic == IMU else Odometry if topic == ODOM else PointCloud2
        msg = deserialize_message(payload, cls)
        stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        times[topic].append(stamp)
        frames[topic].add(msg.header.frame_id)
        if topic == ODOM:
            frames["odom_child"].add(msg.child_frame_id)
        if cls != PointCloud2:
            continue
        pair = waiting.setdefault(stamp, {})
        pair[topic] = points(msg)
        if len(pair) != 2:
            continue
        raw, base = pair[RAW], pair[BASE]
        del waiting[stamp]
        _, i, j = np.intersect1d(raw["time"], base["time"], return_indices=True)
        if (
            len(i) != len(base)
            or not np.array_equal(raw["ring"][i], base["ring"][j])
            or not np.array_equal(raw["intensity"][i], base["intensity"][j])
        ):
            raise ValueError(f"Ambiguous raw to base correspondences at {stamp}")
        source, target = raw["xyz"][i], base["xyz"][j]
        if transform is None:
            # Reserve all following scans and the entire second bag for verification.
            transform = rigid_transform(source, target)
        rotation, origin = transform
        errors = np.linalg.norm(source @ rotation.T + origin - target, axis=1)
        lengths = np.linalg.norm(source, axis=1)
        if lengths.min() <= 0 or not np.isfinite(errors).all():
            raise ValueError("Invalid polar sample")
        index = len(scan_stamps)
        if bank is not None and index % 12 == 0:
            bank_indices.append(index)
            bank.append(
                (
                    (source @ rotation.T / lengths[:, None]).astype("<f4"),
                    base["time"][j],
                    base["ring"][j],
                )
            )
        scan_stamps.append(stamp)
        widths.append(len(base))
        raw_widths.append(len(raw))
        matches.append(len(j))
        max_dt.append(base["time"].max())
        rings.append(np.unique(base["ring"]))
        residuals.append(errors)
        ranges.append(lengths)
    unmatched = {str(stamp): sorted(pair) for stamp, pair in waiting.items()}
    if any(BASE in pair for pair in waiting.values()):
        raise ValueError(f"A cloud_base has no matching raw scan in {directory}")
    stats = {}
    for name, stamps in times.items():
        gaps = np.diff(np.array(stamps, np.int64)) / 1e9
        if not (gaps > 0).all():
            raise ValueError(f"Non-monotone {name}")
        stats[name] = dict(
            messages=len(stamps),
            mean_hz=float(1 / gaps.mean()),
            interval_s=distribution(gaps),
            frame_ids=sorted(frames[name]),
        )
    intervals = np.diff(scan_stamps) / 1e9
    periods = [intervals[min(i, len(intervals) - 1)] for i in bank_indices]
    manifest = {
        path.name: hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
        for path in sorted(directory.glob("*.mcap"))
    }
    report = dict(
        mcap_sha256=manifest,
        channels=stats,
        odom_child=sorted(frames["odom_child"]),
        base_returns=distribution(widths),
        raw_returns=distribution(raw_widths),
        unpaired_headers=unmatched,
        paired_clouds=len(matches),
        matched_returns=int(sum(matches)),
        returns_origin_range_m=distribution(np.concatenate(ranges)),
        rings=np.unique(np.concatenate(rings)).tolist(),
        time_last_return_s=distribution(max_dt),
        raw_to_base_error_m=distribution(np.concatenate(residuals)),
    )
    return transform, report, periods


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mapping", type=Path)
    parser.add_argument("localization", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    bank = []
    transform, mapping, periods = inspect(args.mapping, None, bank)
    _, localization, _ = inspect(args.localization, transform, None)
    rotation, origin = transform
    directions, times, rings = (np.concatenate(column) for column in zip(*bank))
    offsets = np.r_[0, np.cumsum([len(frame[0]) for frame in bank])]
    arrays = args.out / "rays.npz"
    np.savez_compressed(
        arrays,
        offsets=offsets,
        directions_base=directions,
        time_s=times,
        ring=rings,
        period_s=periods,
    )
    profile = dict(
        source="official organizer MCAP",
        raw_frame="utlidar_lidar",
        base_frame="base_link",
        imu_frame="utlidar_imu",
        origin_base_m=origin.tolist(),
        R_base_lidar=rotation.tolist(),
        estimator="Kabsch on first paired scan in mapping, matched by time/ring/intensity",
        point_step=32,
        little_endian=True,
        fields=CLOUD_LAYOUT,
        template_scans=len(bank),
        templates="Directions of observed cloud_base returns in every twelfth mapping scan. "
        "Missing emissions and surface reflectivity cannot be recovered. "
        "Template ranges and scene coordinates are discarded.",
        template_sha256=hashlib.sha256(arrays.read_bytes()).hexdigest(),
        brief_nominal_hz={"lidar": 14, "odometry": 100, "imu": 100, "camera": 15},
        unknown=[
            "IMU to body calibration",
            "hardware/firmware revision",
            "non-return directions",
            "hardware range limits and noise model",
            "camera stream absent from the bags",
        ],
        mapping=mapping,
        localization=localization,
    )
    (args.out / "profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    print(
        json.dumps(
            {
                "origin": origin.tolist(),
                "bank_frames": len(bank),
                "mapping": mapping["raw_to_base_error_m"],
                "localization": localization["raw_to_base_error_m"],
            }
        )
    )


if __name__ == "__main__":
    main()

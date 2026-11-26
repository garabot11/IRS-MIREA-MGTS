"""Verify the live ROS map/planner contract and per-scan deadlines against serialized results."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from nav_msgs.msg import Path as PathMessage
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2

from hackathon_solution.run import bag_reader, metadata, verify
from inspection.voxel_cloud import read_ply


def audit(source, output):
    report = verify(output, source)
    with (output / "frames.csv").open() as f:
        rows = list(csv.DictReader(f))
    by_time = {int(row["playback_ns"]): row for row in rows}
    last = changed = planned = successful = 0
    for row in rows:
        if not row.get("map_revision"):
            continue
        before, revision, added = (
            int(row[k]) for k in ("map_revision_before", "map_revision", "new_voxels")
        )
        if before != last or revision != before + bool(
            added or int(row.get("changed_grid_cells") or 0)
        ):
            raise ValueError("Discontinuous map revision")
        if added and row["state"] != "TRACKING":
            raise ValueError(
                "Scan without independent tracking support contributed to map"
            )
        if row["planner_revision"] and int(row["planner_revision"]) != revision:
            raise ValueError("Planner used an obsolete revision")
        last = revision
        changed += added
    reader = bag_reader(output / "rosbag2", metadata(output / "rosbag2"))
    delta = None
    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic == "/kontur/planned_path":
            route = deserialize_message(data, PathMessage)
            if (
                len(route.poses) != int(by_time[t]["plan_points"])
                or route.header.frame_id != "map"
            ):
                raise ValueError("Recorded plan differs from the planner output")
            planned += 1
            successful += int(len(route.poses) >= 2)
        if topic == "/kontur/online_delta":
            delta = deserialize_message(data, PointCloud2)
            if (
                delta.width != int(by_time[t]["online_voxels"])
                or delta.header.frame_id != "map"
            ):
                raise ValueError(
                    "Recorded map increment differs from the mapper output"
                )
    if not (
        changed
        and last > 1
        and successful
        and delta is not None
        and delta.width == changed
    ):
        raise ValueError("Incomplete online map or plan stream")
    saved = read_ply((output / "online_delta.ply").read_bytes())
    xyz = np.column_stack([saved[k] for k in "xyz"])
    if not np.array_equal(xyz, np.frombuffer(delta.data, "<f4").reshape(-1, 3)):
        raise ValueError("Final map on disk differs from the last live ROS message")
    gaps_ms = np.diff([int(row["sensor_ns"]) for row in rows]) / 1e6
    latency = [float(row["available_to_output_ms"]) for row in rows]
    return dict(
        **report,
        online_revisions=last,
        final_voxels=changed,
        plans=planned,
        paths_with_motion=successful,
        before_next_scan=int((latency[:-1] < gaps_ms).sum()),
        comparable_scan_gaps=len(gaps_ms),
        missed_indices=np.flatnonzero(latency[:-1] >= gaps_ms).tolist(),
        latency_ms=dict(
            zip(
                ("median", "p95", "p99", "max"),
                np.percentile(latency, [50, 95, 99, 100]).tolist(),
            )
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.source, args.output), indent=2))

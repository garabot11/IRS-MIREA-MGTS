"""ROS 2 live localization, result rosbag, independent held-out scans and timing."""
import argparse
from collections import Counter, deque
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import rclpy
import rosbag2_py
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path as PathMessage
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.serialization import serialize_message
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker

from .tracker import apply, FixedMap, Tracker


def measured_walls(map_file, map_sha):
    """Embed the measured surface so the ROS bag can be opened on another machine."""
    manifest = map_file.parent / "wall-support.json"
    if not manifest.is_file():
        return None
    info = json.loads(manifest.read_text())
    if info["status"] != "ready" or info["source_sha256"] != map_sha:
        return None
    from .v2.wall_surface import FACET
    data = (map_file.parent / "wall-surface.stl").read_bytes()
    if (hashlib.sha256(data).hexdigest() != info["mesh_sha256"]
            or len(data) != 84 + FACET.itemsize * info["triangles"]):
        raise ValueError("Wall surface differs from the map manifest")
    faces = np.frombuffer(data, FACET, offset=84)
    msg = Marker(ns="measured_walls", id=0, type=Marker.TRIANGLE_LIST, action=Marker.ADD)
    msg.header.frame_id = "map"
    msg.pose.orientation.w = 1.0
    msg.scale.x = msg.scale.y = msg.scale.z = 1.0
    msg.color.r, msg.color.g, msg.color.b, msg.color.a = .52, .74, .79, .56
    msg.points = [Point(x=float(x), y=float(y), z=float(z))
                  for x, y, z in faces["vertices"].reshape(-1, 3)]
    return msg


def ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def matrix(message):
    p, q = message.position, message.orientation
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    result[:3, 3] = [p.x, p.y, p.z]
    return result


def set_coordinates(position, orientation, transform):
    position.x, position.y, position.z = transform[:3, 3].tolist()
    orientation.x, orientation.y, orientation.z, orientation.w = Rotation.from_matrix(
        transform[:3, :3]).as_quat().tolist()


def tf(stamp, parent, child, transform):
    msg = TransformStamped()
    msg.header.stamp, msg.header.frame_id, msg.child_frame_id = stamp, parent, child
    set_coordinates(msg.transform.translation, msg.transform.rotation, transform)
    return msg


def read_points(msg, with_mask=False):
    fields = {f.name: f for f in msg.fields}
    if any(k not in fields or fields[k].datatype != PointField.FLOAT32 for k in ("x", "y", "z")):
        raise ValueError("Expected float32 x,y,z in PointCloud2")
    dtype = np.dtype({"names": list("xyz"), "formats": [">f4" if msg.is_bigendian else "<f4"] * 3,
                      "offsets": [fields[k].offset for k in "xyz"], "itemsize": msg.point_step})
    records = np.ndarray((msg.height, msg.width), buffer=msg.data, dtype=dtype,
                         strides=(msg.row_step, msg.point_step))
    points = np.stack([records[k].ravel() for k in "xyz"], axis=-1)
    squared = np.einsum("ij,ij->i", points, points)
    valid = np.isfinite(points).all(axis=1) & (squared > 0.8**2) & (squared < 16**2)
    return (points[valid], valid) if with_mask else points[valid]


def cloud(xyz, stamp):
    msg = PointCloud2()
    msg.header.stamp, msg.header.frame_id = stamp, "map"
    msg.height, msg.width, msg.point_step = 1, len(xyz), 12
    msg.row_step = msg.width * msg.point_step
    msg.fields = [PointField(name=k, offset=i * 4, datatype=PointField.FLOAT32, count=1)
                  for i, k in enumerate("xyz")]
    msg.data = np.ascontiguousarray(xyz, dtype="<f4").tobytes()
    msg.is_dense = True
    return msg


def distribution(values):
    a = np.array([x for x in values if x is not None], dtype=float)
    if not len(a):
        return None
    return dict(zip(("p50", "p95", "p99", "max"), np.percentile(a, [50, 95, 99, 100]).tolist()))


def metadata(index, topic, cls, latched):
    typename = cls.__module__.split(".")[0] + "/msg/" + cls.__name__
    fields = dict(name=topic, type=typename, serialization_format="cdr")
    # Jazzy stores structured QoS. Humble stores the equivalent YAML string.
    if latched:
        qos = ('- history: keep_last\n  depth: 1\n  reliability: reliable\n'
               '  durability: transient_local\n  liveliness: automatic\n'
               '  deadline: {sec: 0, nsec: 0}\n  lifespan: {sec: 0, nsec: 0}\n'
               '  liveliness_lease_duration: {sec: 0, nsec: 0}\n'
               '  avoid_ros_namespace_conventions: false')
    try:
        profiles = rosbag2_py.to_rclcpp_qos_vector(qos, 9) if latched else []
        return rosbag2_py.TopicMetadata(id=index, offered_qos_profiles=profiles, **fields)
    except (AttributeError, TypeError):
        legacy = ('- history: 1\n  depth: 1\n  reliability: 1\n  durability: 1\n'
                  '  liveliness: 1\n  deadline: {sec: 9223372036, nsec: 854775807}\n'
                  '  lifespan: {sec: 9223372036, nsec: 854775807}\n'
                  '  liveliness_lease_duration: {sec: 9223372036, nsec: 854775807}\n'
                  '  avoid_ros_namespace_conventions: false') if latched else ""
        return rosbag2_py.TopicMetadata(offered_qos_profiles=legacy, **fields)


class LocalizationNode(Node):
    def __init__(self, args):
        super().__init__("kontur_localization", parameter_overrides=[
            Parameter("use_sim_time", Parameter.Type.BOOL, not args.wall_clock)])
        self.args = args
        args.output.mkdir(parents=True, exist_ok=True)
        if any((args.output / name).exists() for name in ("rosbag2", "frames.csv", "timing.json")):
            raise FileExistsError("Choose a new output directory")
        start = time.perf_counter()
        self.fixed = FixedMap(args.map)
        self.map_sha = hashlib.sha256(args.map.read_bytes()).hexdigest()
        self.wall_msg = measured_walls(args.map, self.map_sha)
        self.tracker = Tracker(self.fixed, args.start_mode)
        self.online = None
        if args.online_mapping:
            from .v2.online import OnlineMap
            self.online = OnlineMap(self.fixed, args.map.parent / "grid.npz")
            profile = json.loads((Path(__file__).parent / "v2/sensor/profile.json").read_text())
            self.lidar_origin = np.array(profile["origin_base_m"])
        self.map_load_ms = (time.perf_counter() - start) * 1000
        self.odoms = deque(maxlen=192)
        self.rows = []
        self.trajectory = PathMessage()
        self.previous = None
        self.first_wall = None
        self.latest_wall = None
        self.last_map = self.last_path = 0
        self.last_output_stamp = 0
        self.path_segment = 0
        self.invalid_inputs = 0
        self.map_msg = cloud(self.fixed.xyz, Time().to_msg())
        self.writer = rosbag2_py.SequentialWriter()
        self.writer.open(rosbag2_py.StorageOptions(uri=str(args.output / "rosbag2"), storage_id="mcap"),
                         rosbag2_py.ConverterOptions("cdr", "cdr"))
        self.pubs = {}
        map_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        for i, (topic, cls) in enumerate((
            ("/kontur/map3d", PointCloud2), ("/kontur/pose", PoseStamped),
            ("/kontur/scan", PointCloud2), ("/kontur/path", PathMessage),
            ("/tf", TFMessage), ("/kontur/status", String)), 1):
            self.pubs[topic] = self.create_publisher(cls, topic, map_qos if topic == "/kontur/map3d" else 10)
            self.writer.create_topic(metadata(i, topic, cls, topic == "/kontur/map3d"))
        if self.wall_msg is not None:
            topic = "/kontur/walls"
            self.pubs[topic] = self.create_publisher(Marker, topic, map_qos)
            self.writer.create_topic(metadata(9, topic, Marker, True))
        if self.online:
            for index, (topic, cls) in enumerate((("/kontur/online_delta", PointCloud2),
                                                  ("/kontur/planned_path", PathMessage)), 7):
                self.pubs[topic] = self.create_publisher(cls, topic, 5)
                self.writer.create_topic(metadata(index, topic, cls, False))
        reliability = ReliabilityPolicy.RELIABLE if args.reliable_input else ReliabilityPolicy.BEST_EFFORT
        sensor_qos = QoSProfile(depth=128, reliability=reliability)
        self.create_subscription(Odometry, "/utlidar/robot_odom", self.odometry, sensor_qos)
        self.create_subscription(PointCloud2, "/utlidar/cloud_base", self.on_scan, sensor_qos)

    def emit(self, topic, message, stamp_ns):
        self.pubs[topic].publish(message)
        self.writer.write(topic, serialize_message(message), stamp_ns)

    def odometry(self, msg):
        if msg.header.frame_id != "odom" or msg.child_frame_id != "base_link":
            return
        q = msg.pose.pose.orientation
        p = msg.pose.pose.position
        if not np.isfinite([p.x, p.y, p.z, q.x, q.y, q.z, q.w]).all() or q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w < 0.25:
            return
        t = ns(msg.header.stamp)
        if not self.odoms or t > self.odoms[-1][0]:
            self.odoms.append((t, matrix(msg.pose.pose)))

    def on_scan(self, msg, info):
        wall_start = time.perf_counter()
        epoch_start = time.time_ns()
        sensor_ns = ns(msg.header.stamp)
        # The organiser bags use a sensor clock ~81 s behind recording time.
        # Acquisition and odometry use the sensor clock; publication uses replay /clock.
        pub_ns = max(self.last_output_stamp + 1, self.get_clock().now().nanoseconds)
        stamp = Time(nanoseconds=pub_ns).to_msg()
        self.last_output_stamp = pub_ns
        self.first_wall = self.first_wall if self.first_wall is not None else wall_start
        row = dict(scan_index=len(self.rows), sensor_ns=sensor_ns, playback_ns=pub_ns,
                   elapsed_wall_s=wall_start-self.first_wall, holdout=False, state="NO_ODOM",
                   bootstrap_ms=0., registration_ms=0., quality_ms=0., decode_ms=0.)
        # Only already-delivered odometry with header <= cloud header is admitted.
        selected = next(((t, p) for t, p in reversed(self.odoms) if t <= sensor_ns), None)
        pose = points = base = None
        if selected:
            odom_ns, base = selected
            row["odom_ns"] = odom_ns
            row["odom_age_ms"] = (sensor_ns - odom_ns)*1e-6
            if msg.header.frame_id != "base_link":
                row["state"] = "WRONG_FRAME"
            elif row["odom_age_ms"] > 100:
                row["state"] = "STALE_ODOM"
            else:
                if self.args.motion_compensation:
                    from .v2.online import compensate
                    points, mask = read_points(msg, with_mask=True)
                    points = compensate(msg, points, mask, selected, self.odoms)
                else:
                    points = read_points(msg)
                row["decode_ms"] = (time.perf_counter()-wall_start)*1000
                if len(points) < 30:
                    row["state"] = "SPARSE"
                else:
                    pose, measured = self.tracker.step(base, points)
                    row.update(measured)
                    if self.previous is not None:
                        dt = (sensor_ns - self.previous[0])*1e-9
                        yaw = math.atan2(base[1, 0], base[0, 0])
                        row["yaw_rate_dps"] = math.degrees(math.remainder(yaw-self.previous[1], 2*math.pi))/dt if dt > 0 else None
                    self.previous = (sensor_ns, math.atan2(base[1, 0], base[0, 0]))
        output = time.perf_counter()
        if row["state"] in ("STALE_ODOM", "NO_ODOM", "WRONG_FRAME", "SPARSE"):
            self.invalid_inputs += 1
            if self.invalid_inputs >= 14 and self.tracker.seed is not None:
                self.tracker.lose_fix()
        else:
            self.invalid_inputs = 0
        if not self.last_map or pub_ns - self.last_map >= 10_000_000_000:
            self.map_msg.header.stamp = stamp
            self.emit("/kontur/map3d", self.map_msg, pub_ns)
            if self.wall_msg is not None and self.last_map == 0:
                self.wall_msg.header.stamp = self.map_msg.header.stamp
                self.emit("/kontur/walls", self.wall_msg, pub_ns)
            self.last_map = pub_ns
        transforms = []
        if base is not None and row["state"] not in ("WRONG_FRAME", "STALE_ODOM"):
            transforms.append(tf(stamp, "odom", "base_link", base))
        if pose is not None:
            if self.path_segment != self.tracker.segment:
                self.trajectory = PathMessage()
                self.last_path = 0
                self.path_segment = self.tracker.segment
            msg_pose = PoseStamped()
            msg_pose.header.frame_id, msg_pose.header.stamp = "map", stamp
            set_coordinates(msg_pose.pose.position, msg_pose.pose.orientation, pose)
            self.emit("/kontur/pose", msg_pose, pub_ns)
            transforms.append(tf(stamp, "map", "odom", self.tracker.transform))
            world = apply(pose, points)
            self.emit("/kontur/scan", cloud(world, stamp), pub_ns)
            if self.online is not None:
                start_mapping = time.perf_counter()
                row["map_revision_before"] = self.online.version
                row["new_voxels"] = (self.online.observe(world, pose[:3, 3], sensor_ns, apply(pose, self.lidar_origin))
                                     if row["state"] == "TRACKING" else 0)
                row["changed_grid_cells"] = self.online.grid_changes if row["state"] == "TRACKING" else 0
                row["map_revision"] = self.online.version
                row["online_voxels"] = len(self.online.confirmed)
                row["mapping_ms"] = (time.perf_counter() - start_mapping) * 1000
                if sensor_ns - self.online.last_publish >= 1_000_000_000 or len(self.rows)+1 == self.args.expected_scans:
                    self.emit("/kontur/online_delta", cloud(self.online.points(), stamp), pub_ns)
                    self.online.last_publish = sensor_ns
                start_planning = time.perf_counter()
                route = self.online.route(pose[:3, 3], sensor_ns)
                if route is not None:
                    msg_route = PathMessage(header=msg_pose.header)
                    for x, y in route["path"]:
                        point = PoseStamped(header=msg_pose.header)
                        point.pose.position.x, point.pose.position.y = x, y
                        point.pose.orientation.w = 1.
                        msg_route.poses.append(point)
                    self.emit("/kontur/planned_path", msg_route, pub_ns)
                    row.update(planner_revision=self.online.version, plan_status=route["status"],
                               plan_points=len(route["path"]), planning_ms=(time.perf_counter()-start_planning)*1000)
            if not self.last_path or pub_ns-self.last_path >= 400_000_000 or len(self.rows)+1 == self.args.expected_scans:
                self.trajectory.header = msg_pose.header
                self.trajectory.poses.append(msg_pose)
                self.emit("/kontur/path", self.trajectory, pub_ns)
                self.last_path = pub_ns
            for name, value in zip(("x","y","z","qx","qy","qz","qw"),
                                   [*pose[:3, 3], *Rotation.from_matrix(pose[:3,:3]).as_quat()]):
                row[name] = float(value)
        if transforms:
            self.emit("/tf", TFMessage(transforms=transforms), pub_ns)
        status = dict(state=row["state"], scan_index=row["scan_index"], sensor_ns=sensor_ns,
                      start_mode=self.args.start_mode, holdout=row["holdout"],
                      segment=self.tracker.segment, rejected_updates=self.tracker.rejected,
                      prior_radius_m=1.5 if self.args.start_mode == "seeded" else None)
        self.emit("/kontur/status", String(data=json.dumps(status)), pub_ns)
        self.latest_wall = time.perf_counter()
        row["output_ms"] = (self.latest_wall-output)*1000
        row["callback_ms"] = (self.latest_wall-wall_start)*1000
        # DDS source time is the publisher's system clock on the same replay host.
        # This includes DDS delivery and executor queue, unlike the historical bag stamp.
        source = info.get("source_timestamp", 0) if isinstance(info, dict) else info.source_timestamp
        if source and 0 <= epoch_start-source < 60_000_000_000:
            row["available_to_output_ms"] = (epoch_start-source)*1e-6 + row["callback_ms"]
        self.rows.append(row)

    def finish(self):
        if hasattr(self.writer, "close"):
            self.writer.close()
        self.writer = None
        rows = self.rows
        keys = list(dict.fromkeys(k for row in rows for k in row))
        with (self.args.output / "frames.csv").open("w",newline="") as stream:
            w = csv.DictWriter(stream, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        def quality(subset):
            samples = [r for r in subset if "localized_residual_m" in r]
            if not samples:
                return {"count": 0}
            return dict(count=len(samples),
                        mean_capped_nn_m={k:float(np.mean([r[k+"_residual_m"] for r in samples]))
                                          for k in ("raw","aligned","localized")},
                        improved_vs_aligned=sum(r["localized_residual_m"] < r["aligned_residual_m"] for r in samples))
        published = [r for r in rows if "x" in r]
        timing = {k:distribution([r.get(k) for r in rows]) for k in
                  ("decode_ms","bootstrap_ms","registration_ms","quality_ms","output_ms",
                   "callback_ms","available_to_output_ms","odom_age_ms","odom_correction_m")}
        steady = [r for r in rows if r.get("bootstrap_ms") == 0 and "x" in r]
        gaps = np.diff([r["sensor_ns"] for r in rows])*1e-6
        report = dict(map_sha256=self.map_sha, start_mode=self.args.start_mode,
                      input_qos="reliable" if self.args.reliable_input else "best_effort",
                      expected_scans=self.args.expected_scans, input_scans=len(rows), poses=len(published),
                      complete=bool(rows) and len(rows)==self.args.expected_scans if self.args.expected_scans else None,
                      states=dict(Counter(r["state"] for r in rows)), timing_ms=timing,
                      steady_callback_ms=distribution([r["callback_ms"] for r in steady]),
                      startup=dict(map_load_ms=self.map_load_ms, bootstrap=self.tracker.bootstrap,
                                   first_pose_wall_s=published[0]["elapsed_wall_s"]+published[0]["callback_ms"]/1000 if published else None),
                      sensor_span_s=(rows[-1]["sensor_ns"]-rows[0]["sensor_ns"])*1e-9 if rows else 0,
                      wall_span_s=self.latest_wall-self.first_wall if rows else 0,
                      median_scan_period_ms=float(np.median(gaps)) if len(gaps) else None,
                      held_out=quality(rows), turn=quality([r for r in rows if abs(r.get("yaw_rate_dps") or 0)>=45]),
                      host_timing="monotonic callback; same-host DDS source timestamp to output",
                      causal_max_future_odom_ns=max((r["odom_ns"]-r["sensor_ns"] for r in rows if "odom_ns" in r), default=None))
        if self.online is not None:
            self.online.save(self.args.output)
            report["online_mapping"] = dict(confirmed_voxels=len(self.online.confirmed), revisions=self.online.version,
                                            goal_xy=self.online.goal.tolist(), plans=dict(Counter(
                                                r["plan_status"] for r in rows if "plan_status" in r)),
                                            mapping_ms=distribution([r.get("mapping_ms") for r in rows]),
                                            planning_ms=distribution([r.get("planning_ms") for r in rows]))
        report["motion_compensation"] = self.args.motion_compensation
        (self.args.output / "timing.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-scans", type=int, default=0)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--start-mode", choices=("seeded","global"), default="seeded")
    parser.add_argument("--wall-clock", action="store_true", help="Use for a live sensor without /clock")
    parser.add_argument("--reliable-input", action="store_true",
                        help="Require reliable DDS publishers on both input topics, e.g. for short replay tests")
    parser.add_argument("--online-mapping", action="store_true")
    parser.add_argument("--motion-compensation", action="store_true")
    args = parser.parse_args()
    rclpy.init(args=[])
    node = LocalizationNode(args)
    try:
        if args.ready_file:
            args.ready_file.touch(exist_ok=False)
        while rclpy.ok() and (not args.expected_scans or len(node.rows)<args.expected_scans):
            rclpy.spin_once(node, timeout_sec=0.25)
    except (ExternalShutdownException, KeyboardInterrupt):
        pass
    finally:
        node.finish()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

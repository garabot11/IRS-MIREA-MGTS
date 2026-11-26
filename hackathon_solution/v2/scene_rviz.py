"""Publish a live MuJoCo scene stream as standard ROS 2 markers and point clouds."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Point, TransformStamped
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, DurabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import JointState, PointCloud2
from std_msgs.msg import String
from tf2_ros import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from .wall_surface import FACET
from ..ros_node import cloud
from inspection.voxel_cloud import read_ply


def point(x, y, z):
    return Point(x=float(x), y=float(y), z=float(z))


def marker(header, ns, index, kind, pos, size, color):
    m = Marker(header=header, ns=ns, id=index, type=kind, action=Marker.ADD)
    m.pose.position = point(*pos)
    m.pose.orientation.w = 1.0
    m.scale.x, m.scale.y, m.scale.z = map(float, size)
    m.color.r, m.color.g, m.color.b, m.color.a = map(float, color)
    return m


def route(header, index, points, z, color):
    m = marker(
        header, "routes", index, Marker.LINE_STRIP, (0, 0, 0), (0.034, 0, 0), color
    )
    m.points = [point(x, y, z) for x, y in points]
    return m


class LiveDisplay(Node):
    def __init__(self, folder, map_dir):
        super().__init__("go2_live_display")
        from std_msgs.msg import Header

        self.header = Header(frame_id="map")
        self.folder = folder.absolute()
        self.cases = None
        self.last_file = self.last_publish = 0
        self.scene = self.create_publisher(MarkerArray, "/go2_live/scene", 3)
        self.lidar = self.create_publisher(PointCloud2, "/go2_live/lidar", 2)
        self.added = self.create_publisher(PointCloud2, "/go2_live/updates", 2)
        self.joints = self.create_publisher(JointState, "/joint_states", 2)
        self.status = self.create_publisher(String, "/go2_live/status", 3)
        latch = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.prior = self.create_publisher(PointCloud2, "/go2_live/surfaces", latch)
        self.walls = self.create_publisher(MarkerArray, "/go2_live/walls", latch)
        self.tf = StaticTransformBroadcaster(self)
        tf = TransformStamped(header=self.header, child_frame_id="arena")
        tf.transform.rotation.w = 1.0
        self.tf.sendTransform(tf)
        rec = read_ply((map_dir / "map3d.ply").read_bytes())
        self.prior.publish(
            cloud(np.column_stack([rec[k] for k in "xyz"]), Time().to_msg())
        )
        mesh = map_dir / "wall-surface.stl"
        if mesh.exists():
            faces = np.frombuffer(mesh.read_bytes(), FACET, offset=84)
            m = marker(
                self.header,
                "measured",
                0,
                Marker.TRIANGLE_LIST,
                (0, 0, 0),
                (1, 1, 1),
                (0.55, 0.77, 0.8, 0.45),
            )
            m.points = [point(*v) for v in faces["vertices"].reshape(-1, 3)]
            self.walls.publish(MarkerArray(markers=[m]))
        self.last_ids = set()
        self.create_timer(0.05, self.tick)

    def tick(self):
        path = self.folder / "current.npz"
        if not path.is_file():
            return
        stat = path.stat().st_mtime_ns
        now = time.monotonic()
        if stat == self.last_file and now - self.last_publish < 1:
            return
        self.last_file, self.last_publish = stat, now
        with np.load(path, allow_pickle=False) as data:
            epoch = str(data["epoch"])
            if self.cases is None or self.cases["epoch"] != epoch:
                desc = json.loads((self.folder / "model.json").read_text())
                if desc["epoch"] != epoch:
                    return
                self.cases = desc
            desc = self.cases
            self.header.stamp = self.get_clock().now().to_msg()
            h = self.header
            shapes = {
                "mesh": Marker.MESH_RESOURCE,
                "box": Marker.CUBE,
                "sphere": Marker.SPHERE,
                "cylinder": Marker.CYLINDER,
            }
            markers = []
            for i, (spec, pos, quat) in enumerate(
                zip(desc["geometry"], data["pos"], data["quat"])
            ):
                if pos[2] < -2:
                    continue
                m = marker(
                    h,
                    "robot",
                    i,
                    shapes[spec["shape"]],
                    pos,
                    spec["scale"],
                    spec["rgba"],
                )
                (
                    m.pose.orientation.x,
                    m.pose.orientation.y,
                    m.pose.orientation.z,
                    m.pose.orientation.w,
                ) = map(float, quat)
                if spec["file"]:
                    m.mesh_resource = (self.folder / "models" / spec["file"]).as_uri()
                markers.append(m)
            markers.append(
                marker(
                    h,
                    "goal",
                    0,
                    Marker.CYLINDER,
                    (*desc["goal"], 0.025),
                    (0.32, 0.32, 0.05),
                    (0.32, 0.95, 0.40, 1),
                )
            )
            for i, (coords, z, color) in enumerate(
                [
                    (data["route"], 0.055, (1, 0.76, 0.18, 1)),
                    (data["track"], 0.065, (0.13, 0.86, 0.78, 1)),
                ]
            ):
                if len(coords) > 1:
                    markers.append(route(h, i, coords, z, color))
            participants = [("Go2", data["xy"])] + list(
                zip(desc["actors"], data["actors"])
            )
            for i, (label, pos) in enumerate(participants):
                m = marker(
                    h,
                    "names",
                    i,
                    Marker.TEXT_VIEW_FACING,
                    pos + [0, 0, 0.64 if label == "G1" else 0.30],
                    (0, 0, 0.11),
                    (0.87, 0.97, 0.96, 1),
                )
                m.text = label
                markers.append(m)
            current_ids = {(m.ns, m.id) for m in markers}
            for ns, i in self.last_ids - current_ids:
                markers.append(Marker(header=h, ns=ns, id=i, action=Marker.DELETE))
            self.last_ids = current_ids
            self.scene.publish(MarkerArray(markers=markers))
            self.lidar.publish(cloud(data["lidar"], h.stamp))
            self.added.publish(cloud(data["added"], h.stamp))
            self.joints.publish(
                JointState(
                    header=h, name=desc["joints"], position=data["joints"].tolist()
                )
            )
            self.status.publish(
                String(
                    data=json.dumps(
                        dict(
                            scenario=desc["case"],
                            seconds=float(data["seconds"]),
                            scan=int(data["scan"]),
                            revision=int(data["revision"]),
                            nav=str(data["nav"]),
                        )
                    )
                )
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stream", type=Path)
    parser.add_argument("map", type=Path)
    args = parser.parse_args()
    rclpy.init()
    node = LiveDisplay(args.stream, args.map)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

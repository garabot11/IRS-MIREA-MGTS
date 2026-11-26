"""Sensor-only boundary around the articulated MuJoCo scene used in the V2 trials."""

from collections import deque
from copy import deepcopy
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np
from .humanoid_actor import HumanoidController, augment

from physics.controller import Go2Controller
from physics.voxel_rays import load_native_rays
from physics.world import ASSET_DIR, WEIGHTS_PATH, build_model

START = np.array([2.75, 4.10])
GOAL = np.array([2.75, 8.35])
START_YAW = math.pi / 2
TEAM_SCENES = ("team_cross", "team_chicane")


def scene_definition(directory, with_dog, with_humanoid=False, wall_mesh=None):
    tree = ET.parse(Path(__file__).with_name("sim_assets") / "unitree-go2.xml")
    root = tree.getroot()
    root.find("compiler").set("meshdir", str(ASSET_DIR / "go2/assets"))
    world = root.find("worldbody")
    if wall_mesh is not None:
        ET.SubElement(root.find("asset"), "mesh", name="measured_walls", file=str(Path(wall_mesh).resolve()))
        ET.SubElement(world, "geom", name="measured_walls", type="mesh", mesh="measured_walls",
                      contype="0", conaffinity="0", group="5", rgba=".60 .75 .78 .52")
    base = world.find("body")
    root.remove(
        root.find("sensor")
    )  # terminal evaluation does not consume pose sensors
    root.remove(root.find("keyframe"))
    ET.SubElement(
        world, "geom", name="floor", type="plane", size="0 0 .05", rgba=".17 .20 .22 1"
    )
    ET.SubElement(world, "light", pos="2 6 5", dir="0 0 -1")
    if with_dog:
        other = deepcopy(base)
        for child in other.iter():
            if child.get("name"):
                child.set("name", "other_" + child.get("name"))
            if child.tag == "geom" and child.get("class") == "visual":
                child.attrib.pop("material", None)
                child.set("rgba", ".9 .35 .08 1")
        world.append(other)
        controls = root.find("actuator")
        for motor in list(controls):
            extra = deepcopy(motor)
            extra.set("name", "other_" + motor.get("name"))
            extra.set("joint", "other_" + motor.get("joint"))
            controls.append(extra)
    for name, shape, location, size in (
        ("side", "box", "0 0 .30", ".13 .14 .30"),
        ("barrier", "box", "0 0 .32", ".23 .24 .32"),
        ("pedestrian", "capsule", "0 0 1.05", ".18 .32"),
    ):
        body = ET.SubElement(world, "body", name=name, mocap="true", pos="0 0 -30")
        ET.SubElement(
            body,
            "geom",
            name=name + "_shell",
            type=shape,
            size=size,
            pos=location,
            rgba=".95 .52 .16 1",
        )
        if name == "pedestrian":
            ET.SubElement(
                body,
                "geom",
                name="ped_head",
                type="sphere",
                size=".115",
                pos="0 0 1.67",
                rgba=".83 .68 .53 1",
            )
            for sign in [-1, 1]:
                ET.SubElement(
                    body,
                    "geom",
                    type="capsule",
                    size=".073",
                    fromto=f"{sign * 0.105} 0 .10 {sign * 0.105} 0 .72",
                    rgba=".28 .44 .70 1",
                )
                ET.SubElement(
                    body,
                    "geom",
                    type="capsule",
                    size=".055",
                    fromto=f"{sign * 0.195} 0 1.29 {sign * 0.25} 0 .84",
                    rgba=".65 .71 .82 1",
                )
    if with_humanoid:
        augment(root, directory)
    path = Path(directory) / "combined.xml"
    tree.write(path, encoding="utf-8")
    return path


class VoxelSurface:
    """Nearest return in the measured 3-D voxel layer, independent of the estimator."""

    def __init__(self, xyz, edge):
        self.edge = edge
        keys = np.floor(xyz / edge).astype(int)
        self.origin = keys.min(axis=0)
        self.grid = np.zeros(tuple(keys.max(axis=0) - self.origin + 1), dtype=bool)
        self.grid[tuple((keys - self.origin).T)] = True
        self.native = load_native_rays(self.grid, self.origin, edge)
        self.backend = "native" if self.native else "numpy"

    def nearest(self, center, directions, max_ranges):
        if self.native is not None:
            return self.native.nearest(center, directions, max_ranges)
        return self.nearest_numpy(center, directions, max_ranges)

    def nearest_numpy(self, center, directions, max_ranges):
        # Increment is < half a map voxel. A thin/grazing hit may be missed.
        # Process near → far, stopping each ray on its first occupied voxel.
        result = max_ranges.copy()
        step = self.edge * 0.48
        active = np.arange(len(directions))
        for near in np.arange(0.30, max_ranges.max(initial=0.3), 32 * step):
            if len(active) == 0:
                break
            ds = near + np.arange(32) * step
            keys = (
                np.floor(
                    (center + directions[active, None] * ds[None, :, None]) / self.edge
                ).astype(np.int32)
                - self.origin
            )
            inside = ((keys >= 0) & (keys < self.grid.shape)).all(axis=2)
            hit = np.zeros(inside.shape, bool)
            hit[inside] = self.grid[tuple(keys[inside].T)]
            hit &= ds[None, :] <= result[active, None]
            any_hit = hit.any(axis=1)
            result[active[any_hit]] = ds[hit[any_hit].argmax(axis=1)]
            active = active[~any_hit & (result[active] > ds[-1])]
        return result


class Simulator:
    def __init__(self, grid, xyz, voxel_edge, sensor_dir, scenario, seed, wall_mesh=None):
        self.scenario = scenario
        self.with_dog = scenario in ("dog", "mixed", *TEAM_SCENES)
        with_humanoid = scenario == "humanoid" or scenario in TEAM_SCENES
        with TemporaryDirectory() as temp:
            self.model = build_model(
                scene_definition(temp, self.with_dog, with_humanoid, wall_mesh),
                grid.cells,
                grid.origin,
                grid.resolution,
                [],
            )
        self.model.opt.timestep = 0.002
        self.data = mj.MjData(self.model)
        self.sensing = mj.MjData(self.model)
        self.primary = Go2Controller(
            self.model, self.data, WEIGHTS_PATH, base_name="base_link"
        )
        self.primary.place(START, START_YAW)
        self.other = None
        if self.with_dog:
            self.other = Go2Controller(
                self.model,
                self.data,
                WEIGHTS_PATH,
                prefix="other_",
                base_name="base_link",
            )
            self.other.place((2.96, 9.12), -math.pi / 2)
            if scenario == "team_chicane":
                self.other.place((3.37, 8.88), -math.pi / 2)
        self.humanoid = None
        if with_humanoid:
            self.humanoid = HumanoidController(self.model, self.data)
            if scenario == "team_chicane":
                self.humanoid.place((3.48, 5.98), math.pi)
                self.humanoid.goal = np.array([1.83, 5.98])
            else:
                self.humanoid.place((1.75, 6.50), 0.0)
        self.actor = self.other or self.humanoid
        self.profiles = np.load(sensor_dir / "rays.npz")
        meta = json.loads((sensor_dir / "profile.json").read_text())
        self.origin = np.array(meta["origin_base_m"])
        self.surface = VoxelSurface(xyz, voxel_edge)
        self.rng = np.random.default_rng(seed)
        self.mocap = {
            k: self.model.body(k).mocapid.item()
            for k in ("side", "barrier", "pedestrian")
        }
        roots = self.model.body_rootid[self.model.geom_bodyid]
        self.my_geoms = set(np.flatnonzero(roots == self.primary.base))
        contacts = (self.model.geom_contype != 0) | (self.model.geom_conaffinity != 0)
        self.my_colliders = np.flatnonzero((roots == self.primary.base) & contacts)
        self.query_targets = {
            name: np.flatnonzero((roots == self.model.body(name).id) & contacts)
            for name in ("barrier", "pedestrian")
        }
        if self.other is not None:
            self.query_targets["dog"] = np.flatnonzero(
                (roots == self.other.base) & contacts
            )
        if self.humanoid is not None:
            self.query_targets["humanoid"] = np.flatnonzero(
                (roots == self.humanoid.base) & contacts
            )
        self.actor_geoms = (
            set(np.flatnonzero(roots == self.actor.base)) if self.actor else set()
        )
        self.actor_feet = (
            {self.model.geom("other_" + leg).id for leg in ("FL", "FR", "RL", "RR")}
            if self.other
            else self.humanoid.feet
            if self.humanoid
            else set()
        )
        self.actor_joint_range = [np.full(12, np.inf), np.full(12, -np.inf)]
        self.actor_contacts = dict(foot_floor=0, body_floor=0, static=0, primary=0)
        self.model.geom_group[list(self.my_geoms)] = 5
        # Suppress transparent decoration and observation of our own robot.
        if self.other is not None:
            vis_other = (roots == self.other.base) & (self.model.geom_contype == 0)
            self.model.geom_group[vis_other] = 5
        self.floor = self.model.geom("floor").id
        if wall_mesh is not None:
            for i in range(self.model.ngeom):
                label = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, i) or ""
                if label.startswith(("occupied_", "unknown_", "boundary_")):
                    self.model.geom_group[i] = 3
        self.team_contacts = {"robot_robot": 0, "body_floor": 0, "obstacle": 0}
        self.team_geoms = {}
        self.team_feet = set()
        if scenario in TEAM_SCENES:
            for root in (self.other.base, self.humanoid.base):
                self.team_geoms.update({int(i): root for i in np.flatnonzero(roots == root)})
            self.team_feet.update(self.humanoid.feet)
            self.team_feet.update(self.model.geom("other_" + k).id for k in ("FL", "FR", "RL", "RR"))
        self.history = deque(maxlen=60)
        self.contacts = {
            "foot_floor": 0,
            "body_floor": 0,
            "static": 0,
            "pedestrian": 0,
            "dog": 0,
            "humanoid": 0,
        }
        self.feet = {self.model.geom(name).id for name in ("FL", "FR", "RL", "RR")}
        self.first_contact = None
        self.joint_min, self.joint_max = np.ones(12) * np.inf, np.ones(12) * -np.inf
        self.next_odom = self.next_imu = self.odom_count = self.imu_count = 0
        self.measurements = []
        self.set_actors(0.0)
        mj.mj_forward(self.model, self.data)
        self.remember()

    def set_actors(self, t):
        # Scenario truth is private to the simulator; the mapper receives only rays.
        self.data.mocap_pos[self.mocap["side"]] = (3.57, 5.82, 0.0)
        self.data.mocap_pos[self.mocap["barrier"]] = (
            (2.75, 6.75, 0.0)
            if self.scenario == "unexpected" and t >= 1.8
            else (0.0, 0.0, -30.0)
        )
        if self.scenario == "team_chicane":
            self.data.mocap_pos[self.mocap["side"]] = (3.78, 4.97, 0.0)
            if t >= 1.8:
                self.data.mocap_pos[self.mocap["barrier"]] = (2.77, 7.41, 0.0)
        if self.scenario in ("pedestrian", "mixed") and 2.0 <= t <= 9.6:
            x = 1.34 + 0.385 * (t - 2.0)
            self.data.mocap_pos[self.mocap["pedestrian"]] = (x, 6.51, 0.0)
        else:
            self.data.mocap_pos[self.mocap["pedestrian"]] = (0.0, 0.0, -30.0)

    def remember(self):
        d = self.data
        self.history.append((d.time, d.qpos.copy(), d.mocap_pos.copy()))

    def other_command(self, d):
        # An independent scripted destination for the other physical robot.
        # Its position, route, and command are private to the world.
        position = d.qpos[self.other.free_pos : self.other.free_pos + 2]
        rotation = d.xmat[self.other.base].reshape(3, 3)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        lane_x = 3.39 if self.scenario == "team_chicane" else 2.04
        delta = np.array([lane_x, max(3.50, position[1] - 1.20)]) - position
        error = math.remainder(math.atan2(delta[1], delta[0]) - yaw, math.tau)
        moving = d.time > 0.9 and position[1] > 3.57
        return (
            0.32 * max(0.0, 1 - abs(error) / 1.1) if moving else 0.0,
            0.0,
            np.clip(2.0 * error, -0.95, 0.95) if moving else 0.0,
        )

    def step(self, command):
        self.primary.set_command(*command)
        for _ in range(10):
            d = self.data
            self.set_actors(d.time)
            if self.other:
                self.other.set_command(*self.other_command(d))
                self.other.actuate()
            if self.humanoid:
                self.humanoid.actuate(d.time)
            self.primary.actuate()
            mj.mj_step(self.model, d)
            self.remember()
            r = d.xmat[self.primary.base].reshape(3, 3)
            if d.time >= self.next_odom:
                # Noisy local body-frame velocity, integrated by the navigation process.
                self.measurements.append(
                    (
                        "odom",
                        d.time,
                        r.T @ d.qvel[:3]
                        + [0.005, -0.002, 0]
                        + self.rng.normal(0, 0.003, 3),
                    )
                )
                self.odom_count += 1
                self.next_odom += 1 / 149.6
            if d.time >= self.next_imu:
                self.measurements.append(
                    (
                        "gyro",
                        d.time,
                        d.qvel[3:6].copy()
                        + [0, 0, 0.001]
                        + self.rng.normal(0, 0.001, 3),
                    )
                )
                self.imu_count += 1
                self.next_imu += 1 / 249.75
            pos = d.qpos[self.primary.pos_indices]
            self.joint_min = np.minimum(self.joint_min, pos)
            self.joint_max = np.maximum(self.joint_max, pos)
            if self.actor:
                indices = (
                    self.other.pos_indices
                    if self.other
                    else self.humanoid.joint_indices
                )
                self.actor_joint_range[0] = np.minimum(
                    self.actor_joint_range[0], d.qpos[indices]
                )
                self.actor_joint_range[1] = np.maximum(
                    self.actor_joint_range[1], d.qpos[indices]
                )
            for contact in d.contact:
                a, b = map(int, contact.geom)
                if self.team_geoms:
                    aa, bb = self.team_geoms.get(a), self.team_geoms.get(b)
                    if aa != bb:
                        if aa is not None and bb is not None:
                            self.team_contacts["robot_robot"] += 1
                        elif aa is not None or bb is not None:
                            actor, adjacent = (a, b) if aa is not None else (b, a)
                            if adjacent == self.floor and actor not in self.team_feet:
                                self.team_contacts["body_floor"] += 1
                            elif adjacent != self.floor and adjacent not in self.my_geoms:
                                self.team_contacts["obstacle"] += 1
                if (a in self.actor_geoms) != (b in self.actor_geoms):
                    actor, adjacent = (a, b) if a in self.actor_geoms else (b, a)
                    key = (
                        ("foot_floor" if actor in self.actor_feet else "body_floor")
                        if adjacent == self.floor
                        else ("primary" if adjacent in self.my_geoms else "static")
                    )
                    self.actor_contacts[key] += 1
                if (a in self.my_geoms) == (b in self.my_geoms):
                    continue
                mine, other = (a, b) if a in self.my_geoms else (b, a)
                owner = self.model.body_rootid[self.model.geom_bodyid[other]]
                if other == self.floor:
                    key = "foot_floor" if mine in self.feet else "body_floor"
                elif self.other and owner == self.other.base:
                    key = "dog"
                elif self.humanoid and owner == self.humanoid.base:
                    key = "humanoid"
                elif owner == self.model.body("pedestrian").id:
                    key = "pedestrian"
                else:
                    key = "static"
                self.contacts[key] += 1
                if key != "foot_floor" and self.first_contact is None:
                    self.first_contact = (float(d.time), key)

    def scan(self, number):
        p = self.profiles
        i = number % len(p["period_s"])
        a, b = p["offsets"][i : i + 2]
        rays = p["directions_base"][a:b].astype(float)
        offsets = p["time_s"][a:b]
        hist = list(self.history)
        cloud_end = self.data.time
        # Eight firing groups use past physical poses and past actor locations.
        bins = np.minimum(
            7, (8 * offsets / max(float(offsets.max()), 1e-6)).astype(int)
        )
        distances = np.full(len(rays), -1.0)
        labels = np.zeros(len(rays), np.uint8)
        for group in np.unique(bins):
            mask = bins == group
            target = cloud_end - float(offsets.max()) + float(offsets[mask].mean())
            _, qpos, mocap = min(hist, key=lambda entry: abs(entry[0] - target))
            snap = self.sensing
            snap.qpos[:] = qpos
            snap.mocap_pos[:] = mocap
            mj.mj_kinematics(self.model, snap)
            mj.mj_comPos(self.model, snap)
            r = snap.xmat[self.primary.base].reshape(3, 3)
            origin = snap.xpos[self.primary.base] + r @ self.origin
            directions = np.ascontiguousarray(rays[mask] @ r.T)
            n = len(directions)
            ids = np.empty(n, np.int32)
            ds = np.zeros(n)
            # The physical broad-phase contact walls are generated from the real grid.
            mj.mj_multiRay(
                self.model,
                snap,
                origin,
                directions.ravel(),
                np.array([1, 1, 1, 1, 1, 0], np.uint8),
                True,
                -1,
                ids,
                ds,
                None,
                n,
                36.6,
            )
            limit = np.where(ds >= 0, ds, 36.6)
            measured = self.surface.nearest(origin, directions, limit)
            physically_visible = (ds >= 0) & (abs(measured - ds) < 1e-5)
            owners = self.model.body_rootid[self.model.geom_bodyid[np.maximum(0, ids)]]
            category = np.zeros(n, np.uint8)
            category[owners == self.model.body("pedestrian").id] = 1
            if self.other:
                category[owners == self.other.base] = 2
            if self.humanoid:
                category[owners == self.humanoid.base] = 5
            category[owners == self.model.body("barrier").id] = 3
            category[owners == self.model.body("side").id] = 4
            labels[mask] = np.where(physically_visible, category, 0)
            ds = np.where((ds >= 0) | (measured < 36.59), measured, -1.0)
            distances[mask] = ds
        ok = distances > 0.36
        points = (
            self.origin
            + rays[ok] * (distances[ok] + self.rng.normal(0, 0.006, ok.sum()))[:, None]
        )
        # Isolated simulation evaluator / supervised labels, never passed to navigation.
        self.evaluation_labels = labels[ok]
        self.evaluation_beam_indices = np.flatnonzero(ok)
        return points, dict(
            emitted=len(rays),
            returned=int(ok.sum()),
            ring_count=int(len(np.unique(p["ring"][a:b]))),
            firing_span_ms=1000 * float(offsets.max() - offsets.min()),
        )

    def truth(self):
        mj.mj_kinematics(self.model, self.data)
        return self.data.xpos[self.primary.base].copy(), self.data.xmat[
            self.primary.base
        ].reshape(3, 3).copy()

    def clearance_truth(self, name):
        """Post-command evaluator: shortest physical surface distance, capped at 2 m."""
        candidates = self.query_targets[name]
        a = self.my_colliders
        pos = self.data.geom_xpos
        bounds = self.model.geom_rbound
        lower = (
            np.linalg.norm(pos[a, None, :] - pos[candidates], axis=2)
            - bounds[a, None]
            - bounds[candidates]
        )
        flat = np.argsort(lower, axis=None)
        best = 2.0
        for index in flat:
            i, j = np.unravel_index(index, lower.shape)
            if lower[i, j] >= best:
                break
            best = min(
                best,
                mj.mj_geomDistance(
                    self.model, self.data, int(a[i]), int(candidates[j]), best, None
                ),
            )
        return float(best)

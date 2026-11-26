"""MuJoCo world in metres, using a bottom-left grid origin and the world Z=0 support."""

import math
from pathlib import Path

import mujoco as mj
import numpy as np

from .bounds import CollisionFootprint
from .controller import Go2Controller

ASSET_DIR = Path(__file__).resolve().parent / "assets"
SCENE_PATH = ASSET_DIR / "go2" / "scene.xml"
WEIGHTS_PATH = ASSET_DIR / "actor.npz"


def rectangles(mask):
    """Merge equal horizontal runs across adjacent rows, preserving holes."""
    previous = {}
    for row in range(mask.shape[0] + 1):
        occupied = (
            mask[row] if row < mask.shape[0] else np.zeros(mask.shape[1], dtype=bool)
        )
        edges = np.flatnonzero(np.diff(np.r_[False, occupied, False]))
        current = {tuple(pair) for pair in edges.reshape(-1, 2)}
        for start, end in sorted(previous.keys() - current):
            yield previous.pop((start, end)), row, start, end
        for run in current:
            previous.setdefault(run, row)


def build_model(scene_file, cells, origin, resolution, actors, *, render_layers=None):
    cells, origin = np.asarray(cells), np.asarray(origin, dtype=float)
    if cells.ndim != 2 or not 0 < cells.size <= 1_000_000:
        raise ValueError("Физическая сцена поддерживает до 1 000 000 ячеек карты")
    if not np.isin(cells, [-1, 0, 100]).all():
        raise ValueError("Карта содержит неизвестные коды занятости")
    if (
        origin.shape != (2,)
        or not np.isfinite(origin).all()
        or not math.isfinite(resolution)
        or resolution <= 0
    ):
        raise ValueError("Некорректные метры карты или начало координат")
    if len(actors) > 64 or any(not math.isfinite(r) or not 0 < r <= 20 for r in actors):
        raise ValueError("Некорректные радиусы или более 64 динамических объектов")
    static_count = 4  # the four outer boundaries are included in the scene limit
    spec = mj.MjSpec.from_file(str(scene_file))
    spec.visual.quality.offsamples = 1
    spec.visual.global_.offwidth = 640
    spec.visual.global_.offheight = 400
    nav_batches = []
    for kind, color, mask in [
        ("occupied", [0.22, 0.36, 0.39, 1], cells == 100),
        ("unknown", [0.21, 0.23, 0.24, 1], (cells != 0) & (cells != 100)),
    ]:
        centers, sizes = [], []
        for i, (r0, r1, c0, c1) in enumerate(rectangles(mask)):
            static_count += 1
            if static_count > 10_000:
                raise ValueError(
                    "Карта требует более 10 000 статических физических препятствий"
                )
            dx, dy = (c1 - c0) * resolution / 2, (r1 - r0) * resolution / 2
            if render_layers is not None:
                measured = render_layers[0] is not None
                half_z = 0.012 if measured else 0.3
                centers.append(
                    [
                        origin[0] + c0 * resolution + dx,
                        origin[1] + r0 * resolution + dy,
                        half_z,
                    ]
                )
                sizes.append([dx, dy, half_z])
                continue
            spec.worldbody.add_geom(
                name=f"{kind}_{i}",
                type=mj.mjtGeom.mjGEOM_BOX,
                pos=[
                    origin[0] + c0 * resolution + dx,
                    origin[1] + r0 * resolution + dy,
                    0.3,
                ],
                size=[dx, dy, 0.3],
                rgba=color,
            )
        if centers:
            rgba = (
                (
                    [0.16, 0.53, 0.58, 0.32]
                    if kind == "occupied"
                    else [0.32, 0.32, 0.38, 0.18]
                )
                if measured
                else color
            )
            nav_batches.append((kind, centers, sizes, rgba))
    w, h = cells.shape[1] * resolution, cells.shape[0] * resolution
    x0, y0 = origin
    for i, (x, y, dx, dy) in enumerate(
        (
            (x0 - 0.05, y0 + h / 2, 0.05, h / 2 + 0.1),
            (x0 + w + 0.05, y0 + h / 2, 0.05, h / 2 + 0.1),
            (x0 + w / 2, y0 - 0.05, w / 2 + 0.1, 0.05),
            (x0 + w / 2, y0 + h + 0.05, w / 2 + 0.1, 0.05),
        )
    ):
        spec.worldbody.add_geom(
            name=f"boundary_{i}",
            group=4 if render_layers is not None else 0,
            type=mj.mjtGeom.mjGEOM_BOX,
            pos=[x, y, 0.3],
            size=[dx, dy, 0.3],
            rgba=[0.22, 0.36, 0.39, 1],
        )
    for i, radius in enumerate(actors):
        body = spec.worldbody.add_body(name=f"actor_{i}", mocap=True, pos=[0, 0, -40])
        body.add_geom(
            name=f"dynamic_{i}",
            type=mj.mjtGeom.mjGEOM_CYLINDER,
            size=[radius, 0.38, 0],
            rgba=[0.98, 0.38, 0.13, 1],
        )
    if render_layers is not None:
        from .render_geometry import add_layers, box_batch

        cloud, voxel_size, zones = render_layers
        spec.geom("floor").group = 4
        for name, centers, sizes, rgba in nav_batches:
            box_batch(spec, name, centers, sizes, rgba, 4)
        add_layers(spec, cloud, voxel_size, cells, origin, resolution, zones)
    return spec.compile()


class World:
    """Go2 and map colliders in a single MuJoCo integration state.

    actors fixes the radius of each simulated object. tick receives world truth
    at the start of the control interval; tracking and planning live above it.
    """

    def __init__(
        self, scene_file, weights_file, cells, origin, resolution, actors, xy, yaw
    ):
        self.model = build_model(scene_file, cells, origin, resolution, actors)
        self.data = mj.MjData(self.model)
        self.controller = Go2Controller(self.model, self.data, weights_file)
        self.controller.initialize(xy, yaw)
        self.floor = self.model.geom("floor").id
        self.feet = {self.model.geom(leg).id for leg in ("FR", "FL", "RR", "RL")}
        root = self.model.body_rootid[self.model.geom_bodyid]
        self.robot = set(
            map(
                int,
                np.flatnonzero(
                    (root == self.controller.base)
                    & (
                        (self.model.geom_contype != 0)
                        | (self.model.geom_conaffinity != 0)
                    )
                ),
            )
        )
        self.actors = [
            (
                self.model.body(f"actor_{i}").mocapid.item(),
                self.model.geom(f"dynamic_{i}").id,
            )
            for i in range(len(actors))
        ]
        self.actor_geoms = {geom for mocap, geom in self.actors}
        self.footprint = CollisionFootprint(
            self.model, self.robot, self.controller.base
        )
        self.peak_radius = 0.0
        self.counters = {"foot_floor": 0, "body_floor": 0, "static": 0, "dynamic": 0}
        self.minimum_gap = None
        self.minimum_center_minus_actor_radius = None
        self.upright_min = 1.0
        self.min_height = self.max_height = float(self.data.qpos[2])
        self.max_planar_speed = 0.0
        self.fault = None
        self.minimum_static_gap = None
        self.walls = np.flatnonzero(
            (self.model.geom_bodyid == 0)
            & (self.model.geom_type == mj.mjtGeom.mjGEOM_BOX)
        )
        self.wall_centers = self.model.geom_pos[self.walls, :2]
        self.wall_half = self.model.geom_size[self.walls, :2]

    def tick(self, command, actors):
        """Advance 20ms; each actor is (x,y,vx,vy) at the start of this interval."""
        if len(actors) != len(self.actors):
            raise ValueError("Изменилась конфигурация физических объектов")
        if self.fault:
            raise RuntimeError(self.fault)
        self.controller.set_command(*command)
        # Broad phase once per tick. Reevaluate if the base moves outside this margin.
        nearby = None
        anchor = None
        for substep in range(self.controller.policy_steps):
            elapsed = self.controller.dt * substep
            for (mid, _), track in zip(self.actors, actors):
                if track is None:
                    self.data.mocap_pos[mid] = [0, 0, -40]
                else:
                    x, y, vx, vy = track
                    self.data.mocap_pos[mid] = [
                        x + elapsed * vx,
                        y + elapsed * vy,
                        0.38,
                    ]
            self.controller.substep()
            if not (
                np.isfinite(self.data.qpos).all()
                and np.isfinite(self.data.qvel).all()
                and np.isfinite(self.data.ctrl).all()
            ):
                self.fault = "nonfinite"
                raise RuntimeError("Неконечное состояние физической модели")
            height = float(self.data.qpos[2])
            self.min_height = min(self.min_height, height)
            self.max_height = max(self.max_height, height)
            self.max_planar_speed = max(
                self.max_planar_speed, float(np.linalg.norm(self.data.qvel[:2]))
            )
            self.peak_radius = max(self.peak_radius, self.footprint.radius(self.data))
            self.upright_min = min(
                self.upright_min, float(self.data.xmat[self.controller.base, 8])
            )
            xy = self.data.xpos[self.controller.base, :2]
            if nearby is None or np.linalg.norm(xy - anchor) > 0.05:
                anchor = xy.copy()
                delta = np.maximum(np.abs(self.wall_centers - xy) - self.wall_half, 0)
                separation = np.linalg.norm(delta, axis=1)
                if self.minimum_static_gap is None:
                    closest = int(self.walls[np.argmin(separation)])
                    self.minimum_static_gap = self._gap_to(closest)
                # 0.65 exceeds the permitted physical envelope (0.45) and tick drift.
                nearby = self.walls[
                    separation
                    <= self.minimum_static_gap + max(0.65, self.peak_radius + 0.12)
                ]
            for wall in nearby:
                self.minimum_static_gap = min(
                    self.minimum_static_gap, self._gap_to(int(wall))
                )
            for contact in self.data.contact:
                a, b = map(int, contact.geom)
                if (a in self.robot) == (b in self.robot):
                    continue
                mine, other = (a, b) if a in self.robot else (b, a)
                if other == self.floor:
                    key = "foot_floor" if mine in self.feet else "body_floor"
                else:
                    key = "dynamic" if other in self.actor_geoms else "static"
                self.counters[key] += 1
            for (_, geom), track in zip(self.actors, actors):
                if track is None:
                    continue
                center_distance = float(
                    np.linalg.norm(
                        self.data.geom_xpos[geom, :2]
                        - self.data.xpos[self.controller.base, :2]
                    )
                )
                radius = self.model.geom_size[geom, 0]
                center_gap = float(center_distance - radius)
                if (
                    self.minimum_center_minus_actor_radius is None
                    or center_gap < self.minimum_center_minus_actor_radius
                ):
                    self.minimum_center_minus_actor_radius = center_gap
                lower_bound = center_gap - self.peak_radius
                if self.minimum_gap is not None and lower_bound >= self.minimum_gap:
                    continue
                gap = min(
                    mj.mj_geomDistance(
                        self.model,
                        self.data,
                        part,
                        geom,
                        center_distance + radius + 5,
                        None,
                    )
                    for part in self.robot
                )
                self.minimum_gap = (
                    gap if self.minimum_gap is None else min(self.minimum_gap, gap)
                )

    def _gap_to(self, geom):
        limit = float(
            np.linalg.norm(
                self.data.geom_xpos[geom] - self.data.xpos[self.controller.base]
            )
            + np.linalg.norm(self.model.geom_size[geom])
            + 5
        )
        return min(
            mj.mj_geomDistance(self.model, self.data, part, geom, limit, None)
            for part in self.robot
        )

    @property
    def up_min(self):
        return self.upright_min

    @property
    def pose(self):
        rot = self.data.xmat[self.controller.base].reshape(3, 3)
        return {
            "x": float(self.data.qpos[0]),
            "y": float(self.data.qpos[1]),
            "yaw": math.atan2(rot[1, 0], rot[0, 0]),
        }

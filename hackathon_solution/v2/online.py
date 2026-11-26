"""Causal point-time compensation, incremental 3-D observations and navigation on an official replay."""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from inspection.planning import Grid, plan
from inspection.voxel_cloud import cloud_records, ply_header


def compensate(msg, points, mask, selected, history):
    """Constant body motion extrapolated from odometry already available at scan start."""
    stamp, current = selected
    field = next((f for f in msg.fields if f.name == "time" and f.datatype == 7), None)
    older = next(
        ((t, p) for t, p in reversed(history) if t <= stamp - 40_000_000), None
    )
    if field is None or older is None or stamp - older[0] > 120_000_000:
        return points
    dt = (stamp - older[0]) * 1e-9
    times = (
        np.ndarray(
            (msg.height, msg.width),
            dtype=">f4" if msg.is_bigendian else "<f4",
            buffer=msg.data,
            offset=field.offset,
            strides=(msg.row_step, msg.point_step),
        )
        .ravel()[mask]
        .astype(float)
    )
    if not np.isfinite(times).all() or np.any((times < 0) | (times > 0.105)):
        return points
    current_rotation = current[:3, :3]
    velocity = current_rotation.T @ (current[:3, 3] - older[1][:3, 3]) / dt
    omega = Rotation.from_matrix(older[1][:3, :3].T @ current_rotation).as_rotvec() / dt
    return (
        Rotation.from_rotvec(times[:, None] * omega).apply(points)
        + times[:, None] * velocity
    )


class OnlineMap:
    def __init__(self, reference, grid_file):
        self.reference = reference
        self.grid = Grid(grid_file)
        self.prior = self.grid.cells.copy()
        self.free_hits = np.zeros(self.prior.shape, dtype=np.uint8)
        self.occupied_until = np.zeros(self.prior.shape, dtype=np.int64)
        self.grid_changes = 0
        self.pending = {}
        self.confirmed = {}
        self.version = 0
        self.last_plan = self.last_publish = 0
        # Pick once from the immutable prior. The official second trajectory is unavailable to the planner.
        clearance = self.grid.clearance([])
        clear_cells = np.argwhere(clearance >= min(0.85, clearance.max()))
        if not len(clear_cells):
            raise ValueError("Prior grid contains no traversable goal")
        centers = self.grid.origin + (clear_cells[:, ::-1] + 0.5) * self.grid.resolution
        self.goal = centers[np.argmin(np.linalg.norm(centers - [0.0, 5.0], axis=1))]
        self.clear_after = 0

    def update_occupancy(self, world, distances, origin, sensor_ns):
        # Only near-horizontal beams spanning the dog's collision height carve free space.
        # The immutable first pass retains veto on occupied cells. Transient returns expire.
        observed = world[(world[:, 2] > 0.12) & (world[:, 2] < 0.70)]
        rays = observed - origin
        lengths = np.linalg.norm(rays, axis=1)
        if len(rays):
            steps = np.arange(0.18, min(8.0, lengths.max()) + 0.09, 0.09)
            indices, offsets = np.nonzero(steps[None, :] < lengths[:, None] - 0.20)
            samples = (
                origin[:2]
                + rays[indices, :2] * (steps[offsets] / lengths[indices])[:, None]
            )
            cells = np.floor(
                (samples - self.grid.origin) / self.grid.resolution
            ).astype(int)
            valid = (
                (cells[:, 0] >= 0)
                & (cells[:, 0] < self.grid.width)
                & (cells[:, 1] >= 0)
                & (cells[:, 1] < self.grid.height)
            )
            c, r = cells[valid].T
            self.free_hits[r, c] = np.minimum(self.free_hits[r, c], 5) + 1
            self.occupied_until[r, c] = np.maximum(
                0, self.occupied_until[r, c] - 300_000_000
            )
        ends = world[(distances > 0.14) & (world[:, 2] > 0.15) & (world[:, 2] < 1.1)]
        cells = np.floor(
            (ends[:, :2] - self.grid.origin) / self.grid.resolution
        ).astype(int)
        valid = (
            (cells[:, 0] >= 0)
            & (cells[:, 0] < self.grid.width)
            & (cells[:, 1] >= 0)
            & (cells[:, 1] < self.grid.height)
        )
        c, r = cells[valid].T
        self.occupied_until[r, c] = sensor_ns + 1_000_000_000
        current = self.prior.copy()
        current[(self.free_hits >= 3) & (self.prior == -1)] = 0
        current[self.occupied_until > sensor_ns] = 100
        self.grid_changes = int(np.count_nonzero(current != self.grid.cells))
        self.grid.cells = current

    def observe(self, world, position, sensor_ns, origin):
        if sensor_ns >= self.clear_after:
            self.pending = {
                k: v for k, v in self.pending.items() if sensor_ns - v[0] <= 12e9
            }
            self.clear_after = sensor_ns + 1_000_000_000
        distances = self.reference.tree.query(world)[0]
        self.update_occupancy(world, distances, origin, sensor_ns)
        novel = world[(distances > 0.14) & (world[:, 2] > -0.45) & (world[:, 2] < 3.2)]
        if not len(novel):
            self.version += bool(self.grid_changes)
            return 0
        cells, idx = np.unique(
            np.floor(novel / 0.12).astype(np.int32), axis=0, return_index=True
        )
        new = 0
        for cell, point in zip(map(tuple, cells), novel[idx]):
            if cell in self.confirmed:
                continue
            first = self.pending.get(cell)
            if first is None:
                self.pending[cell] = (sensor_ns, position.copy(), point.copy())
            elif (
                sensor_ns - first[0] >= 700_000_000
                and np.linalg.norm(first[1] - position) >= 0.18
            ):
                measured = (first[2] + point) * 0.5
                self.confirmed[cell] = measured
                del self.pending[cell]
                new += 1
        self.version += bool(new or self.grid_changes)
        return new

    def route(self, current, sensor_ns):
        if sensor_ns - self.last_plan < 500_000_000:
            return None
        self.last_plan = sensor_ns
        return plan(self.grid, current[:2], [self.goal], radius=0.30)

    def points(self):
        return np.asarray(list(self.confirmed.values()), dtype=np.float32).reshape(
            -1, 3
        )

    def save(self, directory):
        xyz = self.points()
        rgb = np.tile([49, 217, 141], (len(xyz), 1))
        records = cloud_records(xyz, rgb, np.full(len(xyz), 2))
        (Path(directory) / "online_delta.ply").write_bytes(
            ply_header(len(xyz), "measured additions from localized second bag")
            + records.tobytes()
        )
        np.savez_compressed(
            Path(directory) / "navigation-grid.npz",
            cells=self.grid.cells,
            origin=self.grid.origin,
            resolution=self.grid.resolution,
        )

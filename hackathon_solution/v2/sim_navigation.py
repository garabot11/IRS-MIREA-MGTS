"""Navigation inputs are onboard motion measurements, time-stamped returns, and the prior map."""

import math
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation
from inspection.planning import _route
from hackathon_solution.tracker import apply


class MotionEstimate:
    def __init__(self, xy, yaw):
        self.odom = np.eye(4)
        self.odom[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
        self.odom[:3, 3] = [*xy, 0.32]
        self.alignment = np.eye(4)
        self.last = {"odom": 0.0, "gyro": 0.0}
        self.pose = self.odom.copy()

    def predict(self, observations):
        for kind, stamp, value in observations:
            dt = stamp - self.last[kind]
            self.last[kind] = stamp
            if kind == "gyro":
                self.odom[:3, :3] = (
                    self.odom[:3, :3] @ Rotation.from_rotvec(value * dt).as_matrix()
                )
            else:
                self.odom[:3, 3] += self.odom[:3, :3] @ value * dt
        self.pose = self.alignment @ self.odom

    def observe(self, fixed_map, scan):
        structural = scan[np.linalg.norm(scan, axis=1) > 0.78]
        if len(structural) < 40:
            return False
        self.alignment, ok = fixed_map.refine(
            apply(self.odom, structural), self.odom, self.alignment, np.eye(4)
        )
        self.pose = self.alignment @ self.odom
        return ok


class OnlineMap:
    def __init__(self, grid, fixed_map, sensor_origin):
        self.grid = grid
        self.fixed = fixed_map
        self.sensor_origin = sensor_origin
        self.prior_clear = distance_transform_edt(grid.cells == 0) * grid.resolution
        self.revision = 0
        self.active = set()
        self.observations = {}
        self.seen_free = set()
        self.ever_occupied = set()
        self.cleared = set()
        self.last_novel = np.empty((0, 3))
        self.recent_hits = []
        self.tracks = []

    def integrate(self, points, pose, t, scan_id, moving=None):
        world = apply(pose, points)
        origin = apply(pose, self.sensor_origin)
        # First-return rays supply free-space evidence before the endpoint. Exclude
        # very near returns and floor-only cells from the footprint-height layer.
        rays = world - origin
        lengths = np.linalg.norm(rays, axis=1)
        close = (lengths > 0.35) & (lengths < 12)
        rays = rays[close]
        d = lengths[close]
        ds = np.arange(0.32, 6, 0.16)
        clear = origin + rays[:, None, :] / d[:, None, None] * ds[None, :, None]
        valid = (
            (ds[None, :] < d[:, None] - 0.20)
            & (clear[:, :, 2] > 0.13)
            & (clear[:, :, 2] < 0.60)
        )
        clear_keys = np.floor(
            (clear[valid, :2] - self.grid.origin) / self.grid.resolution
        ).astype(int)
        free_now = {tuple(k) for k in clear_keys}
        self.seen_free.update(free_now)
        eligible = (
            (world[:, 2] > 0.12)
            & (world[:, 2] < 1.85)
            & (lengths > 0.55)
            & (lengths < 7.0)
        )
        w = world[eligible]
        xy = np.floor((w[:, :2] - self.grid.origin) / self.grid.resolution).astype(int)
        valid = (
            (xy >= 0).all(axis=1)
            & (xy[:, 0] < self.grid.width)
            & (xy[:, 1] < self.grid.height)
        )
        w, xy = w[valid], xy[valid]
        w = w[self.prior_clear[xy[:, 1], xy[:, 0]] > 0.24]
        w = w[self.fixed.tree.query(w)[0] > 0.17] if len(w) else w
        self.last_novel = w
        self.recent_hits = [(tt, pts) for tt, pts in self.recent_hits if t - tt < 0.24]
        self.recent_hits.append((t, w))
        self.update_tracks(t)
        # Neural veto affects only permanent map output. Local collision
        # avoidance and the live occupancy layer always retain all measurements.
        veto = set()
        if moving is not None:
            veto = set(map(tuple, np.floor(world[moving] / 0.12).astype(int)))
        keys, ids = np.unique(np.floor(w / 0.12).astype(int), axis=0, return_index=True)
        hit_cells = set()
        for key, p in zip(map(tuple, keys), w[ids]):
            cell = tuple(
                np.floor((p[:2] - self.grid.origin) / self.grid.resolution).astype(int)
            )
            hit_cells.add(cell)
            state = self.observations.get(key)
            if state is None:
                self.observations[key] = dict(
                    p=p.copy(),
                    count=1,
                    first=t,
                    last=t,
                    scan=scan_id,
                    cell=cell,
                    miss=0,
                    persistent=False,
                    motion_blocked=key in veto,
                )
            else:
                state["motion_blocked"] |= key in veto
                state["p"] += (p - state["p"]) / (state["count"] + 1)
                state.update(count=state["count"] + 1, last=t, scan=scan_id, miss=0)
                if state["count"] >= 8 and t - state["first"] >= 2.1:
                    state["persistent"] = True
        current = set()
        for key, state in list(self.observations.items()):
            age = t - state["last"]
            cell = state["cell"]
            if cell in free_now and cell not in hit_cells and age > 0.16:
                state["miss"] += 1
            expires = (age > 1.3 and not state["persistent"]) or (
                age > 0.35 and state["miss"] >= 4
            )
            if expires:
                self.cleared.add(key)
                del self.observations[key]
            elif state["count"] >= 2 and state["p"][2] < 0.78 and age < 1.3:
                current.add(cell)
        if current != self.active:
            self.revision += 1
        self.active = current
        self.ever_occupied.update(current)

    def update_tracks(self, t):
        # Group sparse returns over the last quarter second, then associate blobs
        # by geometric proximity. Body IDs and actor poses are unavailable here.
        from scipy.spatial import cKDTree
        from scipy.sparse.csgraph import connected_components

        w = np.vstack([pts for _, pts in self.recent_hits])
        self.tracks = [tr for tr in self.tracks if t - tr["last"] < 0.70]
        if len(w) < 3:
            return
        tree = cKDTree(w[:, :2])
        graph = tree.sparse_distance_matrix(tree, 0.32, output_type="coo_matrix")
        n, labels = connected_components(graph, directed=False)
        used = set()
        for i in range(n):
            blob = w[labels == i]
            if len(blob) < 3:
                continue
            center = np.median(blob[:, :2], axis=0)
            options = [
                (np.linalg.norm(center - tr["center"]), j, tr)
                for j, tr in enumerate(self.tracks)
                if j not in used and np.linalg.norm(center - tr["center"]) < 0.58
            ]
            if options:
                _, j, tr = min(options, key=lambda x: x[0])
                history = [(tt, p) for tt, p in tr["history"] if t - tt < 0.78]
                history.append((t, center))
                dt = t - history[0][0]
                # Fit velocity over multiple observations; two noisy extremal
                # hits on opposite sides of a stationary surface cannot set it.
                if dt > 0.42 and len(history) >= 6:
                    times = np.array([tt for tt, _ in history])
                    times -= times.mean()
                    xy = np.array([p for _, p in history])
                    centered = xy - xy.mean(axis=0)
                    velocity = times @ centered / max(1e-6, times @ times)
                    residual = np.linalg.norm(
                        centered - times[:, None] * velocity, axis=1
                    ).mean()
                    tr["velocity"] = velocity if residual < 0.09 else np.zeros(2)
                tr.update(
                    center=center,
                    last=t,
                    history=history,
                    height=np.percentile(blob[:, 2], 90),
                )
            else:
                j = len(self.tracks)
                self.tracks.append(
                    dict(
                        center=center,
                        last=t,
                        history=[(t, center)],
                        velocity=np.zeros(2),
                        height=np.percentile(blob[:, 2], 90),
                    )
                )
            used.add(j)

    def crossing_risk(self, pose, t):
        xy = pose[:2, 3]
        forward = pose[:2, 0]
        forward = forward / np.linalg.norm(forward)
        lateral = np.array([-forward[1], forward[0]])
        for tr in self.tracks:
            velocity = tr["velocity"]
            # A moving return may enter the path before the next subgoal. Retain
            # clearance for the measured torso width, stance and tracking delay.
            moving = 0.19 < np.linalg.norm(velocity) < 1.75
            tall = (
                tr["height"] > 0.85
                and len(tr["history"]) >= 4
                and t - tr["history"][0][0] < 0.60
            )
            if not moving and not tall:
                continue
            centers = tr["center"] + np.arange(0, 2.0, 0.20)[:, None] * velocity
            delta = centers - xy
            if np.any(
                (delta @ forward > -0.10)
                & (delta @ forward < 1.94)
                & (abs(delta @ lateral) < 0.72)
            ):
                return True
        return False

    def records(self):
        return [
            v
            for v in self.observations.values()
            if v["persistent"] and not v["motion_blocked"]
        ]


class Planner:
    def __init__(self, grid, goal):
        self.grid = grid
        self.goal = np.asarray(goal)
        self.last_rev = None
        self.route = []
        self.free = None
        self.radius = 0.49
        self.read_revision = 0

    def command(self, pose, occupancy, revision):
        xy = pose[:2, 3]
        yaw = math.atan2(pose[1, 0], pose[0, 0])
        if self.last_rev != revision:
            cells = self.grid.cells.copy()
            for c, r in occupancy:
                if 0 <= c < self.grid.width and 0 <= r < self.grid.height:
                    cells[r, c] = 100
            self.free = (
                distance_transform_edt(cells == 0) * self.grid.resolution > self.radius
            )
            self.last_rev = revision
            self.read_revision = revision
        if np.linalg.norm(self.goal - xy) < 0.20:
            self.route = []
            return (0.0, 0.0, 0.0), "goal"
        if self.free is None or not self.free[self.grid.index(xy)]:
            self.route = []
            return (0.0, 0.0, 0.0), "yield"
        route = _route(self.grid, self.free, [xy.tolist(), self.goal.tolist()])
        if not route or len(route) < 2:
            self.route = []
            return (0.0, 0.0, 0.0), "yield"
        self.route = route
        waypoint = np.asarray(route[1])
        delta = waypoint - xy
        angle = math.remainder(math.atan2(delta[1], delta[0]) - yaw, math.tau)
        # Near sharp bends orient first to preserve lateral clearance.
        speed = min(0.47, np.linalg.norm(self.goal - xy)) * max(
            0.0, 1 - abs(angle) / 0.56
        )
        turn = np.clip(2.15 * angle, -0.90, 0.90)
        return (float(speed), 0.0, float(turn)), "detour" if len(
            route
        ) > 2 else "direct"

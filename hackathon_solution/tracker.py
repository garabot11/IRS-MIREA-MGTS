"""Causal registration to an immutable XYZ map. All transforms act on column vectors."""
from time import perf_counter

import cv2
import numpy as np
from scipy.linalg import solve
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from inspection.voxel_cloud import read_ply


def apply(transform, xyz):
    return xyz @ transform[:3, :3].T + transform[:3, 3]


def thin(points, cell):
    keys = np.floor(points / cell).astype(np.int32)
    _, indices = np.unique(keys, axis=0, return_index=True)
    return points[indices]


class FixedMap:
    def __init__(self, path):
        records = read_ply(path.read_bytes())
        self.xyz = np.column_stack([records[name] for name in ("x", "y", "z")]).astype(float)
        self.tree = cKDTree(self.xyz)
        self.walls = cKDTree(self.xyz[self.xyz[:, 2] > 0.15])
        # Estimate normals from measured neighbouring voxels. The eigengap
        # downweights corners and points without a well-defined tangent plane.
        self.normals = np.empty_like(self.xyz)
        self.confidence = np.empty(len(self.xyz))
        for start in range(0, len(self.xyz), 5000):
            sl = slice(start, start + 5000)
            _, neighbours = self.tree.query(self.xyz[sl], k=14)
            centered = self.xyz[neighbours]
            centered -= centered.mean(axis=1, keepdims=True)
            eigen, directions = np.linalg.eigh(centered.transpose(0, 2, 1) @ centered)
            self.normals[sl] = directions[:, :, 0]
            self.confidence[sl] = (eigen[:, 1] - eigen[:, 0]) / np.maximum(eigen[:, 2], 1e-10)

    def candidates(self, train, anchor):
        """Correlate rotated wall footprints against the entire map, one grid cell per candidate."""
        cell = 0.25
        walls = self.xyz[(self.xyz[:, 2] > 0.42) & (self.xyz[:, 2] < 1.9), :2]
        scan = train[(train[:, 2] > 0.42) & (train[:, 2] < 1.9), :2]
        if len(scan) < 5:
            return []
        xy = thin(walls, cell)
        origin = xy.min(axis=0) - 2 * cell
        indices = np.floor((xy - origin) / cell).astype(int)
        image = np.ones(tuple(indices.max(axis=0)[::-1] + 4), dtype=np.uint8)
        image[indices[:, 1], indices[:, 0]] = 0
        distance = cv2.distanceTransform(image, cv2.DIST_L2, 5) * cell
        field = np.exp(-0.5 * (distance / 0.32) ** 2)
        local = thin(scan, cell) - anchor[:2]
        half = int(np.ceil(np.linalg.norm(local, axis=1).max() / cell)) + 2
        peaks = []
        for angle in range(-180, 180, 8):
            rotation = Rotation.from_euler("z", angle, degrees=True).as_matrix()
            offsets = np.rint(local @ rotation[:2, :2].T / cell).astype(int) + half
            kernel = np.zeros((2 * half + 1, 2 * half + 1), dtype=np.float32)
            kernel[offsets[:, 1], offsets[:, 0]] = 1
            response = cv2.filter2D(field, -1, kernel / kernel.sum(), borderType=cv2.BORDER_CONSTANT)
            for _ in range(3):
                y, x = np.unravel_index(response.argmax(), response.shape)
                center = origin + cell * (np.array([x, y]) + 0.5)
                peaks.append((float(response[y, x]), angle, center))
                cv2.circle(response, (int(x), int(y)), 5, -1, -1)
        distinct = []
        for _, angle, center in sorted(peaks, key=lambda item: -item[0]):
            if any(np.linalg.norm(center - previous) < 0.85 and
                   abs((angle - yaw + 180) % 360 - 180) < 18 for previous, yaw, _ in distinct):
                continue
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_euler("z", angle, degrees=True).as_matrix()
            transform[:2, 3] = center - transform[:2, :2] @ anchor[:2]
            distinct.append((center, angle, transform))
            if len(distinct) == 12:
                break
        return [transform for _, _, transform in distinct]

    def align_start(self, train, transform):
        # With few non-floor returns, gravity and height are more reliable
        # than a free 6-D fit. Estimate only the horizontal start correction.
        for _ in range(24):
            world = apply(transform, train)
            dist, idx = self.walls.query(world)
            good = dist < min(1.0, np.quantile(dist, 0.90))
            if good.sum() < 24:
                break
            a, b = world[good, :2], self.walls.data[idx[good], :2]
            ca, cb = a.mean(axis=0), b.mean(axis=0)
            cross = (a - ca).T @ (b - cb)
            angle = np.arctan2(cross[0, 1] - cross[1, 0], np.trace(cross))
            increment = np.eye(4)
            increment[:3, :3] = Rotation.from_euler("z", angle).as_matrix()
            increment[:2, 3] = cb - increment[:2, :2] @ ca
            transform = increment @ transform
            if np.linalg.norm(increment[:2, 3]) < 0.001 and abs(angle) < 0.0005:
                break
        return transform

    def refine(self, world, base, previous, seed):
        transform = previous.copy()
        target = seed @ base
        for _ in range(4):
            src = apply(transform, world)
            current = transform @ base
            center = current[:3, 3]
            distance, idx = self.tree.query(src)
            good = (distance < 0.48) & (self.confidence[idx] > 0.16)
            structural = world[:, 2] > 0.17
            if (good.sum() < max(18, 0.18 * len(world)) or structural.sum() < 10 or
                    np.mean(distance[structural] < 0.48) < 0.40):
                return previous, False
            idx = idx[good]
            normal = self.normals[idx]
            error = np.sum((src[good] - self.xyz[idx]) * normal, axis=1)
            jac = np.column_stack((np.cross(src[good] - center, normal), normal))
            weights = self.confidence[idx] * np.minimum(1, 0.075 / np.maximum(abs(error), 1e-8))
            weighted = jac.T * weights
            # Regularize absolute drift along repeated corridor walls. Damping
            # also bounds each local update in the presence of moving returns.
            prior = np.r_[Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).as_rotvec(),
                          target[:3, 3] - center]
            strength = np.array([12., 12., 5., 2.4, 2.4, 8.])
            delta = solve(weighted @ jac + np.diag(strength + [8., 8., 1., 2.5, 2.5, 7.]),
                          -weighted @ error + strength * prior, assume_a="pos")
            angle = delta[:3] * min(1, 0.017 / max(1e-8, np.linalg.norm(delta[:3])))
            shift = delta[3:] * min(1, 0.045 / max(1e-8, np.linalg.norm(delta[3:])))
            update = np.eye(4)
            update[:3, :3] = Rotation.from_rotvec(angle).as_matrix()
            update[:3, 3] = center + shift - update[:3, :3] @ center
            transform = update @ transform
            if np.linalg.norm(delta) < 0.001:
                break
        displacement = np.linalg.norm(apply(transform, base[:3, 3]) - apply(previous, base[:3, 3]))
        return (transform, True) if displacement < 0.18 else (previous, False)


class Tracker:
    def __init__(self, fixed_map, mode):
        self.map = fixed_map
        self.mode = mode
        self.prefix = []
        self.anchor = None
        self.seed = None
        self.transform = None
        self.bootstrap = {}
        self.count = 0
        self.search_count = 0
        self.rejected = 0
        self.segment = 0

    def lose_fix(self):
        self.seed = self.transform = self.anchor = None
        self.prefix.clear()
        self.search_count = self.rejected = 0

    def initialize(self, world, base):
        self.search_count += 1
        if self.anchor is None:
            self.anchor = base[:3, 3].copy()
        self.prefix.append(world)
        if len(self.prefix) > 168:
            del self.prefix[:2]
        if self.search_count != 12 and (self.search_count < 24 or self.search_count % 24):
            return
        train = thin(np.vstack(self.prefix[::2]), 0.18)
        valid = thin(np.vstack(self.prefix[1::2]), 0.16)
        train, valid = train[train[:, 2] > 0.22], valid[valid[:, 2] > 0.22]
        if len(train) < 35 or len(valid) < 35:
            return
        candidates = self.map.candidates(train, self.anchor)
        if self.mode == "seeded":
            candidates.insert(0, np.eye(4))
        scores = []
        for index, candidate in enumerate(candidates):
            transform = self.map.align_start(train, candidate)
            dist = self.map.walls.query(apply(transform, valid))[0]
            scores.append((float(np.minimum(dist, 0.5).mean()), float((dist < 0.25).mean()),
                           transform, index))
        if not scores:
            return
        scores.sort(key=lambda s: s[0])
        best = scores[0]
        if self.mode == "seeded":
            # The team documents a weak shared-odometry-origin assumption.
            # It is a hypothesis that also has to fit the observed geometry.
            best = next(s for s in scores if s[3] == 0)
        alternative = next((s for s in scores if np.linalg.norm(
            apply(s[2], self.anchor) - apply(best[2], self.anchor)) > 0.85 or
            Rotation.from_matrix(s[2][:3, :3] @ best[2][:3, :3].T).magnitude() > 0.30), None)
        margin = alternative[0] - best[0] if alternative else 0.
        self.bootstrap = dict(mode=self.mode, prefix_scans=len(self.prefix), train=len(train),
                              validation=len(valid), capped_residual_m=best[0], support=best[1],
                              alternative_margin_m=margin)
        accepted = best[0] < 0.18 and best[1] >= 0.70
        if self.mode == "global":
            accepted = accepted and np.linalg.norm(base[:3, 3] - self.anchor) > 1.5 and margin > 0.035
        else:
            def inside_prior(transform):
                shift = np.linalg.norm(apply(transform, self.anchor) - self.anchor)
                return shift <= 1.5 and Rotation.from_matrix(transform[:3,:3]).magnitude() <= np.deg2rad(30)
            # Distant lookalike corridors are genuine global ambiguity. In
            # seeded operation their poses are outside the declared prior.
            admissible = [s for s in scores if inside_prior(s[2])]
            accepted = accepted and inside_prior(best[2]) and best[0] <= min(
                (s[0] for s in admissible), default=0.) + 0.015
            self.bootstrap.update(prior_radius_m=1.5, prior_yaw_deg=30, admissible=len(admissible))
        if accepted:
            self.seed = best[2]
            self.transform = self.seed.copy()
            self.prefix.clear()
            self.segment += 1

    def step(self, base, points):
        self.count += 1
        world = apply(base, points)
        row = dict(bootstrap_ms=0., registration_ms=0., quality_ms=0., holdout=False)
        if self.seed is None:
            start = perf_counter()
            self.initialize(world, base)
            row["bootstrap_ms"] = (perf_counter() - start) * 1000
            if self.seed is None:
                return None, dict(row, state="SEARCHING")
        elif (self.count - 1) % 5 == 0:
            # Whole scans excluded from estimation; evaluation cannot influence future poses.
            row["holdout"] = True
        state = "PREDICTED" if row["holdout"] else "TRACKING"
        if not row["holdout"]:
            start = perf_counter()
            self.transform, accepted = self.map.refine(world, base, self.transform, self.seed)
            row["registration_ms"] = (perf_counter() - start) * 1000
            self.rejected = 0 if accepted else self.rejected + 1
            if not accepted:
                state = "DEGRADED"
            if self.rejected >= 14:
                self.lose_fix()
                return None, dict(row, state="SEARCHING")
        pose = self.transform @ base
        if row["holdout"]:
            start = perf_counter()
            walls = world[world[:, 2] > 0.17]
            if len(walls) > 8:
                for name, transform in (("raw", np.eye(4)), ("aligned", self.seed),
                                        ("localized", self.transform)):
                    distances = self.map.tree.query(apply(transform, walls))[0]
                    row[name + "_residual_m"] = float(np.minimum(distances, 0.60).mean())
            row["quality_ms"] = (perf_counter() - start) * 1000
        row["odom_correction_m"] = float(np.linalg.norm(pose[:3, 3] - apply(self.seed, base[:3, 3])))
        return pose, dict(row, state=state)

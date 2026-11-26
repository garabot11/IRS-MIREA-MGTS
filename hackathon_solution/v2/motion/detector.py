"""Online scores for moving lidar returns from a fixed map and a short history of registered scans."""

from collections import deque
import numpy as np
from scipy.spatial import cKDTree


class TemporalFeatures:
    def __init__(self, reference):
        self.reference = reference
        self.history = deque(maxlen=12)

    def step(self, base_points, pose):
        world = base_points @ pose[:3, :3].T + pose[:3, 3]
        tree = cKDTree(world)
        prior = self.reference.query(world)[0]
        distances, indices = tree.query(world, k=8)
        distances = distances[:, 1:]
        neighbors = world[indices]
        features = [
            np.clip(prior, 0, 1),
            np.clip(world[:, 2], -0.5, 2),
            np.linalg.norm(base_points, axis=1).clip(0, 10) / 5,
            distances[:, 1].clip(0, 1),
            distances[:, -1].clip(0, 2),
            np.std(neighbors[:, :, 2], axis=1).clip(0, 0.5),
            np.mean(prior[indices], axis=1).clip(0, 1),
        ]
        for lag in (1, 3, 6, 12):
            if len(self.history) >= lag:
                earlier, old_tree = self.history[-lag]
                d, closest = old_tree.query(world)
                relative = world - earlier[closest]
                features.extend(
                    [
                        d.clip(0, 1),
                        np.linalg.norm(relative[:, :2], axis=1).clip(0, 1),
                        abs(relative[:, 2]).clip(0, 1),
                        np.median(d[indices], axis=1).clip(0, 1),
                    ]
                )
            else:
                features.extend([np.ones(len(world)) for _ in range(4)])
        self.history.append((world, tree))
        return np.column_stack(features).astype("f4")


class MotionDetector:
    def __init__(self, reference_tree, weights):
        self.features = TemporalFeatures(reference_tree)
        with np.load(weights, allow_pickle=False) as data:
            self.params = {key: data[key] for key in data.files}
        self.threshold = float(self.params["threshold"])

    def score(self, base_points, pose):
        x = self.features.step(base_points, pose)
        p = self.params
        for k in range(2):
            x = np.maximum(0, x @ p[f"w{k}"] + p[f"b{k}"])
        logits = (x @ p["w2"] + p["b2"]).ravel()
        return 1 / (1 + np.exp(-np.clip(logits, -50, 50)))

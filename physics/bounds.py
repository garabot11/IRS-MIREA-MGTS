"""Conservative radius enclosing each simulated contact shape projected onto XY."""

import math

import mujoco as mj
import numpy as np


class CollisionFootprint:
    def __init__(self, model, ids, base):
        self.model = model
        self.base = base
        ids = np.asarray(sorted(ids))
        self.groups = [
            (kind, ids[model.geom_type[ids] == kind])
            for kind in np.unique(model.geom_type[ids])
        ]
        directions = np.arange(32) * (2 * math.pi / 32)
        self.directions = np.vstack(
            [np.cos(directions), np.sin(directions), np.zeros(32)]
        )

    def radius(self, data):
        maximum = 0.0
        for kind, ids in self.groups:
            size = self.model.geom_size[ids]
            axes = (
                np.transpose(data.geom_xmat[ids].reshape(-1, 3, 3), (0, 2, 1))
                @ self.directions
            )
            if kind == mj.mjtGeom.mjGEOM_SPHERE:
                support = size[:, 0, None]
            elif kind == mj.mjtGeom.mjGEOM_BOX:
                support = (size[:, :, None] * abs(axes)).sum(axis=1)
            elif kind == mj.mjtGeom.mjGEOM_CYLINDER:
                support = (
                    np.hypot(axes[:, 0], axes[:, 1]) * size[:, 0, None]
                    + abs(axes[:, 2]) * size[:, 1, None]
                )
            else:
                raise ValueError(f"Неподдерживаемая физическая геометрия {kind}")
            offsets = (data.geom_xpos[ids] - data.xpos[self.base]) @ self.directions
            maximum = max(maximum, float((offsets + support).max()))
        return maximum / math.cos(math.pi / self.directions.shape[1])

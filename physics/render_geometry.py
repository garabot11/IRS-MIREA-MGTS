"""Static colour batches rendered in the same MuJoCo depth buffer as Go2."""

import cv2
import mujoco as mj
import numpy as np

from .scene import mesh_batches
from .world import rectangles

# Counterclockwise triangles looking at each cube face from outside.
CORNERS = np.array(
    [
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ]
)
TRIANGLES = np.array(
    [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [3, 7, 6],
        [3, 6, 2],
        [0, 4, 7],
        [0, 7, 3],
        [1, 2, 6],
        [1, 6, 5],
    ]
)


def box_batch(spec, name, centers, half_sizes, color, group):
    vertices = np.asarray(centers)[:, None, :] + CORNERS * np.asarray(
        half_sizes
    ).reshape(-1, 1, 3)
    faces = TRIANGLES[None, :, :] + (8 * np.arange(len(centers)))[:, None, None]
    spec.add_mesh(
        name=name,
        uservert=vertices.astype(np.float32).ravel(),
        userface=faces.astype(np.int32).ravel(),
        smoothnormal=False,
        maxhullvert=16,
    )
    spec.worldbody.add_geom(
        name=name,
        type=mj.mjtGeom.mjGEOM_MESH,
        meshname=name,
        contype=0,
        conaffinity=0,
        group=group,
        rgba=color,
    )


def add_layers(spec, cloud, voxel_size, cells, origin, resolution, zones):
    if cloud is not None:
        # Hide the decorative ground plane wherever it would cut the measured room.
        spec.geom("floor").group = 5
        tiles, labels = mesh_batches(cloud)
        for i, tile in enumerate(tiles):
            group = cloud[labels == i]
            box_batch(
                spec,
                f"measured_{i}",
                group["center"],
                group["half_size"],
                [*(tile[2:] / 255), 1],
                1,
            )
    if zones:
        mask = np.zeros(cells.shape, np.uint8)
        for zone in zones:
            xy = zone["points"] if isinstance(zone, dict) else zone
            pixels = ((np.asarray(xy) - origin) / resolution - 0.5) * 256
            cv2.fillPoly(
                mask,
                [np.clip(np.rint(pixels), -(2**29), 2**29).astype(np.int32)],
                1,
                shift=8,
            )
        centers, sizes = [], []
        for r0, r1, c0, c1 in rectangles(mask.astype(bool)):
            centers.append(
                [
                    origin[0] + (c0 + c1) * resolution / 2,
                    origin[1] + (r0 + r1) * resolution / 2,
                    0.025,
                ]
            )
            sizes.append(
                [(c1 - c0) * resolution / 2, (r1 - r0) * resolution / 2, 0.012]
            )
        if centers:
            box_batch(
                spec, "navigation_zones", centers, sizes, [0.85, 0.27, 0.12, 0.65], 4
            )

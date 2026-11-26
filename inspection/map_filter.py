"""Evaluate isolated lidar returns while preserving repeated hits and nearby surfaces."""

import math

import numpy as np
from scipy.spatial import cKDTree


SUPPORT_RADII = (1.25, 1.75, 2.5)
DEFAULT_SUPPORT_RADIUS = 1.75
FILTER_ALGORITHM = "temporal_spatial_support_v1"


def normalize_map_filter(value=None):
    if value is None:
        return {"support_radius_voxels": DEFAULT_SUPPORT_RADIUS}
    if not isinstance(value, dict) or set(value) != {"support_radius_voxels"}:
        raise ValueError("Для фильтра карты задайте support_radius_voxels.")
    radius = value["support_radius_voxels"]
    if type(radius) not in (float, int) or radius not in SUPPORT_RADII:
        raise ValueError(
            "Радиус поддержки должен быть 1.25, 1.75 или 2.5 шага вокселя."
        )
    return {"support_radius_voxels": float(radius)}


def filter_supported_voxels(
    xyz, visits, voxel_size_m, support_radius_voxels=DEFAULT_SUPPORT_RADIUS
):
    """Evaluate all presets on the same centroids and independent observation windows."""
    radius = normalize_map_filter({"support_radius_voxels": support_radius_voxels})[
        "support_radius_voxels"
    ]
    if not math.isfinite(voxel_size_m) or voxel_size_m <= 0:
        raise ValueError("Некорректный метрический шаг вокселя.")
    repeated = visits >= 2
    single = np.flatnonzero(~repeated)
    distances = np.full(len(xyz), np.inf)
    if len(single) and len(xyz) > 1:
        tree = cKDTree(xyz)
        for start in range(0, len(single), 65536):
            indices = single[start : start + 65536]
            # The nearest point is the query itself. Ask only for the second distance.
            nearest, _ = tree.query(xyz[indices], k=[2])
            distances[indices] = nearest[:, 0]
    profiles = []
    for candidate in SUPPORT_RADII:
        supported = repeated | (distances <= candidate * voxel_size_m)
        points = xyz[supported]
        profiles.append(
            dict(
                support_radius_voxels=candidate,
                retained_voxels=len(points),
                rejected_voxels=len(xyz) - len(points),
                bounds_m=dict(
                    min=points.min(axis=0).tolist(), max=points.max(axis=0).tolist()
                )
                if len(points)
                else None,
            )
        )
    keep = repeated | (distances <= radius * voxel_size_m)
    retained = int(keep.sum())
    return keep, dict(
        algorithm=FILTER_ALGORITHM,
        support_radius_voxels=radius,
        voxel_size_m=voxel_size_m,
        input_voxels=len(xyz),
        one_window_voxels=len(single),
        repeated_voxels=len(xyz) - len(single),
        retained_voxels=retained,
        rejected_voxels=len(xyz) - retained,
        profiles=profiles,
    )

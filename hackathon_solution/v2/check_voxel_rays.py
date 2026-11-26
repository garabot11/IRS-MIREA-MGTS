"""Check the optional CPU ray tracer against NumPy on all recorded firing templates."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from inspection.voxel_cloud import read_ply
from .sim_world import VoxelSurface


def compare(map_path, sensor_dir):
    points = read_ply(map_path.read_bytes())
    xyz = np.column_stack([points[k] for k in "xyz"])
    surface = VoxelSurface(xyz, 0.07)
    if surface.backend != "native":
        raise RuntimeError(
            "Run this check in the compiled physics image with KONTUR_VOXEL_RAYS=native"
        )
    rng = np.random.default_rng(2026)
    ray_count = 0
    reference_s = native_s = 0.0
    bank_path = sensor_dir / "rays.npz"
    with np.load(bank_path, allow_pickle=False) as bank:
        offsets = bank["offsets"]
        directions = bank["directions_base"].astype(np.float64)
        for a, b in zip(offsets[:-1], offsets[1:]):
            for first in range(a, b, 80):
                rays = directions[first : min(first + 80, b)]
                origin = np.array(
                    [
                        rng.normal(2.75, 0.25),
                        rng.uniform(3.5, 9.0),
                        rng.uniform(0.2, 0.5),
                    ]
                )
                limits = rng.uniform(0.4, 36.6, len(rays))
                limits[::8] = 36.6
                before = perf_counter()
                reference = surface.nearest_numpy(origin, rays, limits)
                after_reference = perf_counter()
                native = surface.nearest(origin, rays, limits)
                after_native = perf_counter()
                if not np.array_equal(reference.view("u8"), native.view("u8")):
                    raise ValueError(
                        f"First-hit range differs in template ray batch starting at {first}"
                    )
                reference_s += after_reference - before
                native_s += after_native - after_reference
                ray_count += len(rays)
    # Source coordinates on cell edges, tangent rays and zero/clipped ranges
    # exercise cases random samples in an indoor scene may otherwise miss.
    border = VoxelSurface(
        np.array([[0.0, 0.0, 0.0], [0.14, 0.0, 0.0], [-0.14, 0.07, 0.14]]), 0.07
    )
    unit = np.vstack((np.eye(3), -np.eye(3)))
    for origin in (
        np.zeros(3),
        np.array([-0.30, 0.07, 0.0]),
        np.array([0.14, 0.0, -0.30]),
    ):
        for max_range in (0.0, 0.30, 0.3336, 0.40, 36.6):
            ranges = np.full(6, max_range)
            if not np.array_equal(
                border.nearest_numpy(origin, unit, ranges),
                border.nearest(origin, unit, ranges),
            ):
                raise ValueError("Voxel boundary mismatch")
    return {
        "map_sha256": hashlib.sha256(map_path.read_bytes()).hexdigest(),
        "rays_sha256": hashlib.sha256(bank_path.read_bytes()).hexdigest(),
        "seed": 2026,
        "templates": len(offsets) - 1,
        "bit_identical_template_rays": ray_count,
        "additional_boundary_rays": 90,
        "numpy_s": reference_s,
        "native_s": native_s,
        "microbenchmark_speedup": reference_s / native_s,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("map_ply", type=Path)
    p.add_argument("sensor", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    summary = json.dumps(compare(args.map_ply, args.sensor), indent=2) + "\n"
    if args.output:
        args.output.write_text(summary)
    print(summary, end="")

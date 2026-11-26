"""Optional CPU voxel ray traversal on the same discrete samples as the NumPy reference."""

import ctypes
import os
from pathlib import Path

import numpy as np

LIBRARY = "libkontur_voxel_rays.so"


class NativeVoxelRays:
    def __init__(self, library, grid, minimum, edge):
        self.grid = np.ascontiguousarray(grid, dtype=np.bool_)
        self.shape = np.array(grid.shape, dtype=np.int64)
        self.minimum = np.ascontiguousarray(minimum, dtype=np.int64)
        self.edge = edge
        self.step = edge * 0.48
        self.trace = library.kontur_voxel_first_hit
        array = np.ctypeslib.ndpointer
        doubles = array(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
        triple = array(dtype=np.float64, ndim=1, shape=(3,), flags="C_CONTIGUOUS")
        dimensions = array(dtype=np.int64, ndim=1, shape=(3,), flags="C_CONTIGUOUS")
        self.trace.argtypes = (
            array(dtype=np.bool_, ndim=3, flags="C_CONTIGUOUS"),
            dimensions,
            dimensions,
            ctypes.c_double,
            triple,
            array(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),
            doubles,
            doubles,
            ctypes.c_size_t,
            ctypes.c_size_t,
        )
        self.trace.restype = None

    def nearest(self, center, directions, max_ranges):
        rays = np.ascontiguousarray(directions, dtype=np.float64)
        limits = np.array(max_ranges, dtype=np.float64, order="C", copy=True)
        if rays.shape != (len(limits), 3):
            raise ValueError("One XYZ direction is required for each ray range")
        # Preserve both levels of the reference sample arithmetic and first-hit
        # comparison. Single arithmetic progressions can round voxel edges differently.
        starts = np.arange(0.30, limits.max(initial=0.3), 32 * self.step)
        samples = (starts[:, None] + np.arange(32) * self.step).ravel()
        self.trace(
            self.grid,
            self.shape,
            self.minimum,
            self.edge,
            np.ascontiguousarray(center, dtype=np.float64),
            rays,
            limits,
            samples,
            len(rays),
            len(samples),
        )
        return limits


def load_native_rays(grid, minimum, edge):
    backend = os.environ.get("KONTUR_VOXEL_RAYS", "auto")
    if backend == "numpy":
        return None
    if backend not in ("auto", "native"):
        raise ValueError("KONTUR_VOXEL_RAYS must be auto, native or numpy")
    paths = (Path(__file__).with_name(LIBRARY), Path("/usr/local/lib") / LIBRARY)
    for path in paths:
        try:
            library = ctypes.CDLL(str(path))
            return NativeVoxelRays(library, grid, minimum, edge)
        except (OSError, AttributeError):
            continue
    if backend == "native":
        raise RuntimeError(
            "Native voxel ray library unavailable. Build the physics image or select numpy."
        )
    return None

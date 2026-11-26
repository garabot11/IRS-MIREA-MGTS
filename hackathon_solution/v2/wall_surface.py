"""Build a visual wall mesh from an inspection PLY in its original metric frame."""

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
from scipy.spatial import cKDTree, Delaunay

from inspection.voxel_cloud import read_ply


# Tuned to the 7 cm measured prior. Every vertex remains an original return.
MAX_EDGE = 0.26
NEIGHBOURHOODS = ((10, 0.175), (16, 0.25), (28, 0.34))
PATCH_DEPTH = 0.24
PATCH_MARGIN = 0.14
PROJECTED_CELL = 0.10
SUPPORT_LIMIT = 0.09
SUBDIVISIONS = 6
FACET = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])


class NoWallSupport(ValueError):
    """A valid measured cloud has insufficient support for a vertical mesh."""


def vertical_samples(xyz):
    tree = cKDTree(xyz)
    scales = [(k, radius) for k, radius in NEIGHBOURHOODS if k <= len(xyz)]
    distances, neighbours = tree.query(xyz, k=scales[-1][0])
    best = np.zeros(len(xyz))
    normals = np.zeros_like(xyz)
    for count, radius in scales:
        local = xyz[neighbours[:, :count]].astype(float)
        local -= local.mean(axis=1, keepdims=True)
        variances, vectors = np.linalg.eigh(local.transpose(0, 2, 1) @ local)
        small, middle, large = variances.T
        normal = vectors[:, :, 0]
        # The fit rewards a thin plane with support in both tangent directions.
        quality = (1 - small / np.maximum(middle, 1e-9)) * (
            middle.clip(min=0) / np.maximum(large, 1e-9)
        ) ** 0.25
        select = (
            (quality > best) & (xyz[:, 2] > 0.01) & (distances[:, count - 1] <= radius)
            & (small < 0.40 * middle) & (middle > 0.15 * large)
            & (np.abs(normal[:, 2]) < 0.40)
        )
        normals[select] = normal[select]
        best[select] = quality[select]
    keep = best > 0
    return xyz[keep], normals[keep]


def supported_faces(points, normals):
    faces = []
    # Project near-X and near-Y facing samples separately. Overlapping depth
    # slabs keep opposite sides of the corridor out of the same triangulation.
    axes = np.argmax(np.abs(normals[:, :2]), axis=1)
    for axis in (0, 1):
        candidates = np.flatnonzero(axes == axis)
        depths = points[candidates, axis]
        for band in np.unique(np.floor(depths / PATCH_DEPTH).astype(int)):
            start = band * PATCH_DEPTH
            ids = candidates[(depths >= start - PATCH_MARGIN) & (depths < start + PATCH_DEPTH + PATCH_MARGIN)]
            # Keep one measured vertex per projected cell, preferring the
            # sample nearest the local depth slab's center.
            bins = np.floor(points[ids][:, (1 - axis, 2)] / PROJECTED_CELL).astype(int)
            order = np.lexsort((np.abs(points[ids, axis] - start - PATCH_DEPTH / 2), bins[:, 0], bins[:, 1]))
            sorted_bins = bins[order]
            first = np.r_[True, np.any(sorted_bins[1:] != sorted_bins[:-1], axis=1)]
            ids = ids[order[first]]
            projected = points[ids][:, (1 - axis, 2)]
            if len(ids) < 3 or np.linalg.matrix_rank(projected - projected[0]) < 2:
                continue
            triangles = ids[Delaunay(projected).simplices]
            verts = points[triangles]
            center_depth = verts[:, :, axis].mean(axis=1)
            edges = verts - np.roll(verts, -1, axis=1)
            cross = np.cross(edges[:, 0], edges[:, 1])
            lengths = np.linalg.norm(cross, axis=1)
            direction = cross / np.maximum(lengths[:, None], 1e-12)
            agreement = np.abs(np.einsum("fi,fki->fk", direction, normals[triangles]))
            valid = (
                (center_depth >= start) & (center_depth < start + PATCH_DEPTH)
                & (np.linalg.norm(edges, axis=2).max(axis=1) <= MAX_EDGE)
                & (lengths > 1e-5) & (np.abs(direction[:, 2]) <= 0.40)
                & (agreement.min(axis=1) > 0.70)
            )
            faces.append(triangles[valid])
    if not faces or not any(len(block) for block in faces):
        raise NoWallSupport("No locally supported vertical triangles above z=0")
    return np.concatenate(faces)


def check_interior(points, faces):
    tree = cKDTree(points)
    vertices = points[faces]
    worst = np.zeros(len(faces))
    for i in range(SUBDIVISIONS + 1):
        for j in range(SUBDIVISIONS + 1 - i):
            k = SUBDIVISIONS - i - j
            if SUBDIVISIONS in (i, j, k):
                continue  # Original vertices have zero distance to the cloud.
            active = np.flatnonzero(worst <= SUPPORT_LIMIT)
            sample = (i * vertices[active, 0] + j * vertices[active, 1] + k * vertices[active, 2]) / SUBDIVISIONS
            worst[active] = np.maximum(worst[active], tree.query(sample)[0])
    accepted = worst <= SUPPORT_LIMIT
    if not accepted.any():
        raise NoWallSupport("No triangles passed the interior support check")
    return faces[accepted], float(worst[accepted].max())


def convert(source: Path, output: Path):
    if source.resolve() == output.resolve():
        raise ValueError("Source PLY and output STL must be different files")
    data = source.read_bytes()
    records = read_ply(data)
    if len(records) < NEIGHBOURHOODS[0][0]:
        raise NoWallSupport("Too few measured points to establish a wall")
    xyz = np.column_stack([records[name] for name in "xyz"])
    points, normals = vertical_samples(xyz)
    faces, support_max = check_interior(points, supported_faces(points, normals))
    vertices = points[faces]
    edges = vertices - np.roll(vertices, -1, axis=1)
    cross = np.cross(edges[:, 0], edges[:, 1])
    areas2 = np.linalg.norm(cross, axis=1)
    # Prefer facing the interior in renderers that cull back faces.
    inward = np.median(xyz, axis=0) - vertices.mean(axis=1)
    flip = np.einsum("ij,ij->i", cross, inward) < 0
    vertices[flip] = vertices[flip][:, [0, 2, 1]]
    cross[flip] *= -1
    facets = np.zeros(len(vertices), dtype=FACET)
    facets["normal"] = cross / areas2[:, None]
    facets["vertices"] = vertices
    centers = vertices.mean(axis=1)
    center_distance = cKDTree(xyz).query(centers)[0]
    payload = (
        b"Measured vertical patches in map frame, metres".ljust(80, b" ")
        + struct.pack("<I", len(facets)) + facets.tobytes()
    )
    output.write_bytes(payload)
    longest_edge = float(np.linalg.norm(edges, axis=2).max())
    return {
        "source_sha256": hashlib.sha256(data).hexdigest(),
        "mesh_sha256": hashlib.sha256(payload).hexdigest(),
        "frame": "map", "units": "m", "source_points": len(xyz),
        "vertical_samples": len(points), "used_measured_vertices": int(np.unique(faces).size),
        "triangles": len(facets), "triangle_area_sum_m2": float(areas2.sum() / 2),
        "max_edge_m": longest_edge,
        "support_samples_per_triangle": (SUBDIVISIONS + 1) * (SUBDIVISIONS + 2) // 2,
        "max_sample_to_vertical_measurement_m": support_max,
        # A point in a subtriangle is within longest_edge/(6*sqrt(3))
        # of a sampled vertex. Distance to the cloud is 1-Lipschitz.
        "whole_surface_support_bound_m": support_max + longest_edge / (SUBDIVISIONS * np.sqrt(3)),
        "centroid_to_measurement_m": dict(zip(("p50", "p95", "max"), np.percentile(center_distance, [50, 95, 100]))),
        "z_range_m": [float(vertices[:, :, 2].min()), float(vertices[:, :, 2].max())],
    }


def save_map_layer(directory, result):
    """Build the display surface in the same staging directory as its measured PLY."""
    mesh = directory / "wall-surface.stl"
    try:
        metrics = convert(directory / "map3d.ply", mesh)
    except NoWallSupport as exc:
        layer = {"status": "insufficient_support", "reason": str(exc)}
    else:
        layer = {"status": "ready", "mesh": mesh.name, **metrics}
    result["wall_surface"] = layer
    (directory / "wall-support.json").write_text(json.dumps(layer, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("map_ply", type=Path)
    parser.add_argument("output_stl", type=Path)
    args = parser.parse_args()
    print(json.dumps(convert(args.map_ply, args.output_stl), indent=2))


if __name__ == "__main__":
    main()

"""Immutable, job-bound visual inputs. Only the separate render model consumes these voxels."""

import json
import math
import os
import re
import stat
from hashlib import file_digest, sha256
from pathlib import Path

import numpy as np

from inspection.voxel_cloud import SIMULATOR_VOXEL_RECIPE, read_ply

MAX_SOURCE_BYTES = 48 * 1024**2
MAX_RENDER_CELLS = 18000
IDENTITY = np.eye(4).tolist()
COLLISION = dict(recipe="grid_prisms_v1", floor_z_m=0, measured_terrain=False)
CLOUD_DTYPE = np.dtype(
    [("center", "<f4", (3,)), ("half_size", "<f4", (3,)), ("rgb", "u1", (3,))]
)


class SceneError(ValueError):
    pass


def bounds(xyz):
    return dict(min=xyz.min(axis=0).tolist(), max=xyz.max(axis=0).tolist())


def grid_bounds(grid):
    origin = np.array(grid["origin"])
    return dict(
        min=origin.tolist(),
        max=(origin + np.array(grid["shape"])[::-1] * grid["resolution"]).tolist(),
    )


def legacy_scene(identity):
    return dict(
        schema_version=0,
        legacy=True,
        job_id=identity.get("job_id"),
        source_id=None,
        reference_id=None,
        frame="map",
        units="m",
        vertical_alignment="map_coordinates_floor_unverified",
        collision=COLLISION.copy(),
        grid=identity.get("grid", {}),
        visual=dict(
            source_kind="none",
            status="unavailable",
            recipe="none",
            reason="Прогон старого формата содержит только план 2D",
        ),
    )


def read_job_file(root, job_id, relative, limit):
    """Open each directory relative to a pinned parent and reject links, including the leaf."""
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        parts = (job_id, *relative.parts)
        for part in parts[:-1]:
            nxt = os.open(part, os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = nxt
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        with os.fdopen(os.open(parts[-1], flags, dir_fd=fd), "rb") as f:
            info = os.fstat(f.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
                raise SceneError(
                    "Файл карты превышает допустимый размер или имеет неверный тип"
                )
            blob = f.read(limit + 1)
            if len(blob) > limit:
                raise SceneError("Превышен размер файла измерений")
            return blob
    except OSError as exc:
        raise SceneError(
            "Файл измерений отсутствует или не принадлежит работе"
        ) from exc
    finally:
        os.close(fd)


def _cover_indices(xyz, step, budget):
    """Choose measured centres until each input lies within half a bucket diagonal."""
    cells = np.floor(xyz / step).astype(np.int64)
    origin = cells.min(axis=0)
    cells -= origin
    shape = tuple(cells.max(axis=0) + 1)
    if math.prod(map(int, shape)) >= np.iinfo(np.int64).max:
        return None
    keys = np.ravel_multi_index(cells.T, shape)
    order = np.argsort(keys, kind="stable")
    bounds = np.r_[0, np.flatnonzero(np.diff(keys[order])) + 1, len(order)]
    if len(bounds) - 1 > budget:
        return None

    chosen = []
    worst_sq = 0.0
    limit_sq = 3 * step**2 / 4
    for start, end in zip(bounds[:-1], bounds[1:]):
        indices = order[start:end]
        points = xyz[indices]
        middle = (cells[indices[0]] + origin + 0.5) * step
        # Stable order makes equal distances reproducible for the same source PLY.
        representative = int(np.argmin(np.square(points - middle).sum(axis=1)))
        nearest_sq = np.full(len(indices), np.inf)
        while True:
            chosen.append(indices[representative])
            if len(chosen) > budget:
                return None
            distances = np.square(points - points[representative]).sum(axis=1)
            np.minimum(nearest_sq, distances, out=nearest_sq)
            representative = int(np.argmax(nearest_sq))
            farthest_sq = nearest_sq[representative]
            if farthest_sq <= limit_sq:
                worst_sq = max(worst_sq, farthest_sq)
                break
    return chosen, math.sqrt(worst_sq)


def reduce_cloud(records, base):
    """Render a bounded subset of measured source voxels at their original size."""
    xyz = np.column_stack([records[k] for k in ("x", "y", "z")])
    if not len(xyz) or np.abs(xyz).max() > 1e6:
        raise SceneError("Облако пусто или координаты вне допустимого диапазона")
    source_bounds = bounds(xyz)
    step = base
    max_error = 0.0
    if len(records) > MAX_RENDER_CELLS:
        source = xyz.astype(np.float64)
        while True:
            step *= 1.25
            cover = _cover_indices(source, step, MAX_RENDER_CELLS)
            if cover is not None:
                indices, max_error = cover
                xyz = xyz[indices]
                samples = records[indices]
                break
    else:
        samples = records
    out = np.empty(len(samples), CLOUD_DTYPE)
    out["center"] = xyz
    out["half_size"] = base / 2
    # The batch palette quantizes the colour of each selected source sample.
    rgb = np.column_stack([samples[k] for k in ("red", "green", "blue")])
    out["rgb"] = np.minimum((rgb.astype(np.uint16) // 32) * 32 + 16, 255)
    lod = dict(
        source_points=len(records),
        planar_patches=0,
        single_voxels=len(out),
        max_center_error_m=max_error,
        max_center_error_bound_m=math.sqrt(3) * step / 2,
        min_primitive_edge_m=base,
        max_primitive_edge_m=base,
    )
    return out, step, source_bounds, lod


def mesh_batches(cloud):
    # Groups keep similarly coloured nearby boxes together for camera-frustum culling.
    tile = np.floor(cloud["center"][:, :2] / 6.0)
    return np.unique(np.column_stack([tile, cloud["rgb"]]), axis=0, return_inverse=True)


def prepare(data, results, directory):
    scene = data.get("scene")
    if scene is None:
        return legacy_scene(data)
    job = data["job_id"]
    source = data.get("visual_source", {"kind": "published"})
    kind = source.get("kind")
    if (
        scene.get("schema_version") != 1
        or scene.get("job_id") != job
        or scene.get("frame") != "map"
        or scene.get("units") != "m"
        or scene.get("vertical_alignment") != "map_coordinates_floor_unverified"
        or scene.get("registration") != data["registration"]
        or scene.get("transforms") != {"map_to_sim": IDENTITY, "ply_to_map": IDENTITY}
        or scene.get("collision") != COLLISION
    ):
        raise SceneError(
            "Не подтверждены происхождение, единицы или преобразование карты"
        )
    grid = scene.get("grid", {})
    if (
        any(grid.get(k) != v for k, v in data["grid"].items())
        or grid.get("sha256") != data["map_sha256"]
    ):
        raise SceneError("Сетка сцены отличается от карты работы")
    # The grid has a planar metric extent; some clients supply explicit floor Z as well.
    limits = grid_bounds(data["grid"])
    supplied = grid.get("bounds", {})
    if any(supplied.get(k, [])[:2] != v for k, v in limits.items()):
        raise SceneError("Границы сетки сцены отличаются от карты")
    visual = dict(scene.get("visual", {}))
    relative = visual.pop("asset_relpath", None)
    scene = dict(scene, visual=visual)
    if (
        kind not in ("none", "variant", "published")
        or visual.get("source_kind") != kind
    ):
        raise SceneError("Источник визуальной сцены отличается от запроса")
    if visual.get("status") == "unavailable":
        if (
            kind == "variant"
            or source.get("expected_sha256")
            or relative
            or not visual.get("reason")
        ):
            raise SceneError("Запрошенный слой измерений недоступен")
        visual.update(
            recipe="none",
            source_sha256=None,
            source_bounds=None,
            source_voxel_m=None,
            render_voxel_m=None,
            render_geoms=0,
            render_cells=0,
            render_bounds=None,
            render_sha256=None,
        )
        return scene
    if (
        kind == "none"
        or visual.get("status") != "ready"
        or not isinstance(relative, str)
    ):
        raise SceneError("Не подтверждена готовность измеренного слоя")
    if kind == "published":
        if relative != "map3d.ply" or visual.get("variant_id") is not None:
            raise SceneError("Недопустимый путь опубликованной карты")
        meta = json.loads(
            read_job_file(results, job, Path("result.json"), 32 * 1024**2)
        ).get("map3d", {})
        if (
            meta.get("status") != "ready"
            or meta.get("frame") != "map"
            or meta.get("voxel_size_m") != visual.get("source_voxel_m")
        ):
            raise SceneError("Не совпадают метаданные опубликованного облака")
    else:
        token = source.get("variant_id")
        if (
            not isinstance(token, str)
            or not re.fullmatch(r"[a-f0-9]{32}", token)
            or visual.get("variant_id") != token
            or not re.fullmatch(r"\.voxel-variants-v1/[a-f0-9]{64}\.ply", relative)
        ):
            raise SceneError("Недопустимый путь или ID варианта работы")
        request_file = Path(".voxel-variants-v1", "requests", token + ".json")
        handle = json.loads(read_job_file(results, job, request_file, 128 * 1024))
        if (
            handle.get("job_id") != job
            or handle.get("variant_id") != token
            or handle.get("status") != "ready"
            or handle.get("cache_key") != Path(relative).stem
            or handle.get("frame") != "map"
            or handle.get("sha256") != source.get("expected_sha256")
            or handle.get("actual_voxel_size_m") != visual.get("source_voxel_m")
        ):
            raise SceneError("Готовый вариант работы изменился")
    blob = read_job_file(results, job, Path(relative), MAX_SOURCE_BYTES)
    actual = sha256(blob).hexdigest()
    if (
        actual != visual.get("source_sha256")
        or source.get("expected_sha256", actual) != actual
    ):
        raise SceneError("Хэш выбранного облака изменился")
    base = visual.get("source_voxel_m")
    if type(base) not in (int, float) or not 0.001 <= base <= 100:
        raise SceneError("Некорректный шаг исходных измерений")
    try:
        records = read_ply(blob)
    except ValueError as exc:
        raise SceneError(str(exc)) from exc
    if len(records) > 2_400_000:
        raise SceneError("Облако превышает 2 400 000 вокселей")
    reduced, size, source_bounds, lod = reduce_cloud(records, base)
    if kind == "published":
        if "points" in meta and meta["points"] != len(records):
            raise SceneError("Число точек файла отличается от публикации карты")
        if meta.get("bounds") and any(
            not np.allclose(meta["bounds"][k], source_bounds[k], rtol=1e-6, atol=1e-5)
            for k in ("min", "max")
        ):
            raise SceneError("Границы облака отличаются от публикации карты")
    (directory / "measured.ply").write_bytes(blob)
    target = directory / "render.npy"
    np.save(target, reduced, allow_pickle=False)
    with target.open("rb") as f:
        render_sha = file_digest(f, "sha256").hexdigest()
    limits = dict(
        min=(reduced["center"] - reduced["half_size"]).min(axis=0).tolist(),
        max=(reduced["center"] + reduced["half_size"]).max(axis=0).tolist(),
    )
    visual.update(
        source_bounds=source_bounds,
        lod=lod,
        recipe=SIMULATOR_VOXEL_RECIPE,
        render_voxel_m=size,
        render_cells=len(reduced),
        render_geoms=len(mesh_batches(reduced)[0]),
        render_bounds=limits,
        render_sha256=render_sha,
        reason=None,
    )
    return scene

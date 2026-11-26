"""Measured, operator-approved previews of a new 3D map from the same recording."""

import json

import numpy as np
from werkzeug.exceptions import BadRequest, Conflict, NotFound

from .map_filter import FILTER_ALGORITHM, filter_supported_voxels
from .storage import encode, now, uid
from .voxel_cloud import read_ply


def artifact_hashes(directory):
    from .voxel_levels import digest

    return {name: digest(directory / name)
            for name in ("result.json", "map3d.ply", "grid.npz", "trajectory.tum")}


def occupancy_diagnostics(store, job, result):
    """Report measured counts from this grid and the filter that originally produced it."""
    parent = job.get("reference_id") or job.get("derived_from_id")
    report = dict(source=("reference_grid" if job.get("reference_id") else
                          "parent_grid" if parent else "recording_scans"),
                  inherited_from_id=parent, affected_by_proposal=False)
    try:
        with np.load(store.results / job["id"] / "grid.npz", allow_pickle=False) as grid:
            cells = grid["cells"]
            height, width = cells.shape
            report.update(width=width, height=height, resolution_m=float(grid["resolution"]),
                          origin_m=grid["origin"].tolist(),
                          cell_counts=dict(free=int(np.count_nonzero(cells == 0)),
                                           occupied=int(np.count_nonzero(cells == 100)),
                                           unknown=int(np.count_nonzero(cells == -1))))
        seen = {job["id"]}
        owner = job["id"]
        while parent:
            if parent in seen:
                raise ValueError("Зацикленная цепочка опорной карты")
            seen.add(parent)
            ancestor = store.ready(parent)
            owner = ancestor["id"]
            result = json.loads((store.results / owner / "result.json").read_text())
            parent = ancestor.get("reference_id") or ancestor.get("derived_from_id")
        report["grid_job_id"] = owner
        report["occupancy_filter"] = result.get("diagnostics", {}).get("occupancy_filter")
    except (OSError, KeyError, ValueError, Conflict, NotFound):
        report["filter_status"] = "unavailable"
    else:
        report["filter_status"] = "measured" if report["occupancy_filter"] else "not_recorded"
    return report


def diagnostics(store, job):
    directory = store.results / job["id"]
    result = json.loads((directory / "result.json").read_text())
    scene = result.get("map3d") or {}
    report = dict(job_id=job["id"], source_id=job["source_id"], reference_id=job["reference_id"],
                  scope="map3d", occupancy_2d=occupancy_diagnostics(store, job, result))
    summary = scene.get("filter")
    if summary and summary.get("algorithm") == FILTER_ALGORITHM:
        baseline = dict(support_radius_voxels=summary["support_radius_voxels"],
                        retained_voxels=summary["retained_voxels"],
                        rejected_voxels=summary["rejected_voxels"],
                        bounds_m=scene.get("bounds"))
        method = "published_filter_metrics"
    elif scene.get("status") == "ready" and not summary:
        # Old jobs contain all prefilter centroids and their measured observation counts.
        path = directory / "map3d.ply"
        try:
            points = read_ply(path.read_bytes())
            xyz = np.column_stack([points[field] for field in ("x", "y", "z")])
            _, summary = filter_supported_voxels(
                xyz, points["observations"], scene["voxel_size_m"])
        except (OSError, ValueError, KeyError) as err:
            return dict(report, status="unavailable", reason=f"Не удалось оценить исходное облако: {err}")
        baseline = dict(support_radius_voxels=None, retained_voxels=len(points),
                        rejected_voxels=0, bounds_m=scene.get("bounds"))
        method = "legacy_measured_ply"
    else:
        return dict(report, status="unavailable",
                    reason=scene.get("reason") or "Нет сопоставимых измерений 3D-фильтра.")
    try:
        artifacts = artifact_hashes(directory)
    except OSError:
        return dict(report, status="unavailable", reason="Не удалось сверить файлы опубликованной работы.")
    return dict(report, status="available", source=method, algorithm=summary["algorithm"],
                artifacts_sha256=artifacts,
                voxel_size_m=summary["voxel_size_m"], input_voxels=summary["input_voxels"],
                repeated_voxels=summary["repeated_voxels"], one_window_voxels=summary["one_window_voxels"],
                current=baseline, profiles=summary["profiles"],
                interpretation="Фильтр меняет 3D-слой. Повторные наблюдения сохраняются при каждом профиле. "
                "Радиус поддержки дан в шагах вокселя. Меньший радиус может убрать редкую поверхность. "
                "Границы измерены по центрам облака, их соответствие реальному обмеру неизвестно.")


def preview(report, radius):
    if report["status"] != "available":
        raise Conflict(report["reason"])
    selected = next((p for p in report["profiles"] if p["support_radius_voxels"] == radius), None)
    if selected is None:
        raise BadRequest("Для выбранного радиуса отсутствуют измерения")
    current = report["current"]
    change = selected["retained_voxels"] - current["retained_voxels"]
    if change == 0:
        raise Conflict("На этой записи настройка сохраняет тот же набор вокселей.")
    if selected["retained_voxels"] < max(1, report["repeated_voxels"]):
        raise Conflict("Выбранный профиль не сохраняет подтверждённую поверхность.")
    return dict(before=current, after=selected, delta_retained_voxels=change,
                repeated_voxels_preserved=report["repeated_voxels"],
                support_radius_m=radius * report["voxel_size_m"],
                risk=("Более строгий фильтр может убрать редко наблюдавшиеся поверхности."
                      if change < 0 else "Более мягкий фильтр может сохранить одиночные шумовые возвраты."))


def proposal_text(job, args, comparison):
    def bounds(profile):
        box = profile["bounds_m"]
        if box is None:
            return "Нет данных"
        return " · ".join(f"{axis} [{low:.2f}, {high:.2f}]" for axis, low, high in
                          zip("XYZ", box["min"], box["max"]))

    before, after = comparison["before"], comparison["after"]
    return (
        f"{job['source_name']} · {job['id']}\n{args['reason']}\n"
        f"Радиус поддержки {comparison['support_radius_m']:.3g} м "
        f"({after['support_radius_voxels']:g} шага).\n"
        f"Вокселей сейчас {before['retained_voxels']}, прогноз {after['retained_voxels']}. "
        f"Повторно наблюдавшихся сохранится {comparison['repeated_voxels_preserved']}.\n"
        f"{comparison['risk']}\n"
        f"Границы центров, м\nСейчас: {bounds(before)}\nПрогноз: {bounds(after)}\n"
        "Новая работа получит эту 3D-карту и исходную навигационную сетку."
    )


def proposal(store, job, args):
    report = diagnostics(store, job)
    comparison = preview(report, args["support_radius_voxels"])
    if not args["reason"].strip():
        raise BadRequest("Укажите обоснование предлагаемого изменения")
    ident = uid()
    # The baseline binds the comparison to the actual measurements and source settings.
    baseline = dict(comparison=comparison, artifacts_sha256=report["artifacts_sha256"],
                    algorithm=report["algorithm"],
                    source_id=job["source_id"], topics=job["topics"],
                    reference_id=job["reference_id"], voxel_size_m=job["voxel_size_m"])
    with store.connection(True) as cx:
        store.ready(job["id"], cx)
        cx.execute("""
            INSERT INTO proposals (id,job_id,name,kind,args,revision,status,before_body,created_at)
            VALUES (?,?,'propose_map_filter','map.filter',?,0,'pending',?,?)
        """, (ident, job["id"], encode(args), encode(baseline), now()))
    return dict(id=ident, job_id=job["id"], kind="map.filter", name="propose_map_filter",
                arguments=args, status="pending", preview=comparison, draft_id=None,
                preview_text=proposal_text(job, args, comparison))


def compare_and_insert(store, cx, row):
    parent = store.ready(row["job_id"], cx)
    args = json.loads(row["args"])
    baseline = json.loads(row["before_body"])
    source_keys = ("source_id", "reference_id", "topics", "voxel_size_m")
    if any(parent[key] != baseline[key] for key in source_keys):
        raise Conflict("Исходные параметры работы изменились. Запросите новую оценку.")
    # Exact hashes bind the displayed comparison to the immutable projection, trajectory and grid.
    # Re-running a spatial nearest-neighbour fit here would hold the write lock during the fit.
    try:
        hashes = artifact_hashes(store.results / parent["id"])
    except OSError as err:
        raise Conflict("Исходная карта недоступна. Запросите новую оценку.") from err
    if baseline["algorithm"] != FILTER_ALGORITHM or hashes != baseline["artifacts_sha256"]:
        raise Conflict("Измерения исходной карты изменились. Запросите новую оценку.")
    return store.insert_job(cx, parent["source_id"], parent["reference_id"], parent["topics"],
                            parent["voxel_size_m"], {"support_radius_voxels": args["support_radius_voxels"]},
                            derived_from_id=parent["id"])


def rebuild(store, job, source, out, progress):
    """Replay the original lidar against the same registered trajectory and compare the result."""
    import shutil
    from tempfile import TemporaryDirectory
    from pathlib import Path

    from .map3d import build_map3d
    from .recording import recording_snapshot
    from .voxel_levels import digest

    parent = store.ready(job["derived_from_id"])
    published = store.results / parent["id"]
    result = json.loads((published / "result.json").read_text())
    original = result["map3d"]
    edge = original["voxel_size_m"]
    with store.connection() as cx:
        row = cx.execute(
            "SELECT before_body FROM proposals WHERE created_job_id=? AND kind='map.filter'",
            (job["id"],),
        ).fetchone()
    if row is None:
        raise ValueError("Нет подтверждённой оценки изменения 3D-карты.")
    baseline = json.loads(row[0])
    if artifact_hashes(published) != baseline["artifacts_sha256"]:
        raise ValueError("Файлы исходной карты изменились после подтверждения.")
    expected = baseline["comparison"]["after"]
    with recording_snapshot(source, out.parent) as (snapshot, checksum):
        bound = result.get("source_manifest_sha256") or original.get("source_manifest_sha256")
        if bound and bound != checksum:
            raise ValueError("Содержимое записи изменилось со времени публикации исходной работы.")
        if not bound:
            progress(0.05, "Проверка исходной записи по опубликованному облаку")
            with TemporaryDirectory(dir=out.parent) as temp:
                check = build_map3d(snapshot, result, published, Path(temp), voxel_size_m=edge, exact=True)
                if check["status"] != "ready" or digest(Path(temp) / "map3d.ply") != digest(published / "map3d.ply"):
                    raise ValueError("Измерения записи не совпали с опубликованным 3D-слоем.")
        for name in ("grid.npz", "trajectory.tum"):
            shutil.copyfile(published / name, out / name)
        if (published / "frames").is_dir():
            shutil.copytree(published / "frames", out / "frames")
        result["map_filter"] = job["map_filter"]
        result["source_manifest_sha256"] = checksum
        result["derived_from_id"] = parent["id"]
        # A confirmed new job uses the current processing algorithm. Legacy hash verification above
        # still replays the parent recipe, and the support-filter preview keeps its measured bounds.
        result.pop("map3d", None)
        updated = build_map3d(
            snapshot, result, published, out,
            progress=lambda f, stage: progress(0.28 + 0.68 * f, stage),
            voxel_size_m=edge, exact=True,
        )
        actual = updated.get("filter") or {}
        if (updated["status"] != "ready" or updated["points"] != expected["retained_voxels"]
                or updated["bounds"] != expected["bounds_m"]
                or actual.get("rejected_voxels") != expected["rejected_voxels"]):
            raise ValueError("Повторные измерения расходятся с подтверждённой оценкой. Проверьте исходную карту.")
        updated["source_manifest_sha256"] = checksum
        result["map3d"] = updated
        (out / "result.json").write_text(encode(result))

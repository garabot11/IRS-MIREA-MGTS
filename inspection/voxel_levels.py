"""Obtain an exact measured input level, bound to this job's recorded source and frame."""

import json
import tempfile
from hashlib import file_digest, sha256
from pathlib import Path

import numpy as np
from werkzeug.exceptions import BadRequest

from .map3d import build_map3d
from .storage import directory_bag
from .voxel_cloud import read_ply


class LevelError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def key_for(obj):
    return sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def recording_members(path, *, allow_missing=False):
    if path.is_file():
        return [("", path)]
    meta = path / "metadata.yaml"
    names = directory_bag(path, allow_missing=allow_missing)
    parts = [("metadata.yaml", meta)]
    for name in sorted(names):
        member = (path / name).resolve()
        if member.suffix.lower() not in (".mcap", ".db3"):
            raise LevelError(
                "source_changed", "Формат файла исходной записи изменился."
            )
        parts.append((name, member))
    return parts


def recording_sha(members):
    return key_for([(name, digest(path)) for name, path in members])


def parts_sha(parts):
    names = sorted(parts, key=lambda name: (name != "metadata.yaml", name))
    return key_for([(name, parts[name]) for name in names])


def source_files(store, job):
    with store.connection() as db:
        row = db.execute(
            "SELECT path FROM sources WHERE id=?", (job["source_id"],)
        ).fetchone()
    if not row:
        return None, []
    path = Path(row["path"]).resolve()
    root = store.inputs if job["source_id"].startswith("local-") else store.uploads
    if not path.is_relative_to(root):
        raise LevelError("source_changed", "Путь к исходной записи изменился.")
    if not path.exists() or (path.is_dir() and not (path / "metadata.yaml").exists()):
        return None, []
    try:
        members = recording_members(path, allow_missing=True)
    except BadRequest as exc:
        raise LevelError("source_changed", exc.description) from exc
    return (path if all(p.is_file() for _, p in members) else None), members


def signature(paths):
    result = []
    for name, path in paths:
        try:
            stat = path.stat()
        except FileNotFoundError:
            result.append((name, None))
            continue
        result.append(
            (
                name,
                stat.st_dev,
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
            )
        )
    return result


def read_manifest(path):
    try:
        data = json.loads(path.read_bytes())
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, UnicodeError, ValueError):
        return {}


def write_json(path, data):
    with tempfile.NamedTemporaryFile(
        dir=path.parent, delete=False, mode="w", encoding="utf-8"
    ) as f:
        json.dump(data, f, ensure_ascii=False, allow_nan=False)
    Path(f.name).replace(path)


def measured_level(store, job, result, requested, workspace, update):
    directory = store.results / job["id"]
    meta = result.get("map3d", {})
    if meta.get("status") != "ready":
        raise LevelError(
            "unsupported_3d",
            meta.get("reason") or "В работе нет измеренного объёмного слоя.",
        )
    base = meta["voxel_size_m"]
    published = directory / "map3d.ply"
    tum = directory / "trajectory.tum"
    frame = result.get("frame", "map")
    trajectory_hash = digest(tum)
    original_hash = digest(published)
    identity = dict(
        schema_version=1,
        algorithm_version="measured-variants-v1",
        units="m",
        job_id=job["id"],
        source_id=job["source_id"],
        frame=frame,
        trajectory_sha256=trajectory_hash,
        result_sha256=digest(directory / "result.json"),
        published_sha256=original_hash,
        topic=meta["source_topic"],
        grid_sha256=digest(directory / "grid.npz"),
        published_voxel_size_m=base,
    )
    origin = np.loadtxt(tum, max_rows=1, ndmin=2)[0, 1:4].tolist()
    publication = meta.get("source_manifest_sha256")
    provenance = dict(
        identity,
        lattice_origin_m=origin,
        source_state="published",
        source_manifest_sha256=publication,
        source_binding="publication-manifest" if publication else "published-ply",
        level_sha256=original_hash,
        bounds=meta["bounds"],
        integrated_returns=meta.get("integrated_returns"),
        observation_windows=meta.get("observation_windows"),
        scans=meta.get("scans"),
        scans_used=meta.get("scans_used"),
        sampled_scans=meta.get("sampled_scans"),
        skipped_by_time=meta.get("skipped_by_time"),
    )
    if requested >= base:
        source = read_ply(published.read_bytes())
        return (
            source,
            base,
            ("published" if requested == base else "coarsened-from-published"),
            False,
            provenance,
        )
    path, members = source_files(store, job)
    before = signature(members)
    generation = workspace / key_for(identity)
    generation.mkdir(exist_ok=True)
    identity_file = generation / "source.json"
    binding = read_manifest(identity_file)
    previous = publication or binding.get("sha256")
    observed = None
    if members:
        update(0.04, "Проверяем исходную запись")
        observed_parts = {
            name: digest(member) for name, member in members if member.is_file()
        }
        if path:
            observed = parts_sha(observed_parts)
            changed = previous and observed != previous
        else:
            pinned = binding.get("members")
            if not previous or not isinstance(pinned, dict) or parts_sha(pinned) != previous:
                raise LevelError(
                    "source_missing", "Подключите все файлы исходной записи для сверки."
                )
            changed = any(
                pinned.get(name) != value for name, value in observed_parts.items()
            )
        if changed:
            raise LevelError(
                "source_changed",
                "Содержимое подключённой записи изменилось. Подключите исходную запись этой карты.",
            )
        if path and previous == observed and binding.get("members") != observed_parts:
            write_json(identity_file, {"sha256": observed, "members": observed_parts})
    provenance.update(
        source_manifest_sha256=previous or observed,
        source_state="verified" if path else "missing",
        source_binding="publication-manifest" if publication else "legacy-ply-match",
    )
    name = f"level-{round(requested * 100)}cm"
    file = generation / (name + ".ply")
    saved = read_manifest(generation / (name + ".json"))
    if previous and saved.get("source_sha256") == previous and file.is_file():
        payload = file.read_bytes()
        if sha256(payload).hexdigest() == saved.get("sha256"):
            records = read_ply(payload)
            if len(records) == saved.get("points"):
                provenance["level_sha256"] = saved["sha256"]
                return records, requested, "raw-replay", True, provenance
    if not path:
        raise LevelError(
            "source_missing",
            "Для этого шага нужна исходная запись. Подключите её или выберите шаг сохранённой карты.",
        )
    with tempfile.TemporaryDirectory(dir=generation) as tmp:
        out = Path(tmp)
        if not previous:
            update(0.09, "Сверяем измерения с сохранённой картой")
            original = build_map3d(
                path, result, directory, out, voxel_size_m=base, exact=True
            )
            if (
                original.get("status") != "ready"
                or digest(out / "map3d.ply") != original_hash
            ):
                raise LevelError(
                    "source_unverified",
                    "Измерения записи не совпадают с сохранённой картой. Подключите её исходную запись.",
                )
        rebuilt = build_map3d(
            path,
            result,
            directory,
            out,
            progress=lambda f, msg: update(0.20 + 0.49 * f, msg),
            voxel_size_m=requested,
            exact=True,
        )
        if rebuilt.get("voxel_size_m") != requested or rebuilt.get("status") != "ready":
            raise LevelError(
                rebuilt.get("code", "level_unavailable"),
                rebuilt.get("reason") or "Невозможно построить запрошенный шаг.",
            )
        if signature(members) != before or digest(tum) != trajectory_hash:
            raise LevelError(
                "inputs_changed",
                "Исходные данные изменились во время расчёта. Повторите выбор.",
            )
        level = out / "map3d.ply"
        payload = level.read_bytes()
        records = read_ply(payload)
        provenance["level_sha256"] = sha256(payload).hexdigest()
        write_json(identity_file, {"sha256": observed, "members": observed_parts})
        level.replace(file)
        write_json(
            generation / (name + ".json"),
            dict(
                sha256=sha256(payload).hexdigest(),
                source_sha256=observed,
                points=len(records),
            ),
        )
    return records, requested, "raw-replay", False, provenance

"""ROS recording to an occupancy grid, timed robot poses, image detections and TUM trajectory."""

import json
import logging
from pathlib import Path
import shutil

import cv2
import numpy as np
from rosbags.highlevel import AnyReaderError
from rosbags.rosbag2 import ReaderError
from rosbags.serde import SerdeError
from scipy.spatial.transform import Rotation

from .recording import recording_snapshot
from .sensors import read
from .mapping import optimize, occupied_grid, prepare_poses
from .geometry import compose
from .map3d import build_map3d
from .map_filter import normalize_map_filter
from .voxel_cloud import DEFAULT_VOXEL_M
from .visual_registration import visual_pairs
from .localization import register_recording


def _run(
    source,
    out_dir,
    progress,
    reference,
    topics=None,
    voxel_size_m=DEFAULT_VOXEL_M,
    map_filter=None,
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    progress = progress or (lambda fraction, stage: None)
    cv2.setNumThreads(2)
    progress(0.01, "Открытие rosbag")
    try:
        with recording_snapshot(source, out.parent) as (snapshot, checksum):
            data = read(snapshot, out, progress, topics)
            return assemble(
                data,
                out,
                reference,
                progress,
                source=snapshot,
                voxel_size_m=voxel_size_m,
                source_manifest_sha256=checksum,
                map_filter=map_filter,
            )
    except (AnyReaderError, ReaderError, SerdeError) as error:
        source = Path(source)
        kind = (
            "каталог ROS 2 bag"
            if source.is_dir()
            else f"файл {source.suffix[1:].upper()}"
        )
        logging.getLogger(__name__).warning("Ошибка чтения %s: %s", source.name, error)
        if "magic" in str(error).lower():
            reason = "Сигнатура файла не соответствует формату записи."
        else:
            reason = "Данные повреждены, неполны или имеют неподдерживаемый формат."
        raise ValueError(
            f"Не удалось прочитать {kind}. {reason} Проверьте исходную запись и повторите загрузку."
        ) from error


def assemble(
    data,
    out,
    reference,
    progress,
    source=None,
    voxel_size_m=DEFAULT_VOXEL_M,
    source_manifest_sha256=None,
    map_filter=None,
):
    map_filter = normalize_map_filter(map_filter)
    times, poses, zrp, diag = optimize(data, progress)
    progress(
        0.75,
        "Расчёт метрической карты"
        if reference is None
        else "Регистрация с картой другого прохода",
    )
    if reference is None:
        diag["occupancy_filter"] = {}
        grid, origin, res = occupied_grid(
            data["scans"], poses, diagnostics=diag["occupancy_filter"]
        )
        np.savez_compressed(out / "grid.npz", cells=grid, origin=origin, resolution=res)
        registration = dict(
            status="reference", score=1.0, reason="Карта построена по этой записи."
        )
    else:
        with np.load(Path(reference) / "grid.npz", allow_pickle=False) as arrays:
            grid = arrays["cells"]
            origin = arrays["origin"]
            res = float(arrays["resolution"])
        ref_result = json.loads((Path(reference) / "result.json").read_text())
        image_pairs = visual_pairs(reference, ref_result, out, data, poses)
        poses, registration = register_recording(
            grid, origin, res, data["scans"], poses, image_pairs
        )
        shutil.copyfile(Path(reference) / "grid.npz", out / "grid.npz")
        if registration["status"] != "localized":
            data["warnings"].append(registration["reason"])
    _, raw_poses, _ = prepare_poses(data)
    if reference is not None and "transform_xy_yaw" in registration:
        raw_poses = np.array(
            [
                compose(np.asarray(registration["transform_xy_yaw"]), p)
                for p in raw_poses
            ]
        )
    difference = np.linalg.norm(raw_poses[:, :2] - poses[:, :2], axis=1)
    diag["drift"]["max_correction_m"] = round(float(difference.max()), 4)
    for sample, error in zip(diag["drift"]["samples"], difference):
        sample["correction_m"] = round(float(error), 4)
    height, width = grid.shape
    diag.update(
        registration=registration,
        navigation_ready=registration["status"] != "unverified",
        distance_m=round(
            float(np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1).sum()), 3
        ),
        end_gap_m=round(float(np.linalg.norm(poses[0, :2] - poses[-1, :2])), 3),
        base_frame=data["base"],
        warnings=data["warnings"],
        scans=len(times),
    )
    for value, name in [
        (-1, "unknown_cells"),
        (0, "free_cells"),
        (100, "occupied_cells"),
    ]:
        diag[name] = int(np.sum(grid == value))
    # A single shared timeline for JSON and TUM; clip the (small) negative first header offset.
    t = np.round(np.clip(times, 0, data["duration"]), 6)
    keep = np.r_[True, np.diff(t) > 1e-7]
    drift = diag["drift"]
    drift["samples"] = [
        {**sample, "t": float(tm)}
        for sample, tm, ok in zip(drift["samples"], t, keep)
        if ok
    ]
    drift["prior_source"] = (
        "recorded_odometry" if len(data["odom"]) >= 2 else "scan_odometry"
    )
    t = t[keep]
    poses = poses[keep]
    zrp = zrp[keep]
    raw_poses = raw_poses[keep]
    quat = Rotation.from_euler(
        "xyz", np.column_stack([zrp[:, 1:3], poses[:, 2]])
    ).as_quat()
    tum = np.column_stack([data["epoch"] + t, poses[:, :2], zrp[:, 0], quat])
    np.savetxt(
        out / "trajectory.tum",
        tum,
        fmt="%.9f",
        header="TUM RGB-D: timestamp tx ty tz qx qy qz qw",
    )
    path = [
        dict(
            t=round(float(tm), 6),
            x=round(float(x), 5),
            y=round(float(y), 5),
            z=round(float(z), 5),
            yaw=round(float(yaw), 6),
        )
        for tm, (x, y, yaw), z in zip(t, poses, zrp[:, 0])
    ]
    for row, raw in zip(path, raw_poses):
        row.update(raw_x=round(float(raw[0]), 5), raw_y=round(float(raw[1]), 5))
    result = dict(
        schema=1,
        map_filter=map_filter,
        mode="mapping" if reference is None else "localization",
        frame="map",
        origin=origin.tolist(),
        resolution=float(res),
        width=width,
        height=height,
        duration=data["duration"],
        start_time=data["epoch"],
        poses=path,
        camera=data["camera"],
        channels=data["channels"],
        diagnostics=diag,
    )
    if source is not None:
        progress(0.89, "Построение измеренной 3D-карты")
        result["map3d"] = build_map3d(
            source,
            result,
            out,
            voxel_size_m=voxel_size_m,
            exact=True,
            bind_source=source_manifest_sha256 is None,
            progress=lambda fraction, stage: progress(0.89 + 0.10 * fraction, stage),
        )
        if source_manifest_sha256 is not None:
            result["source_manifest_sha256"] = source_manifest_sha256
            result["map3d"]["source_manifest_sha256"] = source_manifest_sha256
    (out / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    )
    progress(1.0, "Расчёт завершён")
    return result


def process_bag(
    source: Path,
    out_dir: Path,
    progress=None,
    topics=None,
    voxel_size_m=DEFAULT_VOXEL_M,
    map_filter=None,
) -> dict:
    return _run(source, out_dir, progress, None, topics, voxel_size_m, map_filter)


def localize_bag(
    source: Path,
    reference_dir: Path,
    out_dir: Path,
    progress=None,
    topics=None,
    voxel_size_m=DEFAULT_VOXEL_M,
    map_filter=None,
) -> dict:
    return _run(
        source, out_dir, progress, reference_dir, topics, voxel_size_m, map_filter
    )

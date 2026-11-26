"""Shared measured-voxel PLY layout and non-destructive spatial filtering."""

import numpy as np

SIMULATOR_VOXEL_RECIPE = "measured_centroid_cover_v2"
DEFAULT_VOXEL_M = 0.10
RESOLUTIONS = (0.05, 0.10, 0.20, 0.40, 0.80)
HEIGHT_COLORS = np.array(
    [[54, 110, 171], [72, 176, 185], [154, 217, 181], [241, 218, 143], [232, 142, 106]]
)
VIEWS_COLORS = np.array([[83, 115, 152], [98, 190, 184], [255, 218, 139]])

FIELDS = (
    ("x", "<f4", "float"),
    ("y", "<f4", "float"),
    ("z", "<f4", "float"),
    ("red", "u1", "uchar"),
    ("green", "u1", "uchar"),
    ("blue", "u1", "uchar"),
    ("observations", "<u2", "ushort"),
)
DTYPE = np.dtype([(k, t) for k, t, _ in FIELDS])
PROPERTIES = [f"property {typename} {name}" for name, _, typename in FIELDS]


def cloud_records(xyz, rgb, observations):
    records = np.empty(len(xyz), DTYPE)
    for key, column in zip(records.dtype.names, (*xyz.T, *rgb.T, observations)):
        records[key] = column
    return records


def ply_header(count, comment):
    return (
        "\n".join(
            [
                "ply",
                "format binary_little_endian 1.0",
                "comment " + comment,
                f"element vertex {count}",
                *PROPERTIES,
                "end_header",
            ]
        )
        + "\n"
    ).encode("ascii")


def read_ply(data):
    edge = data.find(b"end_header\n", 0, 8192)
    if edge < 0:
        raise ValueError("Нет заголовка PLY")
    lines = data[:edge].decode("ascii").splitlines()
    rows = [line for line in lines if not line.startswith("comment ")]
    if (
        len(rows) != 3 + len(FIELDS)
        or rows[:2] != ["ply", "format binary_little_endian 1.0"]
        or rows[3:] != PROPERTIES
    ):
        raise ValueError("Неизвестный формат измеренного PLY")
    kind, element, total = rows[2].split()
    total = int(total)
    start = edge + len(b"end_header\n")
    if (
        (kind, element) != ("element", "vertex")
        or total < 0
        or len(data) - start != total * DTYPE.itemsize
    ):
        raise ValueError("Неполное облако PLY")
    records = np.frombuffer(data, dtype=DTYPE, offset=start, count=total)
    if any(not np.isfinite(records[k]).all() for k in ("x", "y", "z")):
        raise ValueError("Неконечные координаты PLY")
    return records


def voxelize(
    records, base_size, size, min_views=1, heights=None, origin=(0.0, 0.0, 0.0)
):
    """One centroid per occupied cell. Preserve a lower bound on independent time windows."""
    if (
        not np.isfinite([base_size, size]).all()
        or not 0 < base_size <= size
        or min_views < 1
    ):
        raise ValueError("Неверный шаг вокселя или число наблюдений")
    keep = records["observations"] >= min_views
    if heights is not None:
        low, high = heights
        if not np.isfinite([low, high]).all() or low > high:
            raise ValueError("Неверные границы высоты")
        keep &= (records["z"] >= low) & (records["z"] <= high)
    retained = records if keep.all() else records[keep]
    if size == base_size or len(retained) == 0:
        return retained
    xyz = np.column_stack([retained[k] for k in ("x", "y", "z")])
    cells = np.floor((xyz.astype(np.float64) - origin) / size).astype(np.int64)
    order = np.lexsort((cells[:, 2], cells[:, 1], cells[:, 0]))
    sorted_cells = cells[order]
    starts = np.r_[
        0, np.flatnonzero(np.any(sorted_cells[1:] != sorted_cells[:-1], axis=1)) + 1
    ]
    counts = np.diff(np.r_[starts, len(order)])
    reduced = np.empty(len(starts), DTYPE)
    for k in ("x", "y", "z", "red", "green", "blue"):
        avg = np.add.reduceat(retained[k][order], starts, dtype=np.float64) / counts
        reduced[k] = avg if k in ("x", "y", "z") else np.rint(avg)
    reduced["observations"] = np.maximum.reduceat(
        retained["observations"][order], starts
    )
    return reduced

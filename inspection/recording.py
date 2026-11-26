"""Read validated ROS bags and bind new publications to a private recording snapshot."""

import shutil
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from werkzeug.exceptions import BadRequest

from .storage import directory_bag


@contextmanager
def reader_path(source):
    source = Path(source)
    if not source.is_dir():
        yield source
        return
    names = directory_bag(source)
    if all(Path(name).name == name for name in names):
        yield source
        return
    # rosbags 0.11.10's directory reader drops each member's parent path.
    # Give it a private flat index with unique aliases, preserving the bag's
    # combined connection table, split ordering, compression and topic counts.
    with TemporaryDirectory(prefix="kontur-bag-index-") as tmp:
        index = Path(tmp)
        metadata = yaml.safe_load((source / "metadata.yaml").read_bytes())
        info = metadata["rosbag2_bagfile_information"]
        aliases = {}
        for i, name in enumerate(names):
            aliases[name] = f"part-{i:06d}{Path(name).suffix}"
            (index / aliases[name]).symlink_to((source / name).resolve(strict=True))
        info["relative_file_paths"] = list(aliases.values())
        for entry in info.get("files", []):
            if entry["path"] in aliases:
                entry["path"] = aliases[entry["path"]]
        (index / "metadata.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False))
        yield index


@contextmanager
def open_recording(source):
    with reader_path(source) as path:
        with AnyReader(
            [path], default_typestore=get_typestore(Stores.ROS2_JAZZY)
        ) as reader:
            yield reader


def verify_source(source, expected):
    from .voxel_levels import recording_members, recording_sha, signature

    try:
        members = recording_members(Path(source))
        before = signature(members)
        current = recording_sha(members)
        after = signature(recording_members(Path(source)))
    except (OSError, BadRequest, ValueError) as exc:
        raise ValueError(
            "Исходная запись изменилась или стала недоступна до публикации."
        ) from exc
    if current != expected or before != after:
        raise ValueError(
            "Исходная запись изменилась во время обработки. Создайте работу повторно."
        )

    return after


@contextmanager
def recording_snapshot(source, staging):
    from .voxel_levels import recording_members, recording_sha, signature

    source = Path(source)
    members = recording_members(source)
    before = signature(members)
    with TemporaryDirectory(prefix="kontur-source-", dir=staging) as tmp:
        directory = Path(tmp)
        if source.is_file():
            snapshot = directory / source.name
            shutil.copyfile(source, snapshot)
        else:
            snapshot = directory
            for name, member in members:
                destination = snapshot / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(member, destination)
        if signature(recording_members(source)) != before:
            raise ValueError(
                "Исходная запись изменилась при чтении. Создайте работу повторно."
            )
        checksum = recording_sha(recording_members(snapshot))
        yield snapshot, checksum
        verify_source(source, checksum)

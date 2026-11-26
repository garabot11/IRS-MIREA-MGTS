"""Helpers for reading ROS 2 recording metadata."""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import yaml
from werkzeug.exceptions import BadRequest
BAG_TYPES = {".mcap", ".db3"}

def uid():
    return uuid.uuid4().hex

def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))

def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def directory_bag(path, *, allow_missing=False):
    """Return the validated recording member names, in metadata order."""
    root = Path(path).resolve()
    meta = root / "metadata.yaml"
    try:
        if (
            not meta.resolve().is_relative_to(root)
            or not meta.is_file()
            or meta.stat().st_size > 16 * 1024**2
        ):
            raise ValueError
        info = yaml.safe_load(meta.read_text())["rosbag2_bagfile_information"]
        names = info["relative_file_paths"]
        if not isinstance(names, list) or not names:
            raise ValueError
        seen = set()
        for name in names:
            if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
                raise ValueError
            relative = PurePosixPath(name)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or relative.suffix.lower() not in BAG_TYPES
            ):
                raise ValueError
            member = (root / relative).resolve()
            if not member.is_relative_to(root) or member in seen:
                raise ValueError
            if not member.is_file():
                if not allow_missing or member.exists():
                    raise ValueError
                # A file blocking an intermediate directory is invalid layout.
                if any(
                    parent.exists() and not parent.is_dir()
                    for parent in member.parents
                    if parent.is_relative_to(root)
                ):
                    raise ValueError
            seen.add(member)
        return names
    except (
        OSError,
        UnicodeError,
        KeyError,
        ValueError,
        TypeError,
        RuntimeError,
        yaml.YAMLError,
    ) as err:
        raise BadRequest(
            "metadata.yaml должен ссылаться на файлы внутри записи"
        ) from err

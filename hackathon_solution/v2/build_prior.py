"""Reproduce the selected 7 cm measured cloud using the original mapping trajectory."""

import argparse
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from inspection.map3d import build_map3d
from .wall_surface import save_map_layer


def build(source, baseline, output):
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / "map"
        stage.mkdir()
        result = json.loads((baseline / "result.json").read_text())
        info = build_map3d(
            source, result, baseline, stage, voxel_size_m=0.07, exact=True
        )
        if info["status"] != "ready" or info["voxel_size_m"] != 0.07:
            raise RuntimeError(info)
        result["map3d"] = info
        save_map_layer(stage, result)
        (stage / "result.json").write_text(json.dumps(result, ensure_ascii=False))
        for name in ("trajectory.tum", "grid.npz"):
            shutil.copyfile(baseline / name, stage / name)
        stage.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "baseline", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    build(args.source, args.baseline, args.output)

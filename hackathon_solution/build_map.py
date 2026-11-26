"""Build the immutable contest map with the same Kontur pipeline used by the desktop app."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from inspection.pipeline import process_bag
from hackathon_solution.v2.wall_surface import save_map_layer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent, prefix=".mapping-") as tmp:
        staged = Path(tmp) / "map"
        result = process_bag(args.bag, staged, topics={
            "lidar": "/utlidar/cloud_base", "odometry": "/utlidar/robot_odom",
            "imu": "/utlidar/imu",
        }, voxel_size_m=0.10)
        if result["map3d"]["status"] != "ready":
            raise RuntimeError(result["map3d"])
        save_map_layer(staged, result)
        (staged / "result.json").write_text(json.dumps(result, ensure_ascii=False))
        staged.rename(args.output)
    print(f"Saved {result['map3d']['points']} points from {result['map3d']['scans_used']} scans")


if __name__ == "__main__":
    main()

"""Verify the exported G1 policy or run a 15 s crossing on the official grid."""

import argparse
import hashlib
import json
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from hackathon_solution.v2.humanoid_actor import (
    ASSETS,
    HumanoidController,
    WalkingPolicy,
    augment,
)


def check_policy(policy_path):
    import torch

    source_hash = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    expected_hash = json.loads((ASSETS / "provenance.json").read_text())[
        "upstream_sha256"
    ]["deploy/pre_train/g1/motion.pt"]
    if source_hash != expected_hash:
        raise ValueError("Unexpected upstream policy checksum")
    reference = torch.jit.load(str(policy_path), map_location="cpu").eval()
    exported = WalkingPolicy()
    errors = []
    rng = np.random.default_rng(2026)
    with torch.inference_mode():
        for obs in rng.normal(0, 0.7, (200, 47)).astype("f4"):
            expected = reference(torch.from_numpy(obs.copy())[None]).numpy()[0]
            errors.append(float(np.abs(expected - exported(obs)).max()))
    maximum = max(errors)
    if maximum > 5e-5:
        raise ValueError(f"Recurrent inference drift {maximum}")
    return {
        "sequential_observations": len(errors),
        "max_abs_error": maximum,
        "upstream_sha256": source_hash,
    }


def check_physics(map_dir):
    import mujoco

    from hackathon_solution.v2.sim_world import scene_definition
    from inspection.planning import Grid
    from physics.controller import Go2Controller
    from physics.world import WEIGHTS_PATH, build_model

    grid = Grid(map_dir / "grid.npz")
    with tempfile.TemporaryDirectory() as tmp:
        scene = scene_definition(tmp, False)
        xml = ET.parse(scene)
        augment(xml, tmp)
        xml.write(scene)
        model = build_model(scene, grid.cells, grid.origin, grid.resolution, [])
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)
    actor = HumanoidController(model, data)
    actor.place((1.75, 6.5), 0.0)
    dog = Go2Controller(model, data, WEIGHTS_PATH, base_name="base_link")
    dog.place((2.75, 4.10), 1.57)
    mujoco.mj_forward(model, data)
    floor = model.geom("floor").id
    roots = model.body_rootid[model.geom_bodyid]
    feet = np.array(sorted(actor.feet))
    side = {f: model.body(model.geom_bodyid[f]).name for f in feet}
    foot_contacts = dict.fromkeys(
        ("hum_left_ankle_roll_link", "hum_right_ankle_roll_link"), 0
    )
    separate_steps = foot_contacts.copy()
    points, angles, commands, uprights, torques = [], [], [], [], []
    nonfoot_ground_contacts = 0
    for _ in range(7500):
        actor.actuate(data.time)
        dog.actuate()
        mujoco.mj_step(model, data)
        points.append(data.xpos[actor.base].copy())
        uprights.append(data.xmat[actor.base, 8])
        angles.append(data.qpos[actor.joint_indices].copy())
        commands.append(actor.last_command.copy())
        torques.append(data.actuator_force[actor.actuators].copy())
        grounded = set()
        for contact in data.contact:
            a, b = map(int, contact.geom)
            if floor in (a, b):
                foot = b if a == floor else a
                if foot in side:
                    foot_contacts[side[foot]] += 1
                    grounded.add(side[foot])
                elif roots[foot] == actor.base:
                    nonfoot_ground_contacts += 1
        if len(grounded) == 1:
            separate_steps[next(iter(grounded))] += 1
    p = np.array(points)
    summary = {
        "mujoco_version": mujoco.__version__,
        "duration_s": float(data.time),
        "grid_sha256": hashlib.sha256((map_dir / "grid.npz").read_bytes()).hexdigest(),
        "total_actuators": model.nu,
        "controlled_g1_actuators": len(actor.actuators),
        "start_xyz": p[0].tolist(),
        "end_xyz": p[-1].tolist(),
        "min_pelvis_z_m": float(p[:, 2].min()),
        "traveled_x_m": float(p[-1, 0] - p[0, 0]),
        "min_upright_z": float(min(uprights)),
        "mean_speed_last_second_xy_mps": float(np.linalg.norm((p[-1] - p[-501])[:2])),
        "joint_ranges_rad": np.ptp(angles, axis=0).tolist(),
        "max_abs_torque_Nm": np.max(np.abs(torques), axis=0).tolist(),
        "max_forward_command_mps": float(np.max(commands, axis=0)[0]),
        "foot_floor_contacts": foot_contacts,
        "single_support_substeps": separate_steps,
        "nonfoot_floor_contacts": nonfoot_ground_contacts,
    }
    if not (
        summary["traveled_x_m"] >= 1
        and summary["min_upright_z"] > 0.9
        and summary["min_pelvis_z_m"] > 0.68
        and min(separate_steps.values()) > 100
        and nonfoot_ground_contacts == 0
        and summary["mean_speed_last_second_xy_mps"] < 0.08
    ):
        raise RuntimeError(summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--map", type=Path)
    mode.add_argument("--original-policy", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            check_physics(args.map) if args.map else check_policy(args.original_policy),
            indent=2,
        )
    )

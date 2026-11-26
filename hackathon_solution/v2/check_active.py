"""Post-run check of active multi-robot detours using logged motion and commands."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(directory):
    trace = json.loads((directory / "evaluation-poses.json").read_text())
    report = json.loads((directory / "summary.json").read_text())
    with (directory / "telemetry.csv").open() as src:
        rows = list(csv.DictReader(src))
    if len(rows) != len(trace) or len(trace) < 8:
        raise ValueError(f"Incomplete pose/telemetry pair in {directory}")
    time = np.array([float(r["time_s"]) for r in rows])
    if (
        np.any(np.diff(time) <= 0)
        or max(abs(t["t"] - tt) for t, tt in zip(trace, time)) > 0.002
    ):
        raise ValueError("Nonmonotone or unsynchronized clocks")
    xy = np.array([[float(r["truth_x"]), float(r["truth_y"])] for r in rows])
    goal = np.array(report["goal"])
    direction = goal - xy[0]
    initial_distance = np.linalg.norm(direction)
    direction /= initial_distance
    normal = np.array([-direction[1], direction[0]])
    lateral = (xy - xy[0]) @ normal
    goal_distance = np.linalg.norm(goal - xy, axis=1)
    actors = ("dog", "humanoid")
    positions = np.array(
        [
            [item.get("actors", {}).get(a, [np.nan, np.nan]) for a in actors]
            for item in trace
        ]
    )
    finite = np.isfinite(positions).all(axis=2)
    if not finite.any(axis=0).all():
        raise ValueError("One or both opponent robot trajectories missing")
    distances = np.linalg.norm(positions - xy[:, None, :], axis=2)
    earlier = np.maximum(0, np.searchsorted(time, time - 0.5, side="right") - 1)
    duration = time - time[earlier]
    speeds = np.linalg.norm(positions - positions[earlier], axis=2) / np.maximum(
        duration[:, None], 1e-6
    )
    moving = (speeds >= 0.12) & (duration[:, None] >= 0.4) & finite
    simultaneous = moving.sum(axis=1) == len(actors)
    widths = np.diff(time, append=time[-1])
    returns = []
    for a in actors:
        if not all("actor_returns" in entry for entry in trace):
            raise ValueError("Current/nearby sensor-return evidence missing")
        returns.append([int(entry["actor_returns"].get(a, 0)) for entry in trace])
    hits = np.array(returns).T
    recent = np.zeros(hits.shape, dtype=bool)
    for i, t in enumerate(time):
        recent[i] = hits[np.searchsorted(time, t - 0.5) : i + 1].any(axis=0)
    observed_encounter = ((distances < 2.50) & recent & moving).any(axis=1)
    cmd = np.array([[float(r[k]) for k in ("cmd_forward", "cmd_yaw")] for r in rows])
    passive = np.array(
        [[float(r[k]) for k in ("passive_forward", "passive_yaw")] for r in rows]
    )
    deviates = (abs(cmd[:, 0] - passive[:, 0]) >= 0.05) | (
        abs(cmd[:, 1] - passive[:, 1]) >= 0.12
    )
    action = observed_encounter & (cmd[:, 0] >= 0.10) & deviates
    candidates = []
    # Compare the start and end of a short physical maneuver while at least
    # one observed, moving body remains within the evaluation envelope.
    # Finite endpoints and time-synchronized samples keep standing jitter
    # and the final unimpeded walk out of the detour evidence.
    for i in np.flatnonzero(action):
        end = np.searchsorted(time, time[i] + 2.00, side="right")
        for j in range(i + 1, end):
            if not observed_encounter[j]:
                break
            if time[j] - time[i] < 0.35:
                continue
            eligible = action[i : j + 1]
            if eligible.sum() < 3 or eligible.mean() < 0.35:
                continue
            advance = goal_distance[i] - goal_distance[j]
            sideways = abs(lateral[j] - lateral[i])
            motion = np.linalg.norm(xy[j] - xy[i])
            candidates.append(
                {
                    "start_s": round(time[i], 3),
                    "end_s": round(time[j], 3),
                    "lateral_change_m": round(float(sideways), 4),
                    "goal_progress_m": round(float(advance), 4),
                    "physical_displacement_m": round(float(motion), 4),
                    "active_control_scans": int(eligible.sum()),
                    "start_live_v_w": cmd[i].round(4).tolist(),
                    "start_passive_v_w": passive[i].round(4).tolist(),
                    "moving_observed_actors": [
                        a
                        for k, a in enumerate(actors)
                        if moving[i, k] and recent[i, k] and distances[i, k] < 2.50
                    ],
                }
            )

    def evidence_rank(result):
        return min(
            result["lateral_change_m"] / 0.10,
            result["goal_progress_m"] / 0.05,
            result["physical_displacement_m"] / 0.18,
        )

    best = max(candidates, key=evidence_rank, default=None)
    contact = report["first_collision"]
    joint_motion = {}
    for k, a in enumerate(actors):
        path = positions[finite[:, k], k]
        joint_motion[a] = {
            "physical_range_xy_m": np.ptp(path, axis=0).round(4).tolist(),
            "lidar_returns": int(hits[:, k].sum()),
            "moving_s": round(float(widths[moving[:, k]].sum()), 3),
        }
    with np.load(directory / "render-states.npz", allow_pickle=False) as states:
        q = states["qpos"]
        if q.shape != (len(rows), 57) or not np.allclose(states["time"], time):
            raise ValueError(
                "Expected a synchronized 19-position block for each of three robots"
            )
    articulation = {}
    # The team XML appends Go2 and then G1 to the primary Go2. Check each
    # free-body position against independent ground truth before extracting
    # the twelve following joints from the corresponding MuJoCo qpos block.
    for index, (name, trajectory) in enumerate(
        (("primary", xy), ("dog", positions[:, 0]), ("humanoid", positions[:, 1]))
    ):
        start = index * 19
        if (
            np.nanmax(np.linalg.norm(q[:, start : start + 2] - trajectory, axis=1))
            > 0.015
        ):
            raise ValueError(f"Unmatched qpos block for {name}")
        articulation[name] = (
            np.ptp(q[:, start + 7 : start + 19], axis=0).round(5).tolist()
        )
    final_distance = float(report["final_goal_distance_m"])
    last_distance = float(goal_distance[-1])
    # The primary loop may finish between scan instants. Compare truth at both.
    checks = {
        "three_floating_articulated_robots": report["actual_dofs"] == 36
        and all(min(spans) > 0.02 for spans in articulation.values())
        and all(
            joint_motion[a]["lidar_returns"]
            and max(joint_motion[a]["physical_range_xy_m"]) >= 0.50
            for a in actors
        ),
        "two_opponents_move_concurrently": float(widths[simultaneous].sum()) >= 0.50,
        "localized_and_planned_on_current_revision": all(
            int(r["map_revision"]) == int(r["planner_revision"]) for r in rows
        ),
        "goal_reached_on_physics": report["goal_time_s"] is not None
        and final_distance <= 0.30
        and last_distance <= 0.30,
        "no_primary_collision": contact is None
        and all(
            int(v) == 0 for k, v in report["contacts"].items() if k != "foot_floor"
        ),
        "observed_active_detour": bool(best and evidence_rank(best) >= 1),
    }
    return {
        "scenario": report["scenario"],
        "seed": report["seed"],
        "checks": checks,
        "accepted": all(checks.values()),
        "synchronized_scans": len(rows),
        "simultaneous_motion_s": round(float(widths[simultaneous].sum()), 3),
        "observed_moving_encounter_s": round(
            float(widths[observed_encounter].sum()), 3
        ),
        "different_forward_commands_in_encounter": int(action.sum()),
        "strongest_detour_window": best,
        "max_cross_track_offset_m": round(float(abs(lateral).max()), 4),
        "initial_goal_distance_m": round(float(initial_distance), 3),
        "final_goal_distance_m": round(final_distance, 4),
        "first_contact": contact,
        "goal_time_s": report["goal_time_s"],
        "opponents": joint_motion,
        "joint_amplitudes_rad": articulation,
        "opponent_contacts": report.get("other_robots_contacts"),
        "sampled_surface_gaps_m": report["min_sampled_surface_gap_m"],
        "evidence": {
            name: file_sha(directory / name)
            for name in (
                "summary.json",
                "evaluation-poses.json",
                "telemetry.csv",
                "render-states.npz",
            )
        },
    }


def compare_passive(active_dir, passive_dir):
    """Compare incoming state only until the first different physical command."""
    paths = [active_dir, passive_dir]
    reports = [json.loads((path / "summary.json").read_text()) for path in paths]
    controls = []
    poses = [json.loads((path / "evaluation-poses.json").read_text()) for path in paths]
    for path in paths:
        with (path / "telemetry.csv").open() as stream:
            controls.append(list(csv.DictReader(stream)))
    left, right = reports
    fields = ("scenario", "seed", "goal", "map_sha256", "grid_sha256", "sensor_sha256")
    if (
        any(left[k] != right[k] for k in fields)
        or right.get("team_policy") != "passive"
    ):
        raise ValueError("Mismatched scenario/scenario inputs or comparison policy")
    causal = (
        "time_s",
        "scan",
        "map_revision",
        "x",
        "y",
        "z",
        "truth_x",
        "truth_y",
        "truth_z",
        "scan_points",
        "active_cells",
    )
    prefix = 0
    for i, (a, b) in enumerate(zip(*controls)):
        if any(a[k] != b[k] for k in causal) or poses[0][i] != poses[1][i]:
            raise ValueError(
                f"Unmatched initial sensor/physical trajectory at scan {i}"
            )
        prefix += 1
        if any(a[k] != b[k] for k in ("cmd_forward", "cmd_yaw")):
            break
    else:
        raise ValueError("The two runs never issued different commands")
    return {
        "scenario": left["scenario"],
        "matched_initial_scans_through_first_command_difference": prefix,
        "first_difference_time_s": round(float(a["time_s"]), 3),
        "active_v_w": [float(a[k]) for k in ("cmd_forward", "cmd_yaw")],
        "passive_v_w": [float(b[k]) for k in ("cmd_forward", "cmd_yaw")],
        "passive_goal_time_s": right["goal_time_s"],
        "passive_first_contact": right["first_collision"],
        "passive_min_sampled_gap_m": right["min_sampled_surface_gap_m"],
        "passive_final_goal_distance_m": right["final_goal_distance_m"],
        "passive_summary_sha256": file_sha(passive_dir / "summary.json"),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("episodes", type=Path, nargs="+")
    p.add_argument("--output", type=Path)
    p.add_argument(
        "--passive",
        type=Path,
        nargs="+",
        help="Matched passive episodes in the same order",
    )
    args = p.parse_args()
    output = {"episodes": [evaluate(d) for d in args.episodes]}
    if args.passive:
        if len(args.episodes) != len(args.passive):
            p.error("Provide one passive episode for each active episode")
        output["matched_passive_controls"] = [
            compare_passive(a, b) for a, b in zip(args.episodes, args.passive)
        ]
    db = json.dumps(output, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(db)
    else:
        print(db, end="")
    raise SystemExit(0 if all(r["accepted"] for r in output["episodes"]) else 1)

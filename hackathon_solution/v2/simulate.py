"""Closed-loop MuJoCo trial. Run from the repository root with the physics image."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation

from inspection.planning import Grid
from inspection.voxel_cloud import cloud_records, ply_header, read_ply
from hackathon_solution.tracker import FixedMap, apply
from .sim_navigation import MotionEstimate, OnlineMap, Planner
from .sim_world import Simulator, START, START_YAW, GOAL, TEAM_SCENES


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ply(path, records, comment):
    path.write_bytes(ply_header(len(records), comment) + records.tobytes())


def csv_write(path, rows):
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=list(dict.fromkeys(k for row in rows for k in row))
        )
        writer.writeheader()
        writer.writerows(rows)


def quantiles(values):
    return (
        {
            k: round(float(np.percentile(values, q)), 4)
            for k, q in (("min", 0), ("p50", 50), ("p95", 95), ("max", 100))
        }
        if values
        else {}
    )


def actor_truth(sim):
    result = {}
    for name in ("barrier", "pedestrian"):
        xyz = sim.data.mocap_pos[sim.mocap[name]]
        if xyz[2] > -1:
            result[name] = xyz[:2].copy()
    if sim.other:
        result["dog"] = sim.data.xpos[sim.other.base, :2].copy()
    if sim.humanoid:
        result["humanoid"] = sim.data.xpos[sim.humanoid.base, :2].copy()
    return result


def run(
    map_dir,
    sensor_dir,
    out,
    scenario,
    frozen,
    pace,
    max_seconds,
    seed,
    permanent_motion_model=None,
    team_policy="active",
    on_navigation=None,
    native_viewer=False,
    wall_mesh=None,
    keep_viewer=False,
    scene_stream=None,
):
    out.mkdir(parents=True, exist_ok=False)
    grid = Grid(map_dir / "grid.npz")
    fixed = FixedMap(map_dir / "map3d.ply")
    records = read_ply((map_dir / "map3d.ply").read_bytes())
    edge = json.loads((map_dir / "result.json").read_text())["map3d"]["voxel_size_m"]
    if native_viewer and wall_mesh is None:
        default_wall = map_dir / "wall-surface.stl"
        if default_wall.is_file():
            wall_mesh = default_wall
    options = {"wall_mesh": wall_mesh} if wall_mesh is not None else {}
    sim = Simulator(grid, fixed.xyz, edge, sensor_dir, scenario, seed, **options)
    estimate = MotionEstimate(START, START_YAW)
    mapper = OnlineMap(grid, fixed, sim.origin)
    detector = None
    if permanent_motion_model:
        from .motion.detector import MotionDetector

        detector = MotionDetector(fixed.tree, permanent_motion_model)
    motion_ms = []
    motion_high = 0
    planner = Planner(grid, GOAL)
    counterfactual = Planner(grid, GOAL)
    active_planner = None
    if scenario in TEAM_SCENES and team_policy == "active" and not frozen:
        from .active_planner import ActivePlanner

        active_planner = ActivePlanner(grid, GOAL)
    rows = []
    route_log = []
    latencies = []
    late_scan = 0
    changed_commands = 0
    scans = 0
    next_scan = 0.085
    command = (0.0, 0.0, 0.0)
    nav = "starting"
    goal_time = None
    events = {}
    previous_event = None
    last_observed = -10.0
    valid_observations = 0
    min_actor = {}
    min_surface = {}
    sensor_reports = []
    ground = []
    map_history = []
    tick_lag = []
    sampled_swept_cells = {}
    last_seen_source = {}
    yield_until = -1.0
    begin = time.perf_counter()
    render_qpos = []
    render_mocap = []
    render_time = []
    view = None
    stream = None
    if scene_stream is not None:
        from .scene_stream import SceneStream

        stream = SceneStream(scene_stream, sim, GOAL)
    if native_viewer:
        from .live_view import NativeScene

        view = NativeScene(sim, GOAL)
        begin = time.perf_counter()
    while sim.data.time < max_seconds:
        sim.step(command)
        estimate.predict(sim.measurements)
        sim.measurements.clear()
        t = float(sim.data.time)
        if t >= next_scan:
            stage = time.perf_counter()
            pts, sensor = sim.scan(scans)
            after_sensor = time.perf_counter()
            # Localize on the immutable static submap; moving returns are excluded
            # from this index while the live layer is available to navigation.
            valid = estimate.observe(fixed, pts)
            after_fit = time.perf_counter()
            motion_start = time.perf_counter()
            moving = (
                detector.score(pts, estimate.pose) >= detector.threshold
                if detector
                else None
            )
            if moving is not None:
                motion_ms.append(1000 * (time.perf_counter() - motion_start))
                motion_high += int(moving.sum())
            prev_revision = mapper.revision
            if valid:
                valid_observations += 1
                last_observed = t
                mapper.integrate(pts, estimate.pose, t, scans, moving)
                if active_planner is not None and hasattr(active_planner, "observe_scan"):
                    active_planner.observe_scan(scans, t, apply(estimate.pose, pts),
                                                apply(estimate.pose, sim.origin))
            after_map = time.perf_counter()
            map_command, live_state = planner.command(
                estimate.pose, mapper.active, mapper.revision
            )
            old_command, frozen_state = counterfactual.command(estimate.pose, set(), 0)
            if mapper.crossing_risk(estimate.pose, t):
                yield_until = t + 0.42
            if t < yield_until:
                map_command, live_state = (0.0, 0.0, 0.0), "yield_to_crossing"
            passive_command, passive_state = map_command, live_state
            selected_plan = counterfactual if frozen else planner
            if active_planner is not None:
                selected_plan = active_planner
                if hasattr(active_planner, "set_reference_command"):
                    active_planner.set_reference_command(
                        passive_command, passive_state, planner.route, mapper.revision, t
                    )
                map_command, live_state = active_planner.command(
                    estimate.pose, mapper.active, mapper.revision, mapper.tracks, t
                )
            command, nav = (
                (old_command, frozen_state) if frozen else (map_command, live_state)
            )
            changed = np.linalg.norm(np.array(map_command) - old_command) > 0.05
            changed_commands += int(changed)
            if t - last_observed > 0.45:
                command = (0.0, 0.0, 0.0)
                nav = "poor_geometry"
            if goal_time is None and nav == "goal":
                goal_time = t
            if on_navigation is not None:
                on_navigation(selected_plan, nav, goal_time)
            end = time.perf_counter()
            latency = (end - stage) * 1000
            latencies.append(latency)
            period = float(
                sim.profiles["period_s"][scans % len(sim.profiles["period_s"])]
            )
            late_scan += int(latency > period * 1000)
            sensor_reports.append(sensor)
            next_scan += period
            # Ground truth is evaluated after the commands and is not passed to navigation.
            truth, rot = sim.truth()
            err = np.linalg.norm(estimate.pose[:3, 3] - truth)
            angle = np.rad2deg(
                Rotation.from_matrix(estimate.pose[:3, :3] @ rot.T).magnitude()
            )
            actors = actor_truth(sim)
            for label, name in (
                (1, "pedestrian"),
                (2, "dog"),
                (3, "barrier"),
                (5, "humanoid"),
            ):
                if np.any(sim.evaluation_labels == label):
                    events.setdefault(name + "_first_return_s", t)
            # Observational attribution is exclusively a post-decision audit. A
            # static return can legitimately re-occupy a formerly swept cell.
            audit_points = apply(estimate.pose, pts)
            audit_mask = (audit_points[:, 2] > 0.12) & (audit_points[:, 2] < 1.85)
            for key, label in zip(
                np.floor(audit_points[audit_mask] / 0.12).astype(int),
                sim.evaluation_labels[audit_mask],
            ):
                last_seen_source[tuple(key)] = int(label)
            for key, point in actors.items():
                min_actor[key] = min(
                    min_actor.get(key, 100.0), float(np.linalg.norm(truth[:2] - point))
                )
                min_surface[key] = min(
                    min_surface.get(key, 2.0), sim.clearance_truth(key)
                )
                if key in ("dog", "pedestrian", "humanoid"):
                    for cell in mapper.active:
                        if math.dist(grid.center((cell[1], cell[0])), point) < 0.48:
                            sampled_swept_cells.setdefault(key, set()).add(cell)
                if any(
                    np.linalg.norm(np.array(grid.center((r, c))) - point) < 0.47
                    for c, r in mapper.active
                ):
                    events.setdefault(key + "_first_observation_s", t)
                    if changed:
                        events.setdefault(key + "_first_command_change_s", t)
            if changed:
                events.setdefault("first_changed_command_s", t)
                if abs(command[0]) + abs(command[2]) > 0.01:
                    events.setdefault("first_nonzero_changed_command_s", t)
            pose = estimate.pose[:3, 3]
            row = dict(
                time_s=t,
                scan=scans,
                old_revision=prev_revision,
                map_revision=mapper.revision,
                planner_revision=selected_plan.read_revision,
                localization="observed" if valid else "predicted",
                nav=nav,
                active_cells=len(mapper.active),
                supported_voxels=sum(
                    v["count"] >= 2 for v in mapper.observations.values()
                ),
                scan_points=len(pts),
                emitted=sensor["emitted"],
                rings=sensor["ring_count"],
                firing_span_ms=sensor["firing_span_ms"],
                x=pose[0],
                y=pose[1],
                z=pose[2],
                truth_x=truth[0],
                truth_y=truth[1],
                truth_z=truth[2],
                error_m=err,
                rotation_error_deg=angle,
                cmd_forward=command[0],
                cmd_yaw=command[2],
                frozen_forward=old_command[0],
                frozen_yaw=old_command[2],
                different_from_fixed_plan=int(changed),
                observed_tracks=len(mapper.tracks),
                yield_until=yield_until,
                sensor_ms=(after_sensor - stage) * 1000,
                fit_ms=(after_fit - after_sensor) * 1000,
                map_ms=(after_map - after_fit) * 1000,
                plan_ms=(end - after_map) * 1000,
                loop_ms=latency,
            )
            if sim.actor:
                actor_command = (
                    sim.other.command if sim.other else sim.humanoid.last_command
                )
                row.update(
                    actor_cmd_forward=actor_command[0],
                    actor_cmd_lateral=actor_command[1],
                    actor_cmd_yaw=actor_command[2],
                )
            if scenario in TEAM_SCENES:
                row.update(passive_forward=passive_command[0], passive_yaw=passive_command[2],
                           passive_nav=passive_state)
                if active_planner is not None:
                    row["active_decision"] = json.dumps(active_planner.decision)
            rows.append(row)
            ground.append(
                {
                    "t": round(t, 3),
                    "xy": truth[:2].round(3).tolist(),
                    "yaw": round(math.atan2(rot[1, 0], rot[0, 0]), 4),
                    "actors": {k: v.round(3).tolist() for k, v in actors.items()},
                }
            )
            if scenario in TEAM_SCENES:
                ground[-1]["actor_returns"] = {
                    name: int(np.count_nonzero(sim.evaluation_labels == label))
                    for name, label in (("dog", 2), ("humanoid", 5))
                }
            if scans % 4 == 0:
                path = selected_plan.route
                map_history.append(
                    dict(
                        t=round(t, 3),
                        cells=sorted([list(map(int, p)) for p in mapper.active]),
                        path=path,
                    )
                )
            render_qpos.append(sim.data.qpos.copy())
            render_mocap.append(sim.data.mocap_pos.copy())
            render_time.append(t)
            if nav != previous_event:
                route_log.append([round(t, 3), nav, mapper.revision])
                previous_event = nav
            if stream is not None:
                stream.publish(estimate.pose, pts, mapper, selected_plan, nav, scans)
            if view is not None:
                view.sync(selected_plan.route, estimate.pose[:2, 3], nav)
            scans += 1
        tick_lag.append(max(0.0, (time.perf_counter() - begin - t) * 1000))
        if sim.first_contact or (goal_time is not None and t - goal_time > 1.6):
            break
        if pace:
            delay = begin + float(sim.data.time) - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
    elapsed = time.perf_counter() - begin
    additions = mapper.records()
    xyz = np.array([s["p"] for s in additions]).reshape(-1, 3)
    delta = cloud_records(
        xyz,
        np.tile([255, 135, 55], (len(xyz), 1)),
        np.array([s["count"] for s in additions], dtype=np.uint16),
    )
    ply(out / "delta.ply", delta, "simulated returns with repeated time support")
    ply(
        out / "augmented.ply",
        np.concatenate([records, delta]),
        "measured input followed by simulated observations",
    )
    occ = grid.cells.copy()
    for c, r in mapper.active:
        occ[r, c] = 100
    np.savez_compressed(
        out / "occupancy.npz", cells=occ, origin=grid.origin, resolution=grid.resolution
    )
    np.savez_compressed(
        out / "render-states.npz",
        qpos=render_qpos,
        mocap=render_mocap,
        time=render_time,
    )
    csv_write(out / "telemetry.csv", rows)
    if additions:
        csv_write(
            out / "delta-provenance.csv",
            [
                dict(
                    x=s["p"][0],
                    y=s["p"][1],
                    z=s["p"][2],
                    first_s=s["first"],
                    last_s=s["last"],
                    last_scan=s["scan"],
                    observations=s["count"],
                )
                for s in additions
            ],
        )
    (out / "evaluation-poses.json").write_text(
        json.dumps(ground, separators=(",", ":")) + "\n"
    )
    (out / "map-history.json").write_text(
        json.dumps(map_history, separators=(",", ":")) + "\n"
    )
    (out / "route-events.json").write_text(json.dumps(route_log) + "\n")
    n = len(rows)
    dynamic_ghosts = {
        name: sum(
            last_seen_source.get(k) == label
            and s["persistent"]
            and not s["motion_blocked"]
            for k, s in mapper.observations.items()
        )
        for label, name in ((1, "pedestrian"), (2, "dog"), (5, "humanoid"))
    }
    report = dict(
        scenario=scenario,
        mode="frozen" if frozen else "online",
        paced=pace,
        seed=seed,
        engine=mj.__version__,
        ray_backend=sim.surface.backend,
        map_sha256=sha(map_dir / "map3d.ply"),
        grid_sha256=sha(map_dir / "grid.npz"),
        source_points=len(records),
        source_voxel_m=edge,
        output_augmented_vertices=len(records) + len(delta),
        sensor_sha256=sha(sensor_dir / "rays.npz"),
        physics_steps=sim.primary.steps,
        actual_dofs=sim.model.nu,
        joint_range_rad=(sim.joint_max - sim.joint_min).round(4).tolist(),
        secondary_physics_steps=sim.other.steps
        if sim.other
        else sim.humanoid.counter
        if sim.humanoid
        else 0,
        secondary_joint_range_rad=(
            (sim.actor_joint_range[1] - sim.actor_joint_range[0]).round(4).tolist()
            if sim.actor
            else []
        ),
        secondary_contacts=sim.actor_contacts,
        motion_gate_sha256=sha(permanent_motion_model)
        if permanent_motion_model
        else None,
        motion_gate_threshold=detector.threshold if detector else None,
        motion_gate_ms=quantiles(motion_ms),
        motion_flagged_returns=motion_high,
        persistent_proposals=sum(s["persistent"] for s in mapper.observations.values()),
        wall_s=elapsed,
        sim_s=sim.data.time,
        real_time_factor=sim.data.time / elapsed,
        goal=GOAL.tolist(),
        goal_time_s=goal_time,
        final_goal_distance_m=float(np.linalg.norm(sim.truth()[0][:2] - GOAL)),
        initial_goal_distance_m=float(np.linalg.norm(START - GOAL)),
        contacts=sim.contacts,
        first_collision=sim.first_contact,
        min_center_distance_to_actor_m=min_actor,
        min_sampled_surface_gap_m=min_surface,
        events=events,
        swept_cells={
            name: dict(
                ever=len(cells),
                still_active=len(cells & mapper.active),
                cleared=len(cells - mapper.active),
            )
            for name, cells in sampled_swept_cells.items()
        },
        persistent_last_hit_from_dynamic_actor=dynamic_ghosts,
        schedule_lag_ms=quantiles(tick_lag),
        samples_over_20ms_schedule_lag=sum(lag > 20 for lag in tick_lag),
        observed_scans=valid_observations,
        scans=n,
        odom_samples=sim.odom_count,
        gyro_samples=sim.imu_count,
        odom_hz=sim.odom_count / sim.data.time,
        gyro_hz=sim.imu_count / sim.data.time,
        lidar_hz=n / sim.data.time,
        range_cutoff_m=36.6,
        scan_time_groups=8,
        return_count=quantiles([s["returned"] for s in sensor_reports]),
        emitted_count=quantiles([s["emitted"] for s in sensor_reports]),
        deadlines_missed=late_scan,
        loop_ms=quantiles(latencies),
        pose_error_m=quantiles([r["error_m"] for r in rows]),
        rotation_error_deg=quantiles([r["rotation_error_deg"] for r in rows]),
        cumulative_path_m=float(
            np.linalg.norm(
                np.diff(np.array([r["xy"] for r in ground]), axis=0), axis=1
            ).sum()
        ),
        final_revision=mapper.revision,
        added_persistent_points=len(delta),
        ever_occupied_cells=len(mapper.ever_occupied),
        last_active_cells=len(mapper.active),
        cleared_voxels=len(mapper.cleared),
        visible_free_cells=len(mapper.seen_free),
        changed_commands=changed_commands,
        nav_states={
            s: sum(r["nav"] == s for r in rows) for s in set(r["nav"] for r in rows)
        },
    )
    if scenario in TEAM_SCENES:
        report["team_policy"] = "frozen" if frozen else team_policy
        report["other_robots_contacts"] = sim.team_contacts
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "scenario",
                    "mode",
                    "scans",
                    "real_time_factor",
                    "goal_time_s",
                    "contacts",
                    "final_goal_distance_m",
                    "pose_error_m",
                    "events",
                    "added_persistent_points",
                    "changed_commands",
                    "loop_ms",
                ]
            },
            indent=2,
        ),
        flush=True,
    )
    if view is not None:
        if keep_viewer:
            view.wait_for_close()
        view.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("map", "sensor", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument(
        "--scenario",
        choices=("normal", "unexpected", "pedestrian", "dog", "mixed", "humanoid", *TEAM_SCENES),
        required=True,
    )
    parser.add_argument("--frozen", action="store_true")
    parser.add_argument("--team-policy", choices=("active", "passive"), default="active")
    parser.add_argument("--viewer", action="store_true", help="Open native interactive MuJoCo")
    parser.add_argument("--keep-viewer", action="store_true", help="Keep final pose visible until the MuJoCo window is closed")
    parser.add_argument("--wall-mesh", type=Path, help="Measured visualization surface in map coordinates")
    parser.add_argument("--scene-stream", type=Path, help="Publish the current physics and navigation state to a shared directory for RViz")
    parser.add_argument("--unpaced", action="store_true")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--permanent-motion-model",
        type=Path,
        help="Optional moving-return gate for exported long-term voxels only",
    )
    a = parser.parse_args()
    run(
        a.map,
        a.sensor,
        a.output,
        a.scenario,
        a.frozen,
        not a.unpaced,
        a.duration,
        a.seed,
        a.permanent_motion_model,
        a.team_policy,
        native_viewer=a.viewer or a.keep_viewer,
        wall_mesh=a.wall_mesh,
        keep_viewer=a.keep_viewer,
        scene_stream=a.scene_stream,
    )

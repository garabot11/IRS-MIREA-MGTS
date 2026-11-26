"""Receding-horizon unicycle planner using the observed map and measured obstacle tracks."""
import math

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from inspection.planning import _route
from .sim_navigation import OnlineMap, Planner


class ActivePlanner:
    radius = 0.49
    step_s = 0.14
    times = np.arange(1, 21) * step_s

    def __init__(self, grid, goal):
        self.grid = grid
        self.goal = np.asarray(goal, dtype=float)
        self.static = self._clearance(grid.cells == 0)
        self.live = self.static
        self.reference = []
        self.route = []
        self.read_revision = None
        self.decision = {}
        self.side = 0
        self.hold_until = 0.
        self.last_time = None
        self.v = self.w = 0.

        self.samples = {}
        self.memory_cells = set()
        self.previous_memory = set()
        self.hits = np.empty((0, 3))
        self.scan_id = None

        self.reference_command = None
        self.baseline = Planner(grid, goal)
        self.crossing_until = 0.
        self.encounter_start = None
        self.encounter_time = 0.
        self.wait_clear = False
        self.track_history = {}

    def set_reference_command(self, passive_command, passive_nav, route, revision, t):
        self.reference_command = (float(t), int(revision), tuple(passive_command), passive_nav, list(route))

    def _body_margin(self, xy, yaw, command):
        """Check the oriented body and its corners while reacting and braking between lidar scans."""
        # Support extremes measured across the controller's articulated standing/walking poses.
        half = np.array([(0.340+0.409)/2, (0.249+0.261)/2])
        offset = np.array([(0.340-0.409)/2, (0.249-0.261)/2])
        v, _, w = command
        pts = self.hits[:, :2]
        if len(self.memory_cells):
            p = [self.grid.center((r, c)) for c, r in self.memory_cells]
            pts = np.vstack((pts, p))
        pts = pts[np.linalg.norm(pts-xy, axis=1) < 1.4]
        if not len(pts):
            return True, (None, None, None)
        steps = np.linspace(0., 0.14 + max(abs(v)/0.9, abs(w)/2.3), 9)
        pos = xy.copy()
        heading = yaw
        values = []
        for i, tm in enumerate(steps):
            if i:
                dt = tm-steps[i-1]
                braking = max(0., (tm+steps[i-1])/2-0.14)
                vel = max(0., v-0.9*braking)
                omega = np.sign(w)*max(0., abs(w)-2.3*braking)
                mid = heading + dt*omega/2
                pos += dt*vel*np.array([np.cos(mid), np.sin(mid)])
                heading += dt*omega
            delta = pts-pos
            fx = delta[:,0]*np.cos(heading)+delta[:,1]*np.sin(heading)
            fy = -delta[:,0]*np.sin(heading)+delta[:,1]*np.cos(heading)
            q = np.abs(np.column_stack((fx, fy))-offset) - half
            sdf = np.linalg.norm(np.maximum(q,0.),axis=1) + np.minimum(np.max(q,axis=1),0.)
            values.append(float(sdf.min()))
        now, worst, end = values[0], min(values), values[-1]
        # A prior pose may already enter the soft lateral margin. Permit departure only
        # when the hard envelope remains free and the candidate does not consume that margin.
        safe = worst >= 0.055 or (now > 0.018 and worst >= now-0.002 and end >= now-0.001)
        return safe, (now,worst,end)

    def observe_scan(self, scan_id, t, world_hits, origin):
        """An absent return is unknown. Clear remembered surfaces only along observed free beams."""
        if self.scan_id == scan_id:
            return
        self.scan_id = scan_id
        pts = np.asarray(world_hits).reshape(-1, 3)
        points = pts[np.isfinite(pts).all(axis=1)]
        rays = points - origin
        lengths = np.linalg.norm(rays, axis=1)
        body = (points[:, 2] > 0.12) & (points[:, 2] < 0.72) & (lengths < 5.)
        self.hits = points[body]
        # Keep a separate height band in each horizontal cell so a ray above a box cannot clear it.
        novel = self.hits[self._sample(self.static, self.hits[:, :2]) > 0.38]
        cell = np.array([0.14, 0.14, 0.14])
        keys, first = np.unique(np.floor(novel/cell).astype(int), axis=0, return_index=True)
        hit_keys = set(map(tuple, keys))
        for key, xyz in zip(map(tuple, keys), novel[first]):
            if key not in self.samples:
                self.samples[key] = dict(xyz=xyz.copy(), first=t, last=t, count=1, misses=0, fixed=False)
                continue
            a = self.samples[key]
            a['last'], a['misses'] = t, 0
            a['count'] += 1
            a['xyz'] += (xyz-a['xyz'])/a['count']
        # Probe the measured interval with a step shorter than one voxel. Stop before the endpoint.
        free_keys = set()
        if self.samples and len(points):
            ds = np.arange(0.36, 4.6, 0.085)
            pos = origin + rays[:, None] / np.maximum(lengths[:, None, None], 1e-8) * ds[None, :, None]
            valid = ((ds[None, :] < lengths[:, None]-0.22) &
                     (pos[:, :, 2] > 0.12) & (pos[:, :, 2] < 0.72))
            free_keys = set(map(tuple, np.unique(np.floor(pos[valid]/cell).astype(int), axis=0)))
        for key, a in list(self.samples.items()):
            if key not in hit_keys and key in free_keys and t-a['last'] > 0.15:
                a['misses'] += 1
            if a['misses'] >= 3 or (not a['fixed'] and t-a['last'] > 0.6):
                del self.samples[key]

    def _remember(self, movers, occupancy, t):
        cells = set()
        for key, sample in list(self.samples.items()):
            p = sample['xyz'][:2]
            # A moving blob invalidates map persistence at its recent measured location.
            if len(movers) and np.any(np.sum((movers-p)**2, axis=1) < 0.48**2):
                del self.samples[key]
            else:
                row, col = self.grid.index(p)
                if (not sample['fixed'] and t-sample['last'] < 0.01 and
                        any((col+dc, row+dr) in occupancy for dc, dr in ((0,0),(1,0),(-1,0),(0,1),(0,-1)))):
                    sample['novel'] = sample.get('novel', 0) + 1
                    sample.setdefault('novel_since', t)
                    sample['fixed'] = sample['novel'] >= 4 and t-sample['novel_since'] >= 0.80
                if sample['fixed']:
                    cells.add((col, row))
        self.memory_cells = cells

    def _clearance(self, free):
        # Treat the map boundary and unobserved cells as obstacles.
        bordered = np.pad(free, 1, constant_values=False)
        return distance_transform_edt(bordered)[1:-1, 1:-1] * self.grid.resolution

    def _sample(self, distances, points):
        indices = np.floor((points - self.grid.origin) / self.grid.resolution).astype(int)
        x, y = indices[..., 0], indices[..., 1]
        inside = (x >= 0) & (x < self.grid.width) & (y >= 0) & (y < self.grid.height)
        return np.where(inside, distances[np.clip(y, 0, self.grid.height - 1),
                                          np.clip(x, 0, self.grid.width - 1)], 0.)

    def _reference(self, xy):
        if not self.reference:
            self.reference = _route(self.grid, self.static > self.radius,
                                    [xy.tolist(), self.goal.tolist()]) or []
        if len(self.reference) < 2:
            return None
        p = np.array(self.reference)
        segments = np.diff(p, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        directions = segments / np.maximum(lengths[:, None], 1e-9)
        along = np.clip(np.sum((xy - p[:-1]) * directions, axis=1), 0, lengths)
        feet = p[:-1] + along[:, None] * directions
        nearest = np.argmin(np.linalg.norm(feet - xy, axis=1))
        station = np.r_[0., np.cumsum(lengths)]
        look = min(station[-1], station[nearest] + along[nearest] + 1.05)
        target = np.array([np.interp(look, station, p[:, axis]) for axis in range(2)])
        return target, directions[nearest], feet[nearest]

    def _tracks(self, tracks, t, xy):
        centers, velocities, radii, mature = [], [], [], []
        # Object identity here links records already associated by OnlineMap from returns.
        # The trace contains no world-body IDs. It merely extends the short velocity-fit window.
        self.track_history = {k:h for k,h in self.track_history.items() if t-h[-1][0] < 0.8}
        for tr in tracks:
            age = t - tr['last']
            center = np.asarray(tr['center'])
            if age < 0 or age > 0.7 or np.linalg.norm(center - xy) > 4.5:
                continue
            velocity = np.asarray(tr['velocity'])
            speed = np.linalg.norm(velocity)
            velocity = velocity * min(1., 1.6 / max(speed, 1e-8))
            centers.append(center + age * velocity)
            velocities.append(velocity)
            # Blob centres lie on visible surfaces and have finite spatial extent.
            radii.append(0.82 if tr['height'] > 0.8 or speed > 0.15 else 0.68)
            history = self.track_history.setdefault(id(tr), [])
            if not history or tr['last'] > history[-1][0]:
                history.append((tr['last'], center.copy()))
            while len(history)>2 and t-history[0][0] > 2.0:
                history.pop(0)
            moving = False
            if len(history) >= 9 and history[-1][0]-history[0][0] >= 0.85:
                stamps = np.array([tt for tt,_ in history])
                points = np.array([pp for _,pp in history])
                trend = np.mean(points[-3:],axis=0)-np.mean(points[:3],axis=0)
                stamps -= stamps.mean()
                drift = stamps @ (points - points.mean(axis=0)) / max(1e-8, stamps @ stamps)
                scatter = np.linalg.norm(points-points.mean(axis=0)-stamps[:,None]*drift,axis=1).mean()
                moving = np.linalg.norm(trend)>0.31 and np.linalg.norm(drift)>0.16 and scatter < 0.13
                if moving:
                    velocities[-1] = drift * min(1., 1.6/max(np.linalg.norm(drift), 1e-8))
            mature.append(moving)
        return np.reshape(centers, (-1, 2)), np.reshape(velocities, (-1, 2)), np.array(radii), np.array(mature, dtype=bool)

    def _guide(self, xy, target, forward, foot, centers, velocities, mature, t):
        lateral = np.array([-forward[1], forward[0]])
        horizon = np.arange(0.2, 4.1, 0.35)
        predicted = centers[:, None] + velocities[:, None] * horizon[None, :, None]
        along = (predicted-xy) @ forward
        cross = (predicted-foot) @ lateral
        threat = mature[:, None] & (along > -0.2) & (along < 3.2) & (np.abs(cross) < 1.12)
        # Also permit a bypass for a genuinely corroborated fixed obstruction of the path.
        ahead = foot + np.linspace(0.25, 2.4, 14)[:, None] * forward
        memory = np.array([self.grid.center((r, c)) for c, r in self.memory_cells]).reshape(-1, 2)
        blocked = bool(len(memory) and cKDTree(memory).query(ahead)[0].min() < self.radius + 0.08)
        self.moving_threat = bool(np.any(threat))
        if self.moving_threat or blocked:
            sides = (-1, 1)
            scores = []
            for side in sides:
                ramp = np.minimum(horizon/1.5, 1.)[:, None]
                probes = xy + horizon[:, None]*0.43*forward + ramp * (foot-xy + side*0.76*lateral)
                walls = self._sample(self.live, probes)
                gaps = np.linalg.norm(probes[None, :, :] - predicted, axis=2)
                gap = float(np.min(gaps)) if centers.size else 2.
                recent = self.side == side and t < self.hold_until
                scores.append(1.35*min(gap,1.55) + min(walls.min(),1.) + 0.28*min(walls.mean(),1.)
                              + 0.16*recent + 0.04*(side == -1))
            self.side = sides[int(np.argmax(scores))]
            self.hold_until = t + 0.85
        elif t >= self.hold_until:
            self.side = 0
        if self.side:
            # Pick a reachable lane offset without relaxing the static body reserve.
            width = min(0.59, np.linalg.norm(self.goal - xy) * 0.75)
            for shift in np.linspace(width, 0., 8):
                guide = target + self.side * shift * lateral
                if self._sample(self.static, guide) > self.radius + 0.07:
                    return guide
        return target

    def _rollouts(self, xy, yaw, nominal_w):
        speeds = np.array([0., 0.17, 0.32, 0.47])
        turns = np.unique(np.r_[[-0.9, -0.60, -0.30, 0., 0.30, 0.60, 0.9], nominal_w])
        vs, ws = np.meshgrid(speeds, turns)
        vs, ws = vs.ravel(), ws.ravel()
        v = np.full_like(vs, self.v)
        w = np.full_like(ws, self.w)
        heading = np.full_like(v, yaw)
        trajectory = np.empty((len(v), len(self.times), 2))
        point = np.broadcast_to(xy, (len(v), 2)).copy()
        for i, tm in enumerate(self.times):
            # Every candidate includes a braking tail; matching obstacle times advance while stopped.
            v += np.clip((vs if tm < 1.85 else 0.) - v, -0.9*self.step_s, 0.9*self.step_s)
            w += np.clip((ws if tm < 1.4 else 0.) - w, -2.3*self.step_s, 2.3*self.step_s)
            mid = heading + self.step_s*w/2
            point += self.step_s * v[:, None] * np.column_stack([np.cos(mid), np.sin(mid)])
            heading += w*self.step_s
            trajectory[:, i] = point
        return vs, ws, trajectory, heading

    def command(self, pose, occupancy, revision, tracks, t):
        xy = np.asarray(pose[:2, 3], dtype=float)
        yaw = math.atan2(pose[1, 0], pose[0, 0])
        dt = 0.07 if self.last_time is None else np.clip(t-self.last_time, 0.01, 0.20)
        self.last_time = t
        centers, velocities, radii, mature = self._tracks(tracks, t, xy)
        self._remember(centers[mature], occupancy, t)
        if self.read_revision != revision or self.memory_cells != self.previous_memory:
            self.previous_memory = self.memory_cells.copy()
            free = self.grid.cells == 0
            for col, row in set(occupancy) | self.memory_cells:
                if 0 <= row < self.grid.height and 0 <= col < self.grid.width:
                    free[row, col] = False
            self.live = self._clearance(free)
            self.read_revision = revision
        self.decision = dict(revision=int(revision), valid_candidates=0, nominal=[0.,0.,0.],
                             min_static_m=None, min_live_m=None, min_track_margin_m=None,
                             near_min_m=None, brake_reserve_m=None, memory_cells=len(self.memory_cells),
                             mature_tracks=int(mature.sum()), side=int(self.side))
        self.route = []
        if np.linalg.norm(self.goal - xy) < 0.15:
            self.v = self.w = 0.
            self.decision['reason'] = 'goal'
            return (0., 0., 0.), 'goal'
        reference = self._reference(xy)
        if reference is None or self._sample(self.static, xy) <= self.radius:
            self.v = self.w = 0.
            self.decision['reason'] = 'static_clearance'
            return (0., 0., 0.), 'yield'
        target, forward, foot = reference
        guide = self._guide(xy, target, forward, foot, centers, velocities, mature, t)
        ref = self.reference_command
        if ref is not None and abs(ref[0]-t) < 1e-5 and ref[1] == revision:
            _, _, base_cmd, base_nav, base_route = ref
        else:
            self.tracks = tracks
            base_cmd, base_nav = self.baseline.command(pose, occupancy, revision)
            if OnlineMap.crossing_risk(self, pose, t):
                self.crossing_until = t + 0.42
            if t < self.crossing_until:
                base_cmd, base_nav = (0.,0.,0.), 'yield_to_crossing'
            base_route = self.baseline.route
        if not self.side:
            self.wait_clear = False
        if self.encounter_start is not None:
            travel = np.linalg.norm(xy-self.encounter_start)
            if not self.side or t-self.encounter_time > 1.70 or travel > 0.51:
                self.encounter_start = None
                self.wait_clear = bool(self.side)
        if self.encounter_start is None and self.side and self.moving_threat and not self.wait_clear:
            self.encounter_start = xy.copy()
            self.encounter_time = t
        def follow(reason):
            cmd = tuple(base_cmd)
            route = list(base_route)
            if len(route)>1 and self.memory_cells and (cmd[0] or cmd[2]):
                path = np.linspace(xy, np.asarray(route[1]), 22)
                obstacles = np.array([self.grid.center((r,c)) for c,r in self.memory_cells])
                if cKDTree(obstacles).query(path)[0].min() < self.radius:
                    # Use confirmed static surfaces even if they have left the narrow scan pattern.
                    revised = _route(self.grid, self.live > self.radius, [xy.tolist(),self.goal.tolist()])
                    if revised and len(revised)>1:
                        route = revised
                        d = np.array(route[1])-xy
                        err = math.remainder(math.atan2(d[1],d[0])-yaw,math.tau)
                        cmd = (min(0.42,np.linalg.norm(self.goal-xy))*max(0.,1.-abs(err)/0.56),
                               0.,float(np.clip(2.15*err,-0.9,0.9)))
            if base_nav == 'goal' and np.linalg.norm(self.goal-xy) >= 0.15:
                delta = self.goal-xy
                a = math.remainder(math.atan2(delta[1],delta[0])-yaw,math.tau)
                cmd = (min(0.18,np.linalg.norm(delta))*max(0.,1.-abs(a)/0.56),0.,float(np.clip(2.15*a,-0.9,0.9)))
            safe, margins = self._body_margin(xy,yaw,cmd)
            nav = 'reference_follow' if cmd[0] or cmd[2] else 'yield'
            if not safe and (cmd[0] or cmd[2]):
                # A combined arc can enter a flank point while a smaller pure turn or a straight
                # translation clears it. Evaluate both real sweeps before declaring a dead end.
                choices = [(0.,0.,cmd[2]*0.42), (min(cmd[0],0.13),0.,0.)]
                cmd, nav, reason = (0.,0.,0.), 'yield', 'reference_swept_guard'
                for trial in choices:
                    allowed, gap = self._body_margin(xy,yaw,trial)
                    if allowed and (trial[0] or abs(trial[2])>0.04):
                        cmd,nav,reason,margins = trial, 'reference_follow', 'limited_recovery', gap
                        break
            self.v, _, self.w = cmd
            self.route = route
            self.decision.update(reason=reason, reference_nav=base_nav, nominal=list(base_cmd), side=int(self.side),
                                 body_now_m=margins[0], body_worst_m=margins[1], body_end_m=margins[2],
                                 near_min_m=float(np.linalg.norm(self.hits[:,:2]-xy,axis=1).min()) if len(self.hits) else None)
            return cmd, nav
        if self.encounter_start is None:
            return follow('reference_recovery' if self.wait_clear else 'reference')
        angle = math.remainder(math.atan2(*(guide - xy)[::-1]) - yaw, math.tau)
        nominal_w = np.clip(2.15*angle, -0.90, 0.90)
        vs, ws, rollout, final_yaw = self._rollouts(xy, yaw, nominal_w)
        static = self._sample(self.static, rollout).min(axis=1)
        live = self._sample(self.live, rollout).min(axis=1)
        margin = np.full(len(vs), np.inf)
        for center, velocity, radius in zip(centers, velocities, radii):
            predicted = center + self.times[:, None] * velocity
            margin = np.minimum(margin, np.min(np.linalg.norm(rollout - predicted, axis=2) - radius - 0.12*np.linalg.norm(velocity), axis=1))
        # First returns also guard the blind area between scans before a blob has enough hits.
        # Use the swept body disk (including rotation), one scan of reaction and braking distance.
        near_pts = self.hits[np.linalg.norm(self.hits[:, :2]-xy, axis=1) < 2.0, :2]
        near_min = np.full(len(vs), np.inf)
        reserve = 0.055 + 0.09*vs + vs**2/(2*0.9) + 0.025*np.abs(ws)
        if len(near_pts):
            near_tree = cKDTree(near_pts)
            near_min = near_tree.query(rollout[:, :8, :])[0].min(axis=1)
            self.decision['near_min_m'] = float(near_tree.query(xy)[0])
        valid = (static > self.radius) & (live > self.radius) & (margin >= 0.)
        reference_yaw = math.atan2(target[1] - xy[1], target[0] - xy[0])
        reference_error = math.remainder(reference_yaw - yaw, math.tau)
        nominal = [min(0.47, np.linalg.norm(self.goal-xy)) * max(0., 1.-abs(reference_error)/0.56),
                   0., float(np.clip(2.15*reference_error, -0.90, 0.90))]
        self.decision.update(valid_candidates=int(valid.sum()), candidates=len(vs), tracks=len(centers),
                             side=int(self.side), guide=guide.tolist(), nominal=nominal)
        if not np.any(valid):
            self.v = self.w = 0.
            self.decision.update(reason='brake_no_safe_rollout', brake_reserve_m=float(reserve.min()),
                                 min_static_m=float(static.max()),
                                 min_live_m=float(live.max()),
                                 min_track_margin_m=float(margin.max()) if centers.size else None)
            return follow('reference_after_infeasible_active')
        end = rollout[:, -1]
        heading_error = np.arctan2(guide[1]-xy[1], guide[0]-xy[0]) - final_yaw
        score = (-1.8*np.linalg.norm(end-guide, axis=1) + 0.16*np.cos(heading_error)
                 + 0.11*np.clip(live-self.radius, 0., 0.4) + 0.30*np.clip(margin, 0., 0.40)
                 - 0.025*np.abs(ws-self.w) - 0.04*np.abs(ws-nominal_w))
        i = int(np.argmax(np.where(valid, score, -np.inf)))
        self.v += float(np.clip(vs[i]-self.v, -0.9*dt, 0.9*dt))
        self.w += float(np.clip(ws[i]-self.w, -2.3*dt, 2.3*dt))
        safe, margins = self._body_margin(xy, yaw, (self.v,0.,self.w))
        self.decision.update(body_now_m=margins[0], body_worst_m=margins[1], body_end_m=margins[2])
        if not safe:
            return follow('reference_after_near_field')
        # Display the selected local trajectory. Beyond it the global continuation is only
        # drawn where routed on the current observed free grid, and is rechecked next scan.
        prefix = [xy.tolist(), *rollout[i, 2:16:3].tolist()]
        continuation = _route(self.grid, self.live > self.radius, [prefix[-1], self.goal.tolist()])
        self.route = prefix + (continuation[1:] if continuation else [])
        nav = 'active_detour' if self.side else 'active_follow'
        if vs[i] == 0 and abs(ws[i]) < 0.01:
            nav = 'yield'
        self.decision.update(reason=nav, brake_reserve_m=float(reserve[i]), min_static_m=float(static[i]), min_live_m=float(live[i]),
                             min_track_margin_m=float(margin[i]) if centers.size else None)
        return (self.v, 0., self.w), nav

"""Footprint-aware 2-D inspection planning and bounded unicycle commands (metres, seconds)."""

import hashlib
import heapq
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


def _number(value, name, low=-1e6, high=1e6):
    if isinstance(value, bool):
        raise ValueError(f"Некорректное значение {name}")
    try:
        value = float(value)
    except (ValueError, TypeError) as error:
        raise ValueError(f"Некорректное значение {name}") from error
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"Значение {name} вне допустимого диапазона [{low}, {high}]")
    return value


def _point(value):
    if isinstance(value, dict):
        value = [value.get("x"), value.get("y")]
    if not isinstance(value, (tuple, list, np.ndarray)) or len(value) != 2:
        raise ValueError("Точка должна содержать две координаты в метрах")
    return (_number(value[0], "x"), _number(value[1], "y"))


def _result(status, path, reason="", events=None):
    length = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
    return dict(
        status=status,
        path=[list(p) for p in path],
        reason=reason,
        events=list(events or []),
        length=round(length, 4),
    )


class Grid:
    def __init__(self, path):
        with np.load(Path(path), allow_pickle=False) as data:
            self.cells = np.array(data["cells"])
            self.origin = np.array(data["origin"], dtype=float)
            self.resolution = _number(
                np.asarray(data["resolution"]).item(), "resolution", 0.001, 100
            )
        if (
            self.cells.ndim != 2
            or self.cells.size == 0
            or self.origin.shape != (2,)
            or not np.isfinite(self.origin).all()
        ):
            raise ValueError("Некорректная метрическая сетка")
        self.height, self.width = self.cells.shape
        self.fingerprint = hashlib.sha256(
            self.cells.tobytes() + self.origin.tobytes() + str(self.resolution).encode()
        ).hexdigest()
        self._free_key = self._free = None

    def window(self, free, a, b, margin=2.5):
        """Use a bounded local subset with exactly the same cell boundaries as the map."""
        lo = np.minimum(a, b) - margin
        hi = np.maximum(a, b) + margin
        r0, c0 = self.index(lo)
        r1, c1 = self.index(hi)
        r0, c0 = max(0, r0), max(0, c0)
        r1, c1 = min(self.height, r1 + 1), min(self.width, c1 + 1)
        sub = object.__new__(Grid)
        sub.origin = self.origin + self.resolution * np.array([c0, r0])
        sub.resolution = self.resolution
        mask = free[r0:r1, c0:c1]
        sub.height, sub.width = mask.shape
        return sub, mask

    def index(self, p):
        c, r = np.floor((np.asarray(p) - self.origin) / self.resolution).astype(int)
        return int(r), int(c)

    def center(self, cell):
        r, c = cell
        x, y = self.origin + (np.array([c, r]) + 0.5) * self.resolution
        return float(x), float(y)

    def pixels(self, points):
        scaled = ((np.asarray(points) - self.origin) / self.resolution - 0.5) * 256
        return np.clip(np.rint(scaled), -(2**29), 2**29).astype(np.int32)

    def clearance(self, zones):
        free = np.asarray(self.cells == 0, dtype=np.uint8)
        for zone in zones:
            polygon = zone.get("points") if isinstance(zone, dict) else zone
            if not isinstance(polygon, (list, tuple)) or not 3 <= len(polygon) <= 4096:
                raise ValueError("Зона должна иметь от 3 до 4096 вершин")
            polygon = np.asarray([_point(v) for v in polygon])
            if (
                abs(
                    np.dot(polygon[:, 0], np.roll(polygon[:, 1], 1))
                    - np.dot(polygon[:, 1], np.roll(polygon[:, 0], 1))
                )
                < 1e-10
            ):
                raise ValueError("Нулевая площадь запретной зоны")
            cv2.fillPoly(free, [self.pixels(polygon)], 0, shift=8)
        clearance = (
            distance_transform_edt(np.pad(free, 1))[1:-1, 1:-1] * self.resolution
        )
        # EDT is measured between cell centres. Allow for both obstacle and swept path cells.
        return clearance - math.sqrt(2) * self.resolution

    def traversable(self, zones, radius):
        return self.clearance(zones) > radius

    def disk_clear(self, zones):
        clearance = self.clearance(zones)
        ox, oy = self.origin

        def clear(x, y, radius):
            r = math.floor((y - oy) / self.resolution)
            c = math.floor((x - ox) / self.resolution)
            return (
                0 <= r < self.height
                and 0 <= c < self.width
                and bool(clearance[r, c] > radius)
            )

        return clear

    def dynamic_mask(self, free, tracks, radius):
        if len(tracks) > 4096:
            raise ValueError("Слишком много препятствий")
        mask = free.copy()
        for track in tracks:
            if not isinstance(track, dict):
                raise ValueError("Трек должен быть объектом")
            x, y = _point(track)
            vx = _number(track.get("vx", 0), "vx", -100, 100)
            vy = _number(track.get("vy", 0), "vy", -100, 100)
            size = _number(track.get("radius", 0.25), "obstacle radius", 0.01, 1e6)
            age = _number(track.get("age_s", track.get("age", 0)), "age", 0, 3600)
            ttl = _number(track.get("ttl_s", track.get("ttl", 2)), "ttl", 0, 3600)
            if age >= ttl:
                continue
            # Sweep a disc along the short, predicted track in world coordinates.
            x += vx * age
            y += vy * age
            horizon = min(1.5, ttl - age)
            ex, ey = x + vx * horizon, y + vy * horizon
            margin = (
                radius
                + size
                + self.resolution / math.sqrt(2)
                + _number(track.get("position_error_m", 0), "position error", 0, 1000)
                + _number(track.get("velocity_error_mps", 0), "velocity error", 0, 1000)
                * (age + horizon)
            )
            r0, c0 = self.index((min(x, ex) - margin, min(y, ey) - margin))
            r1, c1 = self.index((max(x, ex) + margin, max(y, ey) + margin))
            r0, c0, r1, c1 = (
                max(0, r0),
                max(0, c0),
                min(self.height - 1, r1),
                min(self.width - 1, c1),
            )
            if r0 > r1 or c0 > c1:
                continue
            cx = self.origin[0] + (np.arange(c0, c1 + 1) + 0.5) * self.resolution - x
            cy = (
                self.origin[1]
                + (np.arange(r0, r1 + 1)[:, None] + 0.5) * self.resolution
                - y
            )
            dx, dy = ex - x, ey - y
            dot = np.clip((cx * dx + cy * dy) / max(dx * dx + dy * dy, 1e-18), 0, 1)
            mask[r0 : r1 + 1, c0 : c1 + 1] &= (cx - dot * dx) ** 2 + (
                cy - dot * dy
            ) ** 2 > margin * margin
        return mask


def _inside(free, cell):
    return 0 <= cell[0] < free.shape[0] and 0 <= cell[1] < free.shape[1] and free[cell]


def _visible(free, a, b):
    # Integer supercover between two cell centres; corner crossings touch both side cells.
    y, x = a
    dy, dx = b[0] - y, b[1] - x
    sy, sx = (1 if dy >= 0 else -1), (1 if dx >= 0 else -1)
    ny, nx = abs(dy), abs(dx)
    iy = ix = 0
    while ix < nx or iy < ny:
        cross = (1 + 2 * ix) * ny - (1 + 2 * iy) * nx
        if cross == 0:
            if not free[y + sy, x] or not free[y, x + sx]:
                return False
            x += sx
            y += sy
            ix += 1
            iy += 1
        elif cross < 0:
            x += sx
            ix += 1
        else:
            y += sy
            iy += 1
        if not free[y, x]:
            return False
    return True


def _astar(free, start, goal):
    if not _inside(free, start) or not _inside(free, goal):
        return None
    if _visible(free, start, goal):
        return [start, goal]
    frontier = [(math.dist(start, goal), 0, start)]
    costs = {start: 0.0}
    previous = {}
    offsets = ((0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, -1), (-1, 1))
    while frontier:
        _, cost, cell = heapq.heappop(frontier)
        if cell == goal:
            path = [goal]
            while path[-1] != start:
                path.append(previous[path[-1]])
            path.reverse()
            result = [start]
            for idx in range(2, len(path)):
                if not _visible(free, result[-1], path[idx]):
                    result.append(path[idx - 1])
            result.append(goal)
            return result
        if cost != costs[cell]:
            continue
        y, x = cell
        for dy, dx in offsets:
            neighbor = y + dy, x + dx
            if not _inside(free, neighbor):
                continue
            if dy and dx and (not free[y + dy, x] or not free[y, x + dx]):
                continue
            nxt = cost + (math.sqrt(2) if dy and dx else 1.0)
            if nxt >= costs.get(neighbor, math.inf):
                continue
            previous[neighbor] = cell
            costs[neighbor] = nxt
            heapq.heappush(frontier, (nxt + math.dist(neighbor, goal), nxt, neighbor))
    return None


def _line_world(grid, free, a, b):
    """Supercover of a segment whose endpoints are arbitrary metric positions."""
    y, x = grid.index(a)
    gy, gx = grid.index(b)
    delta = np.array(b) - a
    sx, sy = np.sign(delta).astype(int)
    tx, ty = math.inf, math.inf
    dx, dy = math.inf, math.inf
    if sx:
        tx = (grid.origin[0] + (x + (sx > 0)) * grid.resolution - a[0]) / delta[0]
        dx = grid.resolution / abs(delta[0])
    if sy:
        ty = (grid.origin[1] + (y + (sy > 0)) * grid.resolution - a[1]) / delta[1]
        dy = grid.resolution / abs(delta[1])
    if not _inside(free, (y, x)):
        return False
    for _ in range(abs(gx - x) + abs(gy - y) + 1):
        if (y, x) == (gy, gx):
            return True
        if x == gx:
            y += sy
            ty += dy
        elif y == gy:
            x += sx
            tx += dx
        elif abs(tx - ty) < 1e-10:
            if not _inside(free, (y + sy, x)) or not _inside(free, (y, x + sx)):
                return False
            x += sx
            y += sy
            tx += dx
            ty += dy
        elif tx < ty:
            x += sx
            tx += dx
        else:
            y += sy
            ty += dy
        if not _inside(free, (y, x)):
            return False
    return False


def _route(grid, free, points):
    path = [points[0]]
    for a, b in zip(points, points[1:]):
        route = _astar(free, grid.index(a), grid.index(b))
        if route is None:
            return None
        local = [a]
        candidates = [*(grid.center(cell) for cell in route), b]
        for i, item in enumerate(candidates):
            if i + 1 < len(candidates) and _line_world(
                grid, free, local[-1], candidates[i + 1]
            ):
                continue
            if math.dist(local[-1], item) > 1e-8:
                local.append(item)
        path.extend(local[1:])
    return path


def follower_tracks(tracks, included_margin=0.):
    """Allow posture and turn clearance around every accepted sensor prediction."""
    reserve = max(0., .55-included_margin)
    return [dict(track, radius=track["radius"]+reserve, follower_reserve_m=reserve) for track in tracks]


def detour_tracks(tracks):
    """Follower reserve affects planning only. Moving obstacles are handled by yielding."""
    return [
        dict(t, radius=t["radius"] + 0.55 + t.get("position_error_m", 0.0))
        for t in tracks
        if math.hypot(t["vx"], t["vy"]) < 1e-8
    ]


def plan(
    grid_path, start, goals, zones=(), dynamic=(), radius=0.30, *, static_margin=0.0
):
    """Plan all ordered goals on known free space, predicting short-lived obstacles."""
    radius = _number(radius, "robot radius", 0.01, 5)
    start = _point(start)
    if len(goals) > 512 or len(zones) > 512:
        raise ValueError("Слишком много целей или зон")
    points = [start] + [_point(g) for g in goals]
    if len(points) == 1:
        return _result("complete", [start], "Все цели пройдены")
    grid = grid_path if isinstance(grid_path, Grid) else Grid(grid_path)
    free = grid.traversable(
        zones, radius + _number(static_margin, "static margin", 0, 1)
    )
    if not _inside(free, grid.index(start)):
        return _result(
            "blocked",
            [],
            "Поза робота вне известной свободной области с требуемым зазором",
        )
    if any(not _inside(free, grid.index(g)) for g in points[1:]):
        return _result(
            "blocked", [], "Цель недоступна с учётом запретных зон и габарита робота"
        )
    events = []
    path = _route(grid, free, points)
    if path is None:
        return _result(
            "blocked",
            [],
            "Свободные области разделены препятствием, зоной или неизвестными ячейками",
        )
    if dynamic:
        with_tracks = grid.dynamic_mask(free, dynamic, radius)
        if not np.array_equal(free, with_tracks):
            alternate = _route(grid, with_tracks, points)
            if alternate is None:
                return _result(
                    "waiting", [], "Ожидание освобождения пути", ["dynamic_wait"]
                )
            if alternate != path:
                events.append("dynamic_replan")
            path = alternate
    return _result("ready", path, events=events)


def global_plan(grid, pose, goals, index, zones, radius, margin, previous=None):
    """Retain the absolute global polyline and goal boundaries while local inputs change."""
    xy = _point(pose)
    if len(goals) > 512 or not 0 <= index <= len(goals):
        raise ValueError("Недопустимая очередь целей")
    geom_key = json.dumps(
        [zones, radius, margin], sort_keys=True, separators=(",", ":")
    )
    if grid._free_key != geom_key:
        grid._free = grid.traversable(
            zones, _number(radius, "radius", 0.01, 5) + _number(margin, "margin", 0, 1)
        )
        grid._free_key = geom_key
    key = [grid.fingerprint, geom_key, [list(_point(g)) for g in goals]]
    if (
        previous
        and "blocked_index" in previous
        and previous.get("key") == key
        and previous["first"] <= index
    ):
        return previous
    path = [list(xy)]
    ends = []
    for goal in goals[index:]:
        segment = _route(grid, grid._free, [_point(path[-1]), _point(goal)])
        if segment is None:
            return dict(
                key=key,
                first=index,
                ends=ends,
                path=path if ends else [],
                blocked_index=index + len(ends),
            )
        path.extend([list(p) for p in segment[1:]])
        ends.append(len(path) - 1)
    return dict(key=key, first=index, ends=ends, path=path, blocked_index=None)


def _subgoal(points, xy, previous, horizon):
    # Bound projection to the reachable arc of this goal leg to avoid jumping at crossings.
    limit = previous + max(horizon, 0.25)
    best = (math.inf, previous)
    segments = []
    length = 0.0
    for a, b in zip(points, points[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        size = math.hypot(dx, dy)
        if size == 0:
            continue
        start = length
        length += size
        segments.append((a, dx, dy, start, size))
        if length < previous or start > limit:
            continue
        k = ((xy[0] - a[0]) * dx + (xy[1] - a[1]) * dy) / size**2
        k = max(0.0, (previous - start) / size, min(1.0, (limit - start) / size, k))
        best = min(
            best,
            (
                math.hypot(xy[0] - a[0] - k * dx, xy[1] - a[1] - k * dy),
                start + k * size,
            ),
        )
    arc = min(length, best[1] + horizon)
    for a, dx, dy, start, size in segments:
        if arc <= start + size:
            f = (arc - start) / size
            return [a[0] + dx * f, a[1] + dy * f], best[1]
    return list(points[-1]), best[1]


def _outward(xy, end, nearby):
    return all((xy[0]-cx)*(end[0]-xy[0]) + (xy[1]-cy)*(end[1]-xy[1]) >= -1e-8
               for cx,cy in nearby)


def _nearby_predictions(xy, tracks, radius):
    nearby = []
    for t in tracks:
        dx,dy = t["vx"],t["vy"]
        horizon = min(1.5,t.get("ttl_s",1.5))
        closest = max(0.,min(horizon,
                        ((xy[0]-t["x"])*dx + (xy[1]-t["y"])*dy) / max(dx*dx+dy*dy,1e-12)))
        centre = (t["x"]+dx*closest,t["y"]+dy*closest)
        if math.dist(centre,xy)<=radius+t["radius"]+t.get("position_error_m",0)+horizon*t.get("velocity_error_mps",0):
            nearby.append(centre)
    return nearby


def _reserve_exit(grid, mask, hard, xy, nearby):
    """Reconnect a displaced pose to the preferred corridor through a short hard-clear segment."""
    start = grid.index(xy)
    if not _inside(hard, start):
        return None
    extent = math.ceil(.45/grid.resolution)
    r,c = start
    r0,c0 = max(0,r-extent),max(0,c-extent)
    candidates=np.argwhere(mask[r0:r+extent+1,c0:c+extent+1])
    if not len(candidates):
        return None
    cells=candidates+np.array([r0,c0])
    centres=grid.origin+(cells[:,::-1]+.5)*grid.resolution
    distances=np.linalg.norm(centres-xy,axis=1)
    for i in np.argsort(distances):
        if distances[i]>.45:
            break
        exit_xy=centres[i].tolist()
        if _outward(xy,exit_xy,nearby) and _line_world(grid,hard,xy,exit_xy):
            return exit_xy
    return None


def local_plan(
    grid,
    pose,
    route,
    index,
    tracks=(),
    obstacles=(),
    *,
    radius,
    margin=0.0,
    previous_progress=None,
    unavailable=None,
):
    full = route["path"]
    xy = list(_point(pose))
    leg = index - route["first"]
    state = dict(
        global_path=full,
        local_path=[],
        local_target=None,
        local_status="waiting",
        local_reason=unavailable or "clear",
        local_progress=dict(index=index, arc=0.0),
        events=[],
    )
    blocked_index = route.get("blocked_index")
    state.update(
        global_status="complete"
        if blocked_index is None
        else ("partial" if full else "blocked"),
        global_reason="clear" if blocked_index is None else "global_no_route",
        global_blocked_index=blocked_index,
    )
    if not full or blocked_index is not None and index >= blocked_index:
        return dict(state, local_status="blocked", local_reason="global_no_route")
    if leg >= len(route["ends"]):
        return dict(state, local_status="complete", local_reason="goals_complete")
    start = route["ends"][leg - 1] if leg else 0
    points = full[start : route["ends"][leg] + 1]
    prev = previous_progress or {}
    arc = prev.get("arc", 0.0) if prev.get("index") == index else 0.0
    target, progress = _subgoal(points, xy, arc, 2.0)
    state.update(local_target=target, local_progress=dict(index=index, arc=progress))
    if unavailable:
        return state
    # A detour whose nominal subgoal is itself occupied can rejoin the current leg
    # a little farther ahead. The executed polyline is clipped to the local horizon.
    ahead, _ = _subgoal(points, xy, arc, 3.6)
    sub, base = grid.window(grid._free, xy, ahead)
    static = sub.dynamic_mask(base, obstacles, radius + margin) if obstacles else base
    mask = sub.dynamic_mask(static, tracks, radius + margin) if tracks else static
    start_xy = xy
    if tracks and not _inside(mask,sub.index(xy)) and any(t.get("follower_reserve_m",0) for t in tracks):
        hard_tracks=[dict(t,radius=t["radius"]-t.get("follower_reserve_m",0)) for t in tracks]
        hard = sub.dynamic_mask(static,hard_tracks,radius+margin)
        nearby = _nearby_predictions(xy,tracks,radius+margin+sub.resolution*math.sqrt(2))
        start_xy = _reserve_exit(sub,mask,hard,xy,nearby)

    def route_to(endpoint):
        if start_xy is None:
            return None
        candidate = _route(sub,mask,[start_xy,endpoint])
        if candidate and start_xy != xy:
            if len(candidate)>1:
                nxt=candidate[1]
                length=math.dist(xy,nxt)
                if _outward(xy,nxt,nearby) and _line_world(sub,hard,xy,nxt) and length>0:
                    for distance in np.arange(sub.resolution*.5,min(length,.45)+.0001,sub.resolution*.5):
                        entry=[a+(b-a)*distance/length for a,b in zip(xy,nxt)]
                        if _line_world(sub,mask,entry,nxt):
                            return [xy,*candidate[1:]]
            return [xy,*candidate]
        return candidate

    path=route_to(target)
    if (obstacles or tracks) and ahead != target:
        continuation = route_to(ahead)
        if path is None and continuation:
            clipped = [continuation[0]]
            remaining = 2.0
            for a, b in zip(continuation, continuation[1:]):
                d = math.dist(a, b)
                if d > remaining:
                    clipped.append(
                        [a[i] + (b[i] - a[i]) * remaining / d for i in (0, 1)]
                    )
                    break
                clipped.append(b)
                remaining -= d
            path = clipped
            state["local_target"] = list(path[-1])
        elif tracks and continuation is None and _route(sub, static, [xy, ahead]):
            path = None
    if path is None:
        blocked_static = bool(obstacles and _route(sub, static, [xy, target]) is None)
        return dict(
            state,
            local_reason="local_static_blocked"
            if blocked_static or not tracks
            else "local_dynamic_wait",
            events=["dynamic_wait"] if tracks and not blocked_static else [],
        )
    state.update(local_path=[list(p) for p in path], local_status="ready")
    if start_xy != xy and len(path)>1 and list(path[1])==start_xy:
        r,c=sub.index(start_xy)
        lo=sub.origin+sub.resolution*np.array([c,r])
        state["reserve_exit_cell"]=[*lo, *(lo+sub.resolution)]
    if tracks and (
        not _line_world(sub, mask, xy, target) or state["local_target"] != target
    ):
        if path != _route(sub, static, [xy, target]):
            state["events"] = ["dynamic_replan"]
    return state


def step(
    grid_path,
    start,
    goals,
    zones=(),
    dynamic=(),
    radius=0.30,
    dt=0.5,
    speed=0.4,
    *,
    global_route=None,
    local_obstacles=(),
):
    """One deterministic inspection step. Commands rotate in place before moving."""
    dt = _number(dt, "dt", 0.01, 1.0)
    speed = _number(speed, "speed", 0.01, 2.0)
    xy = _point(start)
    yaw = _number(start.get("yaw", 0) if isinstance(start, dict) else 0, "yaw")
    goal_list = [_point(g) for g in goals]
    reached = 0
    while reached < len(goal_list) and math.dist(xy, goal_list[reached]) < 0.025:
        reached += 1
    remaining = goal_list[reached:]
    if len(remaining) > 512:
        raise ValueError("Слишком много целей")
    grid = grid_path if isinstance(grid_path, Grid) else Grid(grid_path)
    route = global_plan(grid, xy, goal_list, reached, zones, radius, 0.0, global_route)
    local = local_plan(
        grid,
        xy,
        route,
        reached,
        dynamic,
        local_obstacles,
        radius=radius,
        previous_progress=(global_route or {}).get("progress"),
    )
    result = _result(
        local["local_status"],
        local["local_path"],
        local["local_reason"],
        local["events"],
    )
    result.update(local, global_route=dict(route, progress=local["local_progress"]))
    pose = dict(x=xy[0], y=xy[1], yaw=yaw)
    command = dict(linear=0.0, angular=0.0)
    result.update(pose=pose, reached=reached, velocity=command)
    if result["status"] != "ready":
        return result
    target = result["path"][1]
    heading = math.atan2(target[1] - xy[1], target[0] - xy[0])
    error = math.atan2(math.sin(heading - yaw), math.cos(heading - yaw))
    distance = min(dt * speed, math.dist(xy, target))
    candidate = (xy[0] + distance * math.cos(yaw), xy[1] + distance * math.sin(yaw))
    turn_first = abs(error) > 0.06
    if 1e-8 < abs(error) <= 0.06:
        # With a small yaw error the physical robot travels slightly off its
        # planned centreline. Test that actual world segment before permitting it.
        free = grid.dynamic_mask(grid._free, local_obstacles, radius) if local_obstacles else grid._free
        free = grid.dynamic_mask(free,dynamic,radius) if dynamic else free
        turn_first = not _line_world(grid, free, xy, candidate)
    if turn_first:
        turn = max(-1.0, min(1.0, error / dt))
        command["angular"] = turn
        pose["yaw"] = math.atan2(math.sin(yaw + turn * dt), math.cos(yaw + turn * dt))
        result["status"] = "moving"
        return result
    command["linear"] = distance / dt
    pose["x"], pose["y"] = candidate
    while (
        reached < len(goal_list)
        and math.dist((pose["x"], pose["y"]), goal_list[reached]) < 0.025
    ):
        reached += 1
        result["events"].append("goal_reached")
    result["reached"] = reached
    result["status"] = "complete" if reached == len(goal_list) else "moving"
    return result

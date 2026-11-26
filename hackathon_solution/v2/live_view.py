"""Native, interactive MuJoCo view of the same data being stepped by navigation."""

import mujoco as mj
import mujoco.viewer
import numpy as np
import time
import threading


class NativeScene:
    def __init__(self, sim, goal):
        self.goal = goal
        self.trail = []
        self.last_sync = 0.
        sim.model.vis.map.znear = 0.015 / sim.model.stat.extent
        sim.model.vis.headlight.ambient[:] = 0.45
        sim.model.geom_rgba[sim.floor, :3] = [0.34, 0.38, 0.4]
        threads_before = set(threading.enumerate())
        self.view = mujoco.viewer.launch_passive(
            sim.model, sim.data, show_left_ui=False, show_right_ui=False
        )
        # On Linux the passive viewer runs its GL teardown on a daemon thread.
        self.ui_threads = [t for t in threading.enumerate() if t not in threads_before
                           and "_launch_internal" in t.name]
        with self.view.lock():
            self.view.cam.lookat[:] = [2.68, 6.25, 0.53]
            self.view.cam.distance = 6.9
            self.view.cam.azimuth = 153
            self.view.cam.elevation = -39
            self.view.opt.geomgroup[3] = 0
            self.view.opt.geomgroup[5] = 1
        self.view.sync()

    def sync(self, route, xy, nav):
        if not self.view.is_running():
            return
        now = time.monotonic()
        if now - self.last_sync < .11 and nav != "goal":
            return
        self.last_sync = now
        if not self.trail or np.linalg.norm(xy - self.trail[-1]) > 0.075:
            self.trail.append(xy.copy())
        with self.view.lock():
            scene = self.view.user_scn
            scene.ngeom = 0
            ident = np.eye(3).ravel()

            def geom(kind, size, position, color):
                g = scene.geoms[scene.ngeom]
                mj.mjv_initGeom(
                    g,
                    kind,
                    np.asarray(size),
                    np.asarray(position),
                    ident,
                    np.array(color, dtype="f4"),
                )
                scene.ngeom += 1
                return g

            geom(
                mj.mjtGeom.mjGEOM_CYLINDER,
                [0.16, 0.018, 0],
                [*self.goal, 0.024],
                [0.3, 0.92, 0.43, 0.94],
            )
            for path, z, color in (
                (self.trail, 0.052, [0.16, 0.9, 0.79, 1]),
                (route, 0.065, [1, 0.77, 0.19, 1]),
            ):
                for a, b in zip(path, path[1:]):
                    if scene.ngeom >= scene.maxgeom:
                        break
                    g = geom(
                        mj.mjtGeom.mjGEOM_CAPSULE,
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        color,
                    )
                    mj.mjv_connector(
                        g, mj.mjtGeom.mjGEOM_CAPSULE, 0.017, [*a, z], [*b, z]
                    )
        self.view.sync()

    def close(self):
        self.view.close()
        for thread in self.ui_threads:
            thread.join()

    def wait_for_close(self):
        """Keep the measured final scene interactive after all metrics have been written."""
        while self.view.is_running():
            self.view.sync()
            time.sleep(0.05)

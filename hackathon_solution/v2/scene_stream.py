"""Export the current physics state for RViz running outside the MuJoCo container."""

import hashlib
import json
import os
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation

from .wall_surface import FACET


def save_mesh(model, mesh, directory):
    start = model.mesh_vertadr[mesh]
    vertices = model.mesh_vert[start : start + model.mesh_vertnum[mesh]]
    start = model.mesh_faceadr[mesh]
    triangles = vertices[model.mesh_face[start : start + model.mesh_facenum[mesh]]]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    facets = np.zeros(len(triangles), dtype=FACET)
    facets["normal"] = normals / np.maximum(
        np.linalg.norm(normals, axis=1, keepdims=True), 1e-15
    )
    facets["vertices"] = triangles
    data = bytes(80) + len(facets).to_bytes(4, "little") + facets.tobytes()
    path = directory / (hashlib.sha256(data).hexdigest() + ".stl")
    if not path.exists():
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
    return path.name


class SceneStream:
    def __init__(self, directory, sim, goal):
        self.directory = Path(directory)
        models = self.directory / "models"
        models.mkdir(parents=True, exist_ok=True)
        self.sim = sim
        self.track = []
        self.ids = []
        model = sim.model
        bodies = {model.body(name).id for name in ("barrier", "pedestrian", "side")}
        geometry = []
        meshes = {}
        for i in range(model.ngeom):
            kind = model.geom_type[i]
            if kind == mj.mjtGeom.mjGEOM_MESH and model.geom_bodyid[i] != 0:
                if model.geom_contype[i] or model.geom_conaffinity[i]:
                    continue
                mesh = model.geom_dataid[i]
                if mesh not in meshes:
                    meshes[mesh] = save_mesh(model, mesh, models)
                shape, size, file = "mesh", [1.0, 1.0, 1.0], meshes[mesh]
            elif model.geom_bodyid[i] in bodies:
                size = model.geom_size[i]
                file = ""
                if kind == mj.mjtGeom.mjGEOM_BOX:
                    shape, size = "box", 2 * size
                elif kind == mj.mjtGeom.mjGEOM_SPHERE:
                    shape, size = "sphere", [2 * size[0]] * 3
                else:
                    shape, size = "cylinder", [2 * size[0], 2 * size[0], 2 * size[1]]
            else:
                continue
            mat = model.geom_matid[i]
            color = model.mat_rgba[mat] if mat >= 0 else model.geom_rgba[i]
            geometry.append(
                dict(shape=shape, scale=list(size), file=file, rgba=color.tolist())
            )
            self.ids.append(i)
        self.joints = [
            j for j in range(model.njnt) if model.jnt_type[j] == mj.mjtJoint.mjJNT_HINGE
        ]
        self.epoch = os.urandom(8).hex()
        names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j) for j in self.joints]
        description = dict(
            epoch=self.epoch,
            case=sim.scenario,
            geometry=geometry,
            joints=names,
            goal=goal.tolist(),
            actors=[
                name
                for name, actor in (("Go2-2", sim.other), ("G1", sim.humanoid))
                if actor is not None
            ],
        )
        tmp = self.directory / "model.tmp"
        tmp.write_text(json.dumps(description))
        tmp.replace(self.directory / "model.json")

    def publish(self, pose, points, mapper, plan, nav, scan):
        d, sim = self.sim.data, self.sim
        self.track.append(pose[:2, 3].copy())
        tmp = self.directory / "current.tmp.npz"
        np.savez(
            tmp,
            epoch=self.epoch,
            seconds=d.time,
            scan=scan,
            nav=nav,
            revision=mapper.revision,
            track=self.track,
            route=np.asarray(plan.route).reshape(-1, 2),
            xy=d.xpos[sim.primary.base],
            actors=np.asarray(
                [d.xpos[a.base] for a in (sim.other, sim.humanoid) if a is not None]
            ).reshape(-1, 3),
            joints=d.qpos[sim.model.jnt_qposadr[self.joints]],
            pos=d.geom_xpos[self.ids],
            quat=Rotation.from_matrix(
                d.geom_xmat[self.ids].reshape(-1, 3, 3)
            ).as_quat(),
            lidar=points @ pose[:3, :3].T + pose[:3, 3],
            added=mapper.last_novel,
        )
        tmp.replace(self.directory / "current.npz")

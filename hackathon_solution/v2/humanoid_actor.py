"""Unitree G1 velocity tracking in MuJoCo using the manufacturer's 12-actuator policy."""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).resolve().parent / "sim_assets" / "humanoid"
JOINTS = tuple(
    f"hum_{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in (
        "hip_pitch",
        "hip_roll",
        "hip_yaw",
        "knee",
        "ankle_pitch",
        "ankle_roll",
    )
)
DEFAULT_ANGLES = np.tile([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0], 2)
KP = np.tile([100.0, 100.0, 100.0, 150.0, 40.0, 40.0], 2)
KD = np.tile([2.0, 2.0, 2.0, 4.0, 2.0, 2.0], 2)


def augment(root_xml, workdir):
    """Append the official G1 assets, floating body and motors to the Go2 scene tree."""
    root = root_xml.getroot() if isinstance(root_xml, ET.ElementTree) else root_xml
    g1 = ET.parse(ASSETS / "g1_12dof.xml").getroot()
    defaults = root.find("default")
    if defaults is None:
        defaults = ET.SubElement(root, "default")
    g1_defaults = ET.SubElement(defaults, "default", {"class": "humanoid"})
    g1_defaults.extend(g1.find("default"))
    body = g1.find("worldbody/body")
    body.set("childclass", "humanoid")
    for element in g1.iter():
        for key in ("name", "mesh", "joint"):
            if key in element.attrib:
                element.set(key, "hum_" + element.attrib[key])
    for mesh in g1.findall("asset/mesh"):
        mesh.set("file", str(ASSETS / "meshes" / mesh.attrib["file"]))
        root.find("asset").append(mesh)
    root.find("worldbody").append(body)
    root.find("actuator").extend(g1.find("actuator"))


class WalkingPolicy:
    """NumPy inference for Unitree's released LSTM(47,64), linear, ELU, linear actor."""

    def __init__(self):
        with np.load(ASSETS / "motion.npz", allow_pickle=False) as weights:
            self.ih = weights["memory.weight_ih_l0"]
            self.hh = weights["memory.weight_hh_l0"]
            self.bias = weights["memory.bias_ih_l0"] + weights["memory.bias_hh_l0"]
            self.fc1, self.fc2 = weights["actor.0.weight"], weights["actor.2.weight"]
            self.b1, self.b2 = weights["actor.0.bias"], weights["actor.2.bias"]
        self.reset()

    def reset(self):
        self.hidden = np.zeros(64, dtype=np.float32)
        self.cell = np.zeros(64, dtype=np.float32)

    def __call__(self, observation):
        gates = (
            self.ih @ np.asarray(observation, dtype=np.float32)
            + self.hh @ self.hidden
            + self.bias
        )
        i, f, candidate, o = np.split(gates, 4)

        def sigmoid(x):
            return 1 / (1 + np.exp(-np.clip(x, -80, 80)))

        self.cell = sigmoid(f) * self.cell + sigmoid(i) * np.tanh(candidate)
        self.hidden = sigmoid(o) * np.tanh(self.cell)
        intermediate = self.fc1 @ self.hidden + self.b1
        # ELU with alpha=1, as in the released policy.
        activated = np.maximum(0.0, intermediate) + np.expm1(
            np.minimum(0.0, intermediate)
        )
        return self.fc2 @ activated + self.b2


class HumanoidController:
    """Apply a crossing velocity command to G1. The floating base moves by contact forces."""

    def __init__(self, model, data):
        self.data = data
        self.base = model.body("hum_pelvis").id
        base_joint = model.body_jntadr[self.base]
        self.base_q = model.jnt_qposadr[base_joint]
        self.base_v = model.jnt_dofadr[base_joint]
        joints = [model.joint(n).id for n in JOINTS]
        self.joint_indices = model.jnt_qposadr[joints]
        self.dof_indices = model.jnt_dofadr[joints]
        self.actuators = np.array([model.actuator(n).id for n in JOINTS])
        foot_bodies = [
            model.body(f"hum_{s}_ankle_roll_link").id for s in ("right", "left")
        ]
        self.feet = set(
            map(
                int,
                np.flatnonzero(
                    np.isin(model.geom_bodyid, foot_bodies) & (model.geom_contype != 0)
                ),
            )
        )
        self.policy = WalkingPolicy()
        self.step = float(model.opt.timestep)
        if not math.isclose(self.step, 0.002, abs_tol=1e-8):
            raise ValueError("The G1 controller expects a 0.002 s physics timestep")
        self.last_command = np.zeros(3)

    def place(self, xy, yaw):
        q = self.data.qpos
        self.initial = np.asarray(xy, dtype=float).copy()
        self.heading = yaw
        self.forward = np.array([math.cos(yaw), math.sin(yaw)])
        self.goal = self.initial + self.forward * 1.8
        q[self.base_q : self.base_q + 3] = (*self.initial, 0.793)
        q[self.base_q + 3 : self.base_q + 7] = (
            math.cos(yaw / 2),
            0.0,
            0.0,
            math.sin(yaw / 2),
        )
        q[self.joint_indices] = 0
        self.data.qvel[self.base_v : self.base_v + 6] = 0
        self.data.qvel[self.dof_indices] = 0
        self.policy.reset()
        self.counter = 0
        self.action = np.zeros(12)
        self.target = DEFAULT_ANGLES.copy()

    def actuate(self, t):
        q, v = self.data.qpos, self.data.qvel
        if self.counter and self.counter % 10 == 0:
            w, x, y, z = q[self.base_q + 3 : self.base_q + 7]
            gravity = [
                2 * (-z * x + w * y),
                -2 * (z * y + w * x),
                1 - 2 * (w * w + z * z),
            ]
            yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            c, s = math.cos(yaw), math.sin(yaw)
            destination = self.goal if t >= 2.0 else self.initial
            error = destination - q[self.base_q : self.base_q + 2]
            forward = np.clip(0.9 * (c * error[0] + s * error[1]), -0.12, 0.42)
            sideways = np.clip(0.8 * (-s * error[0] + c * error[1]), -0.18, 0.18)
            turn = np.clip(math.remainder(self.heading - yaw, 2 * math.pi), -0.45, 0.45)
            self.last_command[:] = (forward, sideways, turn)
            phase = 2 * math.pi * t / 0.8
            obs = np.r_[
                0.25 * v[self.base_v + 3 : self.base_v + 6],
                gravity,
                self.last_command * [2.0, 2.0, 0.25],
                q[self.joint_indices] - DEFAULT_ANGLES,
                0.05 * v[self.dof_indices],
                self.action,
                math.sin(phase),
                math.cos(phase),
            ]
            self.action = self.policy(obs)
            self.target = DEFAULT_ANGLES + 0.25 * self.action
        self.data.ctrl[self.actuators] = (self.target - q[self.joint_indices]) * KP - v[
            self.dof_indices
        ] * KD
        self.counter += 1

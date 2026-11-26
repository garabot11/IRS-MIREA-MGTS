"""Control a simulated floating-base Go2 by twelve joint torques."""

import math

import mujoco
import numpy as np


class Go2Controller:
    dt = 0.002
    policy_steps = 10
    max_speed = 0.60
    max_turn = 0.95

    def __init__(self, model, data, weights_path, *, prefix="", base_name="base"):
        self.model, self.data = model, data
        legs = ("FR", "FL", "RR", "RL")
        joints = ("hip", "thigh", "calf")
        self.actuators = [
            model.actuator(f"{prefix}{leg}_{joint}").id for leg in legs for joint in joints
        ]
        joint_ids = model.actuator_trnid[self.actuators, 0]
        self.pos_indices = model.jnt_qposadr[joint_ids]
        self.vel_indices = model.jnt_dofadr[joint_ids]
        self.base = model.body(prefix + base_name).id
        self.free_dof = int(model.jnt_dofadr[model.body_jntadr[self.base]])
        self.free_pos = int(model.jnt_qposadr[model.body_jntadr[self.base]])
        self.default = np.tile([0.0, 0.8, -1.5], 4)
        self.scale = np.tile([0.125, 0.25, 0.25], 4)
        with np.load(weights_path, allow_pickle=False) as weights:
            self.layers = [
                (weights[f"{i}.weight"], weights[f"{i}.bias"]) for i in (0, 2, 4, 6)
            ]
        self.command = np.zeros(3)
        self.actions = np.zeros(12)
        self.targets = self.default.copy()
        self.steps = self.still_steps = 0
        self.hold = None

    def initialize(self, xy, yaw):
        if self.steps:
            raise RuntimeError("Начальную позу можно задавать только до запуска физики")
        model, data = self.model, self.data
        model.opt.timestep = self.dt
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        self.place(xy, yaw)
        mujoco.mj_forward(model, data)

    def place(self, xy, yaw):
        """Place this body when initializing a world shared by multiple controllers."""
        data = self.data
        data.qpos[self.free_pos:self.free_pos + 3] = (*xy, 0.32)
        data.qpos[self.free_pos + 3:self.free_pos + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        data.qpos[self.pos_indices] = self.default

    def set_command(self, forward, lateral=0.0, turn=0.0):
        command = np.asarray([forward, lateral, turn], dtype=float)
        if not np.isfinite(command).all():
            raise ValueError("Неконечная команда движения")
        if lateral != 0 or forward < 0:
            raise ValueError("Команда выходит за проверенные направления движения Go2")
        command[0] = min(forward, self.max_speed)
        command[2] = np.clip(command[2], -self.max_turn, self.max_turn)
        self.command = command

    def actuate(self):
        """Write only this robot's twelve torques before a shared physics step."""
        data = self.data
        quiet = not np.any(self.command)
        if not quiet:
            self.still_steps = 0
            self.hold = None
        if self.steps % self.policy_steps == 0 and self.hold is None:
            rot = data.xmat[self.base].reshape(3, 3)
            obs = np.concatenate(
                (
                    0.25 * data.qvel[self.free_dof + 3:self.free_dof + 6],
                    -rot[2],
                    self.command,
                    data.qpos[self.pos_indices] - self.default,
                    0.05 * data.qvel[self.vel_indices],
                    self.actions,
                )
            )
            value = np.clip(obs, -100, 100).astype(np.float32)
            for weights, bias in self.layers[:-1]:
                value = weights @ value + bias
                value = np.where(value > 0, value, np.expm1(np.minimum(value, 0)))
            weights, bias = self.layers[-1]
            self.actions = np.clip(weights @ value + bias, -100, 100)
            self.targets = self.default + self.scale * self.actions
        self.still_steps += int(quiet)
        if (
            self.still_steps >= 200
            and self.hold is None
            and np.linalg.norm(data.qvel[self.free_dof:self.free_dof + 2]) < 0.12
        ):
            self.hold = data.qpos[self.pos_indices].copy()
        if self.hold is None:
            torques = (
                20 * (self.targets - data.qpos[self.pos_indices])
                - 0.5 * data.qvel[self.vel_indices]
            )
        else:
            torques = (
                55 * (self.hold - data.qpos[self.pos_indices])
                - 2 * data.qvel[self.vel_indices]
            )
        data.ctrl[self.actuators] = np.clip(torques, -23.5, 23.5)
        self.steps += 1

    def substep(self):
        self.actuate()
        mujoco.mj_step(self.model, self.data)

    def tick(self, command):
        self.set_command(*command)
        for _ in range(self.policy_steps):
            self.substep()

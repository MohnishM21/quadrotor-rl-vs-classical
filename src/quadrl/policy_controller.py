"""Wraps a trained policy as a quadbench Controller.

quadbench runs it exactly like PID, LQR, geometric and MPC: same plant, same
200 Hz rate, same scenarios, same metrics. The observation is built by the
same function the training environment uses (observation.py).
"""

import numpy as np
import torch

from quadbench.controllers.base import Controller
from quadbench.dynamics import P, Q, V, W
from quadbench.rotations import quat_to_rotmat

from .observation import ACT_DIM, action_to_thrust, build_obs, update_integral
from .ppo import load_policy


class PolicyController(Controller):
    def __init__(self, params, policy, name="RL"):
        super().__init__(params)
        self.policy = policy.to("cpu").eval()
        self.dtype = next(policy.parameters()).dtype
        self.name = name
        self.reset()

    @classmethod
    def from_checkpoint(cls, params, path, name="RL"):
        policy, _ = load_policy(path, "cpu")
        return cls(params, policy, name)

    def reset(self):
        self.integral = torch.zeros(1, 3, dtype=self.dtype)
        self.last_action = torch.zeros(1, ACT_DIM, dtype=self.dtype)

    @torch.no_grad()
    def compute(self, t, x, traj, dt):
        ref = traj(t)

        def row(a):
            return torch.as_tensor(np.asarray(a), dtype=self.dtype).reshape(1, -1)

        pos, vel, omega = row(x[P]), row(x[V]), row(x[W])
        R = torch.as_tensor(quat_to_rotmat(x[Q]), dtype=self.dtype).reshape(1, 3, 3)
        ref_pos = row(ref.pos)
        self.integral = update_integral(self.integral, pos, ref_pos, dt)
        obs = build_obs(pos, vel, R, omega, ref_pos, row(ref.vel), row(ref.acc),
                        self.integral, self.last_action, self.params.gravity)
        action = self.policy.act_deterministic(obs)
        self.last_action = action
        f = action_to_thrust(action, self.params)[0].double().numpy()
        # quadbench expects a wrench; its mixer maps this back to exactly f (f is within limits).
        return self.params.rotor_thrusts_to_wrench(f)

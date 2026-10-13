"""Batched quadrotor dynamics in PyTorch: thousands of vehicles stepped in parallel on the GPU.

This is a line-by-line port of quadbench.dynamics (same state layout, equations
and RK4 integrator), so a policy trained here can be evaluated in quadbench
against the classical controllers. tests/test_torch_dynamics.py checks that both
produce the same trajectory to 1e-9 in float64.

Every vehicle can have its own mass, inertia, motor time constant, drag and
wind, which is how domain randomization is applied.
"""

from dataclasses import dataclass

import torch

P, V, Q, W, F = slice(0, 3), slice(3, 6), slice(6, 10), slice(10, 13), slice(13, 17)
STATE_DIM = 17


def quat_multiply(p, q):
    pw, px, py, pz = p.unbind(-1)
    qw, qx, qy, qz = q.unbind(-1)
    return torch.stack([
        pw * qw - px * qx - py * qy - pz * qz,
        pw * qx + px * qw + py * qz - pz * qy,
        pw * qy - px * qz + py * qw + pz * qx,
        pw * qz + px * qy - py * qx + pz * qw,
    ], dim=-1)


def quat_to_rotmat(q):
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)


@dataclass
class BatchParams:
    """Per-vehicle physical parameters, each a tensor with leading dimension n."""
    mass: torch.Tensor          # (n,)
    inertia: torch.Tensor       # (n, 3) principal moments
    motor_tau: torch.Tensor     # (n,) 0 means instantaneous
    drag: torch.Tensor          # (n,) linear drag, N per m/s
    wind_const: torch.Tensor    # (n, 3) N
    gust_amp: torch.Tensor      # (n, 3) N
    gust_freq: torch.Tensor     # (n, 3) Hz
    gust_phase: torch.Tensor    # (n, 3) rad

    @staticmethod
    def nominal(n, quad_params, device="cpu", dtype=torch.float32):
        z3 = torch.zeros(n, 3, device=device, dtype=dtype)
        return BatchParams(
            mass=torch.full((n,), quad_params.mass, device=device, dtype=dtype),
            inertia=torch.tensor(quad_params.inertia, device=device, dtype=dtype).repeat(n, 1),
            motor_tau=torch.zeros(n, device=device, dtype=dtype),
            drag=torch.zeros(n, device=device, dtype=dtype),
            wind_const=z3.clone(), gust_amp=z3.clone(), gust_freq=z3.clone(), gust_phase=z3.clone(),
        )

    def wind(self, t):
        """t: (n,) seconds -> (n, 3) world-frame force."""
        return self.wind_const + self.gust_amp * torch.sin(2 * torch.pi * self.gust_freq * t[:, None] + self.gust_phase)


class BatchQuadrotor:
    def __init__(self, quad_params, device="cpu", dtype=torch.float32):
        self.qp = quad_params
        self.device, self.dtype = device, dtype
        self.allocation = torch.tensor(quad_params.allocation, device=device, dtype=dtype)
        self.f_max = quad_params.max_rotor_thrust      # same motors whatever the mass
        self.gravity = quad_params.gravity

    def derivative(self, t, x, f_cmd, bp: BatchParams):
        f = x[:, F]
        wrench = f @ self.allocation.T                  # (n, 4): T, tau_x, tau_y, tau_z
        thrust, tau = wrench[:, 0], wrench[:, 1:]

        R = quat_to_rotmat(x[:, Q])
        force = R[:, :, 2] * thrust[:, None] - bp.drag[:, None] * x[:, V] + bp.wind(t)

        omega = x[:, W]
        J_omega = bp.inertia * omega
        dx = torch.empty_like(x)
        dx[:, P] = x[:, V]
        dx[:, V] = force / bp.mass[:, None]
        dx[:, 5] -= self.gravity
        zeros = torch.zeros_like(omega[:, :1])
        dx[:, Q] = 0.5 * quat_multiply(x[:, Q], torch.cat([zeros, omega], dim=-1))
        dx[:, W] = (tau - torch.cross(omega, J_omega, dim=-1)) / bp.inertia
        lagged = bp.motor_tau > 0
        safe_tau = torch.where(lagged, bp.motor_tau, torch.ones_like(bp.motor_tau))
        dx[:, F] = torch.where(lagged[:, None], (f_cmd - f) / safe_tau[:, None], torch.zeros_like(f))
        return dx

    def step(self, t, x, f_cmd, dt, bp: BatchParams):
        """One RK4 step. t: (n,) time of each vehicle. f_cmd: (n, 4) rotor thrusts, held constant."""
        f_cmd = f_cmd.clamp(0.0, self.f_max)
        instant = (bp.motor_tau <= 0)[:, None]
        x = x.clone()
        x[:, F] = torch.where(instant, f_cmd, x[:, F])
        k1 = self.derivative(t, x, f_cmd, bp)
        k2 = self.derivative(t + dt / 2, x + dt / 2 * k1, f_cmd, bp)
        k3 = self.derivative(t + dt / 2, x + dt / 2 * k2, f_cmd, bp)
        k4 = self.derivative(t + dt, x + dt * k3, f_cmd, bp)
        x_next = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        x_next[:, Q] = x_next[:, Q] / x_next[:, Q].norm(dim=-1, keepdim=True)
        return x_next

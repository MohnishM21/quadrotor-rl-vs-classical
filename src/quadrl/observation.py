"""What the policy sees and how its output becomes rotor thrust.

The training environment and the quadbench evaluation wrapper both call these
functions, so the policy gets identical inputs in training and evaluation.

Observation (28 values):
    position error   (3)  p - p_ref, length clipped to 2 m
    velocity error   (3)  v - v_ref, length clipped to 8 m/s
    reference accel  (3)  a_ref / g, the same feedforward the classical controllers get
    rotation matrix  (9)  body -> world, row-major
    body rates       (3)  omega / 10
    error integral   (3)  conditional integral of position error, like the PID's
    last action      (4)

Action (4 values in [-1, 1]): one per rotor. 0 is the nominal hover thrust,
+1 is the motor's maximum, -1 is zero thrust.
"""

import torch

OBS_DIM = 28
ACT_DIM = 4
POS_CLIP = 2.0
VEL_CLIP = 8.0
INT_ZONE = 0.3      # integrate only within this distance of the reference (m), per axis
INT_CLIP = 0.3      # m s


def clip_norm(v, max_norm):
    norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return v * (max_norm / norm).clamp(max=1.0)


def update_integral(integral, pos, ref_pos, dt):
    err = pos - ref_pos
    active = (err.abs() < INT_ZONE).to(err.dtype)
    return (integral + active * err * dt).clamp(-INT_CLIP, INT_CLIP)


def build_obs(pos, vel, R, omega, ref_pos, ref_vel, ref_acc, integral, last_action, gravity):
    return torch.cat([
        clip_norm(pos - ref_pos, POS_CLIP),
        clip_norm(vel - ref_vel, VEL_CLIP),
        ref_acc / gravity,
        R.reshape(R.shape[0], 9),
        omega / 10.0,
        integral / INT_CLIP,
        last_action,
    ], dim=-1)


def action_to_thrust(action, quad_params):
    """Maps [-1, 1] to rotor thrust with 0 at the nominal hover thrust."""
    f_hover = quad_params.hover_thrust / 4.0
    f_max = quad_params.max_rotor_thrust
    a = action.clamp(-1.0, 1.0)
    return (f_hover + a * (f_max - f_hover)).clamp_min(0.0)

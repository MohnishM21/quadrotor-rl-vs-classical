"""The GPU simulator must match quadbench exactly, or the comparison is not fair."""

import numpy as np
import pytest
import torch

from quadbench.dynamics import PlantConfig, Quadrotor, make_state
from quadbench.params import QuadParams
from quadrl.torch_dynamics import BatchParams, BatchQuadrotor

qp = QuadParams()


def run_both(x0, commands, motor_tau=0.0, drag=0.0, wind=(0.0, 0.0, 0.0), mass_factor=1.0, dt=1e-3):
    true = qp.scaled(mass_factor=mass_factor)
    wind = np.array(wind)
    ref = Quadrotor(true, PlantConfig(motor_tau=motor_tau, linear_drag=drag, wind_force=lambda t: wind))
    batch = BatchQuadrotor(qp, "cpu", torch.float64)
    bp = BatchParams.nominal(1, qp, "cpu", torch.float64)
    bp.mass[:] = true.mass
    bp.motor_tau[:] = motor_tau
    bp.drag[:] = drag
    bp.wind_const[:] = torch.tensor(wind)

    x_ref, x_bat = x0.copy(), torch.tensor(x0)[None]
    t = 0.0
    for f in commands:
        x_ref = ref.step(t, x_ref, f, dt)
        x_bat = batch.step(torch.tensor([t], dtype=torch.float64), x_bat, torch.tensor(f)[None], dt, bp)
        t += dt
    return x_ref, x_bat[0].numpy()


@pytest.mark.parametrize("plant", [
    {},
    {"motor_tau": 0.03},
    {"drag": 0.02, "wind": (0.02, -0.01, 0.005)},
    {"mass_factor": 1.3, "motor_tau": 0.01},
])
def test_matches_quadbench(plant):
    rng = np.random.default_rng(1)
    x0 = make_state(position=(0.1, -0.2, 1.0), velocity=(0.5, 0.1, -0.2), euler=(0.4, -0.3, 1.0),
                    omega=(2.0, -1.0, 3.0), params=qp)
    commands = rng.uniform(0.0, qp.max_rotor_thrust, size=(500, 4))
    x_ref, x_bat = run_both(x0, commands, **plant)
    np.testing.assert_allclose(x_bat, x_ref, atol=1e-9, rtol=1e-9)


def test_vehicles_are_independent():
    batch = BatchQuadrotor(qp, "cpu", torch.float64)
    bp = BatchParams.nominal(3, qp, "cpu", torch.float64)
    bp.mass[1] = 2 * qp.mass                      # only vehicle 1 is heavy
    x = torch.tensor(make_state(position=(0, 0, 1), params=qp)).repeat(3, 1)
    f = torch.full((3, 4), qp.hover_thrust / 4, dtype=torch.float64)
    for _ in range(100):
        x = batch.step(torch.zeros(3, dtype=torch.float64), x, f, 1e-3, bp)
    assert abs(x[0, 2] - 1.0) < 1e-9 and abs(x[2, 2] - 1.0) < 1e-9
    assert x[1, 2] < 0.99

import math

import numpy as np
import pytest
import torch

from quadbench.dynamics import make_state
from quadbench.params import QuadParams
from quadbench.simulator import Scenario, simulate
from quadbench.trajectories import figure_eight
from quadrl.env import EnvConfig, VecQuadEnv, random_quaternions
from quadrl.observation import ACT_DIM, OBS_DIM, action_to_thrust
from quadrl.policy_controller import PolicyController
from quadrl.ppo import Policy

qp = QuadParams()


def make_env(n=64, dr=True, dtype=torch.float32):
    return VecQuadEnv(EnvConfig(n_envs=n, domain_randomization=dr), qp, "cpu", dtype, seed=0)


def test_shapes_and_finite():
    env = make_env()
    obs = env.reset()
    assert obs.shape == (64, OBS_DIM)
    for _ in range(50):
        obs, rew, term, trunc, info = env.step(torch.rand(64, ACT_DIM) * 2 - 1)
        assert torch.isfinite(obs).all() and torch.isfinite(rew).all()


def test_zero_action_is_hover_thrust():
    f = action_to_thrust(torch.zeros(1, 4), qp)
    np.testing.assert_allclose(f.numpy(), qp.hover_thrust / 4, rtol=1e-6)
    assert action_to_thrust(-torch.ones(1, 4), qp).max() == 0.0
    assert action_to_thrust(torch.ones(1, 4), qp).max().item() == pytest.approx(qp.max_rotor_thrust, rel=1e-6)


def test_reference_derivatives():
    env = make_env(n=256, dtype=torch.float64)
    env.reset()
    t, h = torch.full((256,), 0.37, dtype=torch.float64), 1e-4
    p1, _, _ = env.reference(t - h)
    p2, v, a = env.reference(t)
    p3, _, _ = env.reference(t + h)
    no_step = (env.t_step - 0.37).abs() > 2 * h
    np.testing.assert_allclose(((p3 - p1) / (2 * h))[no_step], v[no_step], atol=1e-5)
    np.testing.assert_allclose(((p3 - 2 * p2 + p1) / h ** 2)[no_step], a[no_step], atol=1e-3)


def test_reference_acceleration_is_capped():
    env = make_env(n=4096)
    env.reset()
    assert (env.amp * env.omega ** 2).max() <= env.cfg.max_ref_acc + 1e-3


def test_random_quaternions_cover_upside_down():
    q = random_quaternions(20000, "cpu", torch.float64)
    # z-component of the body z-axis, R[2,2] = 1 - 2(x^2 + y^2); uniform rotations give uniform R[2,2]
    r22 = 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)
    assert abs(r22.mean().item()) < 0.03
    assert (r22 < -0.9).float().mean().item() == pytest.approx(0.05, abs=0.01)


def test_episodes_end_by_time_limit_or_crash_and_reset():
    env = make_env(n=8, dr=False)
    env.reset()
    zero = torch.zeros(8, ACT_DIM)
    ended = torch.zeros(8, dtype=torch.long)
    for _ in range(env.max_steps):
        _, _, term, trunc, _ = env.step(zero)
        ended += (term | trunc).long()
        assert not (term & trunc).any()
    assert (ended >= 1).all()                      # nobody outlives the 5 s limit
    assert (env.steps < env.max_steps).all()       # and every env was reset


def test_training_env_and_quadbench_fly_the_policy_identically():
    """Same start, reference and policy: the training env and quadbench must give the same flight.

    This covers the dynamics port, the observation, the error integral and the
    action mapping together. If it fails, evaluation results would not reflect training.
    """
    torch.manual_seed(0)
    policy = Policy(OBS_DIM, ACT_DIM).double()
    with torch.no_grad():                                   # make the untrained policy react to its inputs
        policy.actor[-1].weight.mul_(30.0)
    traj = figure_eight(period=4.0)
    x0 = make_state(position=(0.2, -0.1, 1.1), euler=(0.2, -0.1, 0.0), params=qp)

    env = make_env(n=1, dr=False, dtype=torch.float64)
    env.reset()
    w = 2 * math.pi / 4.0
    env.amp[:] = torch.tensor([1.5, 0.75, 0.0], dtype=torch.float64)
    env.omega[:] = torch.tensor([w, 2 * w, 2 * w], dtype=torch.float64)
    env.phase[:] = 0.0
    env.step_offset[:] = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)   # the centre (0, 0, 1)
    env.t_step[:] = -1.0
    env.x[:] = torch.tensor(x0)[None]
    env.integral[:] = 0.0
    env.last_action[:] = 0.0
    obs = env._observe()
    env_positions = []
    for _ in range(200):
        obs, *_ = env.step(policy.act_deterministic(obs))
        env_positions.append(env.x[0, :3].clone().numpy())

    scenario = Scenario("parity", "", traj, 1.0, lambda p: x0.copy())
    result = simulate(PolicyController(qp, policy), scenario, qp)
    # quadbench stops the run if the vehicle reaches the ground; the training env has no ground.
    k = len(result.x) - 1
    assert k >= 100
    np.testing.assert_allclose(np.array(env_positions)[:k], result.x[1:, :3], atol=1e-8)
    # The policy actually moved the vehicle, so the comparison is not trivial.
    assert np.linalg.norm(result.x[-1, :3] - x0[:3]) > 0.05

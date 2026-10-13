"""PPO must learn a simple task through the same in-place buffer interface as the real env."""

import torch

from quadrl.ppo import PPOConfig, Policy, RunningMeanStd, train


class CopyTargetEnv:
    """Each step shows a fresh random target in [-1, 1]; the reward is highest for action == target.

    Like VecQuadEnv, it returns the same buffer objects every step and overwrites
    them in place. Consecutive observations are independent, so if PPO ever pairs
    an action with the wrong observation (an earlier version stored the observation
    after stepping) the task becomes unlearnable and this test fails.
    """

    def __init__(self, n=256):
        self.n, self.device = n, "cpu"
        self.steps = torch.zeros(n)
        self.obs_buf = torch.zeros(n, 1)
        self.final_obs_buf = torch.zeros(n, 1)
        self.rew_buf = torch.zeros(n)
        self.term_buf = torch.zeros(n, dtype=torch.bool)
        self.trunc_buf = torch.zeros(n, dtype=torch.bool)

    def reset(self):
        self.obs_buf.uniform_(-1, 1)
        return self.obs_buf

    def step(self, action):
        reward = 1.0 - (action.clamp(-1, 1) - self.obs_buf).abs().squeeze(-1)
        self.steps.add_(1)
        trunc = self.steps >= 10
        self.steps.mul_(~trunc)
        self.obs_buf.uniform_(-1, 1)
        self.final_obs_buf.copy_(self.obs_buf)
        self.rew_buf.copy_(reward)
        self.trunc_buf.copy_(trunc)
        return self.obs_buf, self.rew_buf, self.term_buf, self.trunc_buf, self.final_obs_buf

    def pop_stats(self):
        return {"episodes": 0, "ep_return": float("nan"), "ep_mean_err": float("nan"), "crash_rate": float("nan")}


def test_ppo_learns_to_copy_the_observation():
    env = CopyTargetEnv()
    cfg = PPOConfig(total_steps=256 * 16 * 40, rollout_steps=16, hidden=32, init_log_std=-0.5, seed=0)
    policy, _ = train(env, 1, 1, cfg, log=lambda *_: None)
    obs = torch.linspace(-0.9, 0.9, 19)[:, None]
    action = policy.act_deterministic(obs)
    assert (action - obs).abs().max() < 0.15


def test_running_mean_std_matches_numpy():
    torch.manual_seed(0)
    data = torch.randn(1000, 3) * torch.tensor([1.0, 5.0, 0.1]) + torch.tensor([2.0, -1.0, 0.0])
    rms = RunningMeanStd(3)
    for chunk in data.split(137):
        rms.update(chunk)
    torch.testing.assert_close(rms.mean, data.mean(0), atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(rms.var, data.var(0, unbiased=False), atol=1e-2, rtol=1e-2)


def test_policy_output_is_bounded():
    policy = Policy(28, 4)
    out = policy.act_deterministic(torch.randn(100, 28) * 100)
    assert out.abs().max() <= 1.0

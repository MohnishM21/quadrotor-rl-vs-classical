"""Proximal Policy Optimization (Schulman et al., 2017), written from scratch for a GPU-vectorized env.

Implementation choices, all standard in continuous-control PPO:
- separate actor and critic MLPs; Gaussian policy with a learned, state-independent std
- asymmetric actor-critic: the critic may also see privileged training-only information
- observation normalization with running mean and variance (saved with the policy)
- reward scaling by the running std of the discounted return
- generalized advantage estimation (GAE)
- time-limit truncation is bootstrapped with V(final observation), not treated as a crash
- clipped surrogate objective, advantage normalization per minibatch, gradient clipping,
  linear learning-rate decay
"""

import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class PPOConfig:
    total_steps: int = 100_000_000
    rollout_steps: int = 32            # per env, per iteration
    epochs: int = 5
    minibatches: int = 4
    gamma: float = 0.995               # 200 Hz control: horizon of about 1 s
    gae_lambda: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    max_grad_norm: float = 1.0
    desired_kl: float = 0.01           # adaptive learning rate (as in rsl_rl); None for linear decay
    hidden: int = 256
    init_log_std: float = -1.6
    seed: int = 0


class RunningMeanStd(nn.Module):
    """Parallel (Chan et al.) running mean and variance. A module so it saves with state_dict."""

    def __init__(self, shape, eps=1e-4):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(eps))

    @torch.no_grad()
    def update(self, x):
        batch_mean, batch_var, n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + n
        self.mean += delta * n / total
        self.var = (self.var * self.count + batch_var * n + delta ** 2 * self.count * n / total) / total
        self.count = total

    def normalize(self, x, clip=10.0):
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-clip, clip)


def mlp(n_in, n_out, hidden, out_gain):
    layers = [nn.Linear(n_in, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, n_out)]
    for layer in layers:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, np.sqrt(2))
            nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(layers[-1].weight, out_gain)
    return nn.Sequential(*layers)


class Policy(nn.Module):
    """Everything needed at deployment: observation normalizer + actor."""

    def __init__(self, obs_dim, act_dim, hidden=256, init_log_std=-1.6):
        super().__init__()
        self.obs_rms = RunningMeanStd(obs_dim)
        self.actor = mlp(obs_dim, act_dim, hidden, out_gain=0.01)
        self.log_std = nn.Parameter(torch.full((act_dim,), init_log_std))

    def distribution(self, obs):
        mean = self.actor(self.obs_rms.normalize(obs))
        return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))

    @torch.no_grad()
    def act_deterministic(self, obs):
        return self.actor(self.obs_rms.normalize(obs)).clamp(-1.0, 1.0)


class Critic(nn.Module):
    def __init__(self, obs_dim, hidden=256):
        super().__init__()
        self.net = mlp(obs_dim, 1, hidden, out_gain=1.0)

    def forward(self, normalized_obs):
        return self.net(normalized_obs).squeeze(-1)


def train(env, obs_dim, act_dim, cfg: PPOConfig, log=print, on_iteration=None):
    device = env.device
    torch.manual_seed(cfg.seed)
    policy = Policy(obs_dim, act_dim, cfg.hidden, cfg.init_log_std).to(device)
    # Asymmetric actor-critic: if the env offers privileged information (true physics
    # parameters, exact errors), only the critic sees it. The deployed policy does not.
    priv_dim = env.priv_buf.shape[1] if hasattr(env, "priv_buf") else 0
    critic_dim = obs_dim + priv_dim
    critic = Critic(critic_dim, cfg.hidden).to(device)
    critic_rms = RunningMeanStd(critic_dim).to(device)

    def critic_input(o, priv):
        return torch.cat([o, priv], dim=-1) if priv_dim else o
    params = list(policy.actor.parameters()) + [policy.log_std] + list(critic.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.lr, eps=1e-5)
    return_rms = RunningMeanStd(()).to(device)

    n, T = env.n, cfg.rollout_steps
    batch = n * T
    iterations = cfg.total_steps // batch
    buf_obs = torch.zeros(T, n, obs_dim, device=device)
    buf_cobs = torch.zeros(T, n, critic_dim, device=device)
    buf_act = torch.zeros(T, n, act_dim, device=device)
    buf_logp = torch.zeros(T, n, device=device)
    buf_rew = torch.zeros(T, n, device=device)
    buf_done = torch.zeros(T, n, device=device)
    buf_val = torch.zeros(T, n, device=device)
    discounted = torch.zeros(n, device=device)

    obs = env.reset()
    history = []
    tic = time.time()

    lr = cfg.lr
    for it in range(iterations):
        if cfg.desired_kl is None:
            lr = cfg.lr * (1.0 - it / iterations)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # ---------------------------------------------------------- rollout
        # The observation normalizer is frozen for the whole iteration. Updating it
        # mid-rollout made the stored log-probabilities inconsistent with the update
        # step, so PPO saw a large policy change that the gradient never made.
        if it == 0:
            policy.obs_rms.update(obs)
            critic_rms.update(critic_input(obs, env.priv_buf if priv_dim else None))
        for t in range(T):
            cobs = critic_input(obs, env.priv_buf if priv_dim else None)
            with torch.no_grad():
                dist = policy.distribution(obs)
                action = dist.sample()
                value = critic(critic_rms.normalize(cobs))
            # env.step overwrites its buffers in place (CUDA graph), so store inputs first.
            buf_obs[t] = obs
            buf_cobs[t] = cobs
            next_obs, reward, terminated, truncated, final_obs = env.step(action)
            done = terminated | truncated

            # Running std of the discounted return, used to scale rewards.
            discounted = discounted * cfg.gamma + reward
            return_rms.update(discounted)
            discounted = discounted * ~done
            scaled = reward / torch.sqrt(return_rms.var + 1e-8)

            # A time-limit cut is not a failure: bootstrap with the value of the last observation.
            # Computed for every env (cheap) to avoid a GPU->CPU sync on "any truncated?".
            with torch.no_grad():
                final_cobs = critic_input(final_obs, env.final_priv_buf if priv_dim else None)
                final_v = critic(critic_rms.normalize(final_cobs))
            scaled = scaled + cfg.gamma * final_v * truncated.float()

            buf_act[t], buf_logp[t] = action, dist.log_prob(action).sum(-1)
            buf_rew[t], buf_val[t] = scaled, value
            buf_done[t] = done.float()
            obs = next_obs

        # ---------------------------------------------------------- GAE
        with torch.no_grad():
            next_value = critic(critic_rms.normalize(critic_input(obs, env.priv_buf if priv_dim else None)))
            adv = torch.zeros_like(buf_rew)
            last = torch.zeros(n, device=device)
            for t in reversed(range(T)):
                v_next = next_value if t == T - 1 else buf_val[t + 1]
                nonterminal = 1.0 - buf_done[t]
                delta = buf_rew[t] + cfg.gamma * v_next * nonterminal - buf_val[t]
                last = delta + cfg.gamma * cfg.gae_lambda * nonterminal * last
                adv[t] = last
            returns = adv + buf_val

        # ---------------------------------------------------------- update
        b_obs, b_act = buf_obs.reshape(batch, -1), buf_act.reshape(batch, -1)
        b_cobs = buf_cobs.reshape(batch, -1)
        b_logp, b_adv, b_ret = buf_logp.reshape(-1), adv.reshape(-1), returns.reshape(-1)
        mb = batch // cfg.minibatches
        clip_fracs, approx_kl = [], torch.tensor(0.0)
        for _ in range(cfg.epochs):
            perm = torch.randperm(batch, device=device)
            for start in range(0, batch, mb):
                i = perm[start:start + mb]
                dist = policy.distribution(b_obs[i])
                logp = dist.log_prob(b_act[i]).sum(-1)
                ratio = (logp - b_logp[i]).exp()
                a = b_adv[i]
                a = (a - a.mean()) / (a.std() + 1e-8)
                pg_loss = torch.max(-a * ratio, -a * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)).mean()
                v_loss = 0.5 * (critic(critic_rms.normalize(b_cobs[i])) - b_ret[i]).pow(2).mean()
                entropy = dist.entropy().sum(-1).mean()
                loss = pg_loss + cfg.value_coef * v_loss - cfg.entropy_coef * entropy
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                optimizer.step()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - (logp - b_logp[i])).mean()
                    clip_fracs.append(((ratio - 1).abs() > cfg.clip).float().mean().item())
                if cfg.desired_kl is not None:
                    # Shrink the step when the policy moved too far, grow it when it barely moved.
                    kl = approx_kl.item()
                    if kl > 2.0 * cfg.desired_kl:
                        lr = max(1e-5, lr / 1.5)
                    elif kl < 0.5 * cfg.desired_kl:
                        lr = min(1e-2, lr * 1.5)
                    for group in optimizer.param_groups:
                        group["lr"] = lr

        policy.obs_rms.update(b_obs)
        critic_rms.update(b_cobs)

        # ---------------------------------------------------------- logging
        steps = (it + 1) * batch
        stats = env.pop_stats()
        row = {"iteration": it + 1, "steps": steps, "seconds": time.time() - tic,
               "episode_return": stats["ep_return"], "episode_mean_error_m": stats["ep_mean_err"],
               "crash_rate": stats["crash_rate"], "policy_std": policy.log_std.exp().mean().item(),
               "approx_kl": approx_kl.item(), "clip_fraction": float(np.mean(clip_fracs)), "value_loss": v_loss.item(),
               "lr": lr}
        history.append(row)
        if (it + 1) % 10 == 0 or it == 0:
            log(f"it {it + 1:4d}/{iterations}  steps {steps / 1e6:6.1f}M  {row['seconds']:6.0f}s  "
                f"return {row['episode_return']:8.1f}  err {row['episode_mean_error_m']:.3f} m  "
                f"crash {row['crash_rate']:.3f}  std {row['policy_std']:.3f}  kl {row['approx_kl']:.4f}  lr {lr:.1e}")
        if on_iteration is not None:
            on_iteration(it, policy, history)

    return policy, history


def save_policy(path, policy, env_cfg, ppo_cfg, history):
    torch.save({"policy": policy.state_dict(), "obs_dim": policy.obs_rms.mean.shape[0],
                "act_dim": policy.log_std.shape[0], "hidden": ppo_cfg.hidden,
                "env_cfg": asdict(env_cfg), "ppo_cfg": asdict(ppo_cfg), "history": history}, path)


def load_policy(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    policy = Policy(ckpt["obs_dim"], ckpt["act_dim"], ckpt["hidden"])
    policy.load_state_dict(ckpt["policy"])
    return policy.to(device).eval(), ckpt

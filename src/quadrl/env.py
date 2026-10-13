"""Vectorized training environment: n quadrotors, each tracking its own random reference.

Every episode samples:
- a reference: hover at a point, a sudden step, or a smooth multi-axis sinusoid
  (figure-8-like paths included);
- a start state: up to 1 m off the reference, tilted up to 40 degrees, or
  (30% of episodes) at a uniformly random attitude, including upside down;
- with domain randomization: mass, inertia, motor lag, drag, wind and sensor noise.

The policy only sees errors relative to the reference, so absolute position
does not matter and there is no ground in training.

Speed: one environment step is about 2,000 small GPU kernels, and on Windows
each launch costs about 30 us. The whole step is therefore written without
data-dependent Python control flow (finished episodes are reset by masking,
not indexing) so it can be recorded once as a CUDA graph and replayed as a
single launch.
"""

import math
from dataclasses import dataclass

import torch

from .observation import ACT_DIM, OBS_DIM, action_to_thrust, build_obs, update_integral

# Privileged critic input (training only, never given to the policy): the randomized
# physics and the exact, unclipped state errors. An asymmetric actor-critic
# (Pinto et al., 2018) learns better values when the dynamics vary per episode.
PRIV_DIM = 20
from .torch_dynamics import F, P, Q, STATE_DIM, V, W, BatchParams, BatchQuadrotor, quat_to_rotmat


@dataclass
class EnvConfig:
    n_envs: int = 4096
    ctrl_dt: float = 0.005            # 200 Hz, same as the quadbench controllers
    sim_dt: float = 0.001             # 1 kHz RK4, same as quadbench
    episode_s: float = 5.0
    domain_randomization: bool = True
    # reference mix
    p_hover: float = 0.35
    p_step: float = 0.25              # the rest are sinusoids
    max_ref_acc: float = 14.0         # m/s^2 per axis; the fast figure-8 needs 13.2
    # start states
    max_start_offset: float = 1.0
    p_random_attitude: float = 0.3
    max_start_tilt_deg: float = 40.0
    # termination
    max_error: float = 4.0
    # reward weights. The reward stays positive during controlled flight, so
    # surviving always pays more than crashing (a negative per-step reward taught
    # the first trial to crash on purpose).
    w_alive: float = 1.0
    w_pos: float = 1.0
    pos_scale: float = 0.5            # m; exp(-err / pos_scale)
    w_precision: float = 0.5
    precision_scale: float = 0.1      # m; width of the bonus for being close
    w_fine: float = 0.5
    fine_scale: float = 0.025         # m; a second, narrower bonus to close the last centimetres
    w_vel: float = 0.05
    w_rate: float = 0.01
    w_yaw: float = 0.3
    w_smooth: float = 0.1


def random_unit_vectors(n, device, dtype):
    v = torch.randn(n, 3, device=device, dtype=dtype)
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def random_quaternions(n, device, dtype):
    """Uniform over SO(3) (Shoemake's method)."""
    u1, u2, u3 = torch.rand(3, n, device=device, dtype=dtype)
    a, b = torch.sqrt(1 - u1), torch.sqrt(u1)
    q = torch.stack([a * torch.sin(2 * math.pi * u2), a * torch.cos(2 * math.pi * u2),
                     b * torch.sin(2 * math.pi * u3), b * torch.cos(2 * math.pi * u3)], dim=-1)
    return torch.where(q[:, :1] < 0, -q, q)


def axis_angle_quaternions(axis, angle):
    half = angle[:, None] / 2
    return torch.cat([torch.cos(half), axis * torch.sin(half)], dim=-1)


def uniform(lo, hi, shape, device, dtype):
    return lo + (hi - lo) * torch.rand(shape, device=device, dtype=dtype)


class VecQuadEnv:
    REF_KEYS = ("amp", "omega", "phase", "step_offset", "t_step")
    DYN_KEYS = ("mass", "inertia", "motor_tau", "drag", "wind_const", "gust_amp", "gust_freq", "gust_phase")

    def __init__(self, cfg: EnvConfig, quad_params, device="cuda", dtype=torch.float32, seed=0,
                 use_cuda_graph=None):
        self.cfg, self.qp = cfg, quad_params
        self.device, self.dtype = device, dtype
        self.n = cfg.n_envs
        self.substeps = round(cfg.ctrl_dt / cfg.sim_dt)
        self.max_steps = round(cfg.episode_s / cfg.ctrl_dt)
        self.sim = BatchQuadrotor(quad_params, device, dtype)
        self.use_graph = (str(device).startswith("cuda")) if use_cuda_graph is None else use_cuda_graph
        self._graph = None
        torch.manual_seed(seed)

        n, kw = self.n, dict(device=device, dtype=dtype)
        self.x = torch.zeros(n, STATE_DIM, **kw)
        self.t = torch.zeros(n, **kw)
        self.steps = torch.zeros(n, **kw)
        self.integral = torch.zeros(n, 3, **kw)
        self.last_action = torch.zeros(n, ACT_DIM, **kw)
        self.bp = BatchParams.nominal(n, quad_params, device, dtype)
        self.noise_scale = torch.zeros(n, **kw)
        # reference: pos(t) = amp * sin(omega t + phase) + step_offset * [t >= t_step]
        self.amp = torch.zeros(n, 3, **kw)
        self.omega = torch.zeros(n, 3, **kw)
        self.phase = torch.zeros(n, 3, **kw)
        self.step_offset = torch.zeros(n, 3, **kw)
        self.t_step = torch.zeros(n, **kw)
        # episode statistics: [sum of returns, sum of mean errors, crashes, episodes]
        self.ep_return = torch.zeros(n, **kw)
        self.ep_err_sum = torch.zeros(n, **kw)
        self.stats = torch.zeros(4, **kw)
        # static input/output buffers (the CUDA graph reads and writes these addresses)
        self.action_buf = torch.zeros(n, ACT_DIM, **kw)
        self.obs_buf = torch.zeros(n, OBS_DIM, **kw)
        self.final_obs_buf = torch.zeros(n, OBS_DIM, **kw)
        self.priv_buf = torch.zeros(n, PRIV_DIM, **kw)
        self.final_priv_buf = torch.zeros(n, PRIV_DIM, **kw)
        self.rew_buf = torch.zeros(n, **kw)
        self.term_buf = torch.zeros(n, dtype=torch.bool, device=device)
        self.trunc_buf = torch.zeros(n, dtype=torch.bool, device=device)
        # constants created up front: host-to-device copies are not allowed inside a CUDA graph
        self._yz_mask = torch.tensor([False, True, True], device=device)
        self._inertia0 = torch.tensor(quad_params.inertia, **kw)
        self._eye3 = torch.eye(3, **kw)

    # ------------------------------------------------------------------ reference
    def reference(self, t):
        arg = self.omega * t[:, None] + self.phase
        stepped = (t >= self.t_step).to(self.dtype)[:, None]
        pos = self.amp * torch.sin(arg) + self.step_offset * stepped
        vel = self.amp * self.omega * torch.cos(arg)
        acc = -self.amp * self.omega ** 2 * torch.sin(arg)
        return pos, vel, acc

    # ------------------------------------------------------------------ sampling
    def _sample(self):
        """Fresh episode parameters and start states for all n envs (used where an episode ends)."""
        n, kw, cfg = self.n, dict(device=self.device, dtype=self.dtype), self.cfg
        fresh = {}

        kind = torch.rand(n, **kw)
        step = (kind >= cfg.p_hover) & (kind < cfg.p_hover + cfg.p_step)
        sine = kind >= cfg.p_hover + cfg.p_step
        amp = torch.stack([uniform(0, 1.5, n, **kw), uniform(0, 1.5, n, **kw), uniform(0, 0.4, n, **kw)], -1)
        omega = 2 * math.pi / uniform(2.5, 8.0, (n, 3), **kw)
        figure8 = (torch.rand(n, **kw) < 0.5)[:, None]            # y and z at twice the x frequency
        omega = torch.where(figure8 & self._yz_mask, 2 * omega[:, :1], omega)
        omega = torch.minimum(omega, torch.sqrt(cfg.max_ref_acc / amp.clamp_min(1e-6)))
        fresh["amp"] = amp * sine[:, None]
        fresh["omega"] = omega
        fresh["phase"] = uniform(0, 2 * math.pi, (n, 3), **kw)
        offset = random_unit_vectors(n, self.device, self.dtype) * uniform(0.3, 2.0, (n, 1), **kw)
        fresh["step_offset"] = offset * step[:, None]
        fresh["t_step"] = uniform(0.5, 3.5, n, **kw)

        if cfg.domain_randomization:
            qp = self.qp
            fresh["mass"] = qp.mass * uniform(0.75, 1.35, n, **kw)
            fresh["inertia"] = self._inertia0 * uniform(0.8, 1.25, (n, 3), **kw)
            fresh["motor_tau"] = uniform(0.005, 0.04, n, **kw) * (torch.rand(n, **kw) < 0.5)
            fresh["drag"] = uniform(0.0, 0.02, n, **kw)
            fresh["wind_const"] = torch.stack([uniform(-0.03, 0.03, n, **kw), uniform(-0.03, 0.03, n, **kw),
                                               uniform(-0.01, 0.01, n, **kw)], -1)
            fresh["gust_amp"] = uniform(0.0, 0.02, (n, 3), **kw)
            fresh["gust_freq"] = uniform(0.1, 1.0, (n, 3), **kw)
            fresh["gust_phase"] = uniform(0, 2 * math.pi, (n, 3), **kw)
            fresh["noise_scale"] = torch.rand(n, **kw)

        # start state, relative to the fresh reference at t = 0
        arg = fresh["phase"]
        ref_pos = fresh["amp"] * torch.sin(arg) + fresh["step_offset"] * (fresh["t_step"] <= 0)[:, None]
        ref_vel = fresh["amp"] * fresh["omega"] * torch.cos(arg)
        x = torch.zeros(n, STATE_DIM, **kw)
        dist = uniform(0, cfg.max_start_offset, (n, 1), **kw)
        x[:, P] = ref_pos + random_unit_vectors(n, self.device, self.dtype) * dist
        x[:, V] = ref_vel + 0.3 * torch.randn(n, 3, **kw)
        tilt = uniform(0, math.radians(cfg.max_start_tilt_deg), n, **kw)
        q_tilt = axis_angle_quaternions(random_unit_vectors(n, self.device, self.dtype), tilt)
        any_att = (torch.rand(n, **kw) < cfg.p_random_attitude)[:, None]
        x[:, Q] = torch.where(any_att, random_quaternions(n, self.device, self.dtype), q_tilt)
        x[:, W] = 2.0 * torch.randn(n, 3, **kw)
        x[:, F] = self.qp.hover_thrust / 4.0
        fresh["x"] = x
        return fresh

    def _targets(self):
        targets = {k: getattr(self, k) for k in self.REF_KEYS}
        targets.update({k: getattr(self.bp, k) for k in self.DYN_KEYS})
        targets["noise_scale"] = self.noise_scale
        targets["x"] = self.x
        return targets

    def _apply(self, fresh, mask):
        """Overwrite the state of envs where `mask` is true, in place."""
        targets = self._targets()
        for key, value in fresh.items():
            target = targets[key]
            m = mask.view(-1, *([1] * (target.dim() - 1)))
            target.copy_(torch.where(m, value, target))
        keep = (~mask).to(self.dtype)
        for buf in (self.t, self.steps, self.ep_return, self.ep_err_sum):
            buf.mul_(keep)
        self.integral.mul_(keep[:, None])
        self.last_action.mul_(keep[:, None])

    def reset(self):
        self._apply(self._sample(), torch.ones(self.n, dtype=torch.bool, device=self.device))
        self.stats.zero_()
        self.obs_buf.copy_(self._observe())
        self.priv_buf.copy_(self._privileged())
        return self.obs_buf

    def _privileged(self):
        qp, bp = self.qp, self.bp
        ref_pos, ref_vel, _ = self.reference(self.t)
        return torch.cat([
            (bp.mass / qp.mass - 1.0)[:, None],
            bp.inertia / self._inertia0 - 1.0,
            (bp.motor_tau / 0.04)[:, None],
            (bp.drag / 0.02)[:, None],
            bp.wind_const / 0.03,
            bp.gust_amp / 0.02,
            self.noise_scale[:, None],
            ((self.x[:, P] - ref_pos) / 2.0).clamp(-5, 5),
            ((self.x[:, V] - ref_vel) / 5.0).clamp(-5, 5),
            (self.steps / self.max_steps)[:, None],
        ], dim=-1)

    # ------------------------------------------------------------------ observation and reward
    def _measure(self):
        """Noisy copy of position, velocity, attitude and rates (sensor noise as in quadbench)."""
        x = self.x
        R = quat_to_rotmat(x[:, Q])
        if not self.cfg.domain_randomization:
            return x[:, P], x[:, V], R, x[:, W]
        s = self.noise_scale[:, None]
        pos = x[:, P] + s * 0.005 * torch.randn_like(x[:, P])
        vel = x[:, V] + s * 0.02 * torch.randn_like(x[:, V])
        d = s * math.radians(0.5) * torch.randn_like(x[:, P])
        zero = torch.zeros_like(d[:, 0])
        skew = torch.stack([zero, -d[:, 2], d[:, 1], d[:, 2], zero, -d[:, 0], -d[:, 1], d[:, 0], zero],
                           -1).reshape(-1, 3, 3)
        R = R @ (self._eye3 + skew)
        omega = x[:, W] + s * 0.02 * torch.randn_like(x[:, W])
        return pos, vel, R, omega

    def _observe(self):
        """Measure, update the error integral in place, and build observations for all envs."""
        pos, vel, R, omega = self._measure()
        ref_pos, ref_vel, ref_acc = self.reference(self.t)
        self.integral.copy_(update_integral(self.integral, pos, ref_pos, self.cfg.ctrl_dt))
        return build_obs(pos, vel, R, omega, ref_pos, ref_vel, ref_acc, self.integral,
                         self.last_action, self.qp.gravity)

    def _reward(self, action):
        cfg = self.cfg
        ref_pos, ref_vel, _ = self.reference(self.t)
        err = (self.x[:, P] - ref_pos).norm(dim=-1)
        vel_err = (self.x[:, V] - ref_vel).norm(dim=-1).clamp(max=5.0)
        rate = self.x[:, W].norm(dim=-1).clamp(max=30.0)
        R = quat_to_rotmat(self.x[:, Q])
        yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0]).abs()
        smooth = (action - self.last_action).abs().mean(dim=-1)
        reward = (cfg.w_alive
                  + cfg.w_pos * torch.exp(-err / cfg.pos_scale)
                  + cfg.w_precision * torch.exp(-(err / cfg.precision_scale) ** 2)
                  + cfg.w_fine * torch.exp(-(err / cfg.fine_scale) ** 2)
                  - cfg.w_vel * vel_err
                  - cfg.w_rate * rate
                  - cfg.w_yaw * yaw
                  - cfg.w_smooth * smooth)
        return reward, err

    # ------------------------------------------------------------------ step
    def _step_impl(self):
        action = self.action_buf.clamp(-1.0, 1.0)
        f_cmd = action_to_thrust(action, self.qp)
        x, t = self.x, self.t
        for _ in range(self.substeps):
            x = self.sim.step(t, x, f_cmd, self.cfg.sim_dt, self.bp)
            t = t + self.cfg.sim_dt
        finite = torch.isfinite(x).all(dim=-1)
        self.x.copy_(torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))
        self.t.copy_(t)
        self.steps.add_(1.0)

        reward, err = self._reward(action)
        err = torch.where(finite, err, torch.full_like(err, 10.0 * self.cfg.max_error))
        reward = torch.where(finite, reward, torch.zeros_like(reward))
        self.last_action.copy_(action)
        terminated = (err > self.cfg.max_error) | ~finite
        truncated = (self.steps >= self.max_steps) & ~terminated
        done = terminated | truncated

        self.ep_return.add_(reward)
        self.ep_err_sum.add_(err.clamp(max=self.cfg.max_error))
        d = done.to(self.dtype)
        self.stats.add_(torch.stack([(self.ep_return * d).sum(), (self.ep_err_sum / self.steps * d).sum(),
                                     terminated.to(self.dtype).sum(), d.sum()]))

        obs_before = self._observe()
        integral_before = self.integral.clone()
        self.final_obs_buf.copy_(obs_before)
        self.final_priv_buf.copy_(self._privileged())

        self._apply(self._sample(), done)
        obs_after = self._observe()                      # only meaningful for the envs just reset
        self.integral.copy_(torch.where(done[:, None], self.integral, integral_before))
        self.obs_buf.copy_(torch.where(done[:, None], obs_after, obs_before))
        self.priv_buf.copy_(torch.where(done[:, None], self._privileged(), self.final_priv_buf))
        self.rew_buf.copy_(reward)
        self.term_buf.copy_(terminated)
        self.trunc_buf.copy_(truncated)

    def _capture(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):                           # warm-up runs, required before capture
                self._step_impl()
        torch.cuda.current_stream().wait_stream(stream)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._step_impl()

    def step(self, action):
        """Returns (obs, reward, terminated, truncated, final_obs). Buffers are reused: copy if kept.

        final_obs is the observation before any reset, used to bootstrap truncated episodes.
        """
        self.action_buf.copy_(action)
        if self.use_graph:
            if self._graph is None:
                self._capture()
            self._graph.replay()
        else:
            self._step_impl()
        return self.obs_buf, self.rew_buf, self.term_buf, self.trunc_buf, self.final_obs_buf

    def pop_stats(self):
        """Mean return, mean error and crash rate of episodes finished since the last call."""
        s = self.stats.tolist()
        self.stats.zero_()
        if s[3] == 0:
            return {"episodes": 0, "ep_return": float("nan"), "ep_mean_err": float("nan"), "crash_rate": float("nan")}
        return {"episodes": int(s[3]), "ep_return": s[0] / s[3], "ep_mean_err": s[1] / s[3], "crash_rate": s[2] / s[3]}

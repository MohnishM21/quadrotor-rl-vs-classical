"""Train a PPO quadrotor policy.

    python scripts/train.py --name ppo_dr                 # with domain randomization (default)
    python scripts/train.py --name ppo_nominal --no-dr    # nominal model only, for the ablation
"""

import argparse
import csv
from pathlib import Path

import torch

from quadbench.params import QuadParams
from quadrl.env import EnvConfig, VecQuadEnv
from quadrl.observation import ACT_DIM, OBS_DIM
from quadrl.ppo import PPOConfig, save_policy, train

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="ppo_dr")
    ap.add_argument("--steps", type=float, default=200e6)
    ap.add_argument("--envs", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-dr", action="store_true", help="train on the nominal model only")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_cfg = EnvConfig(n_envs=args.envs, domain_randomization=not args.no_dr)
    ppo_cfg = PPOConfig(total_steps=int(args.steps), seed=args.seed)
    env = VecQuadEnv(env_cfg, QuadParams(), device, seed=args.seed)

    models, logs = ROOT / "models", ROOT / "results" / "training"
    models.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    checkpoint = models / f"{args.name}.pt"

    def on_iteration(it, policy, history):
        if (it + 1) % 50 == 0:
            save_policy(checkpoint, policy, env_cfg, ppo_cfg, history)

    print(f"training {args.name} on {device}: {env_cfg.n_envs} envs, {ppo_cfg.total_steps / 1e6:.0f}M steps, "
          f"domain randomization {'on' if env_cfg.domain_randomization else 'off'}")
    policy, history = train(env, OBS_DIM, ACT_DIM, ppo_cfg, on_iteration=on_iteration)
    save_policy(checkpoint, policy, env_cfg, ppo_cfg, history)

    with open(logs / f"{args.name}.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    print(f"saved {checkpoint}")


if __name__ == "__main__":
    main()

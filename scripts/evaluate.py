"""Evaluate trained policies against the classical controllers, entirely inside quadbench.

    python scripts/evaluate.py                        # all scenarios + robustness sweeps + figures
    python scripts/evaluate.py --plots-only           # redraw from saved results

Writes results/benchmark.{md,csv}, results/sweeps.csv and results/figures/.
"""

import argparse
import csv
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from quadbench.controllers import ALL_CONTROLLERS
from quadbench.dynamics import PlantConfig
from quadbench.metrics import summarize
from quadbench.params import QuadParams
from quadbench.scenarios import SCENARIOS
from quadbench.simulator import Scenario, simulate
from quadrl.policy_controller import PolicyController

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
POLICIES = {"RL (DR)": ROOT / "models" / "ppo_dr.pt", "RL (nominal)": ROOT / "models" / "ppo_nominal.pt"}
CLASSICAL = ["PID", "LQR", "Geometric", "MPC"]
SETTLING = {"hover_recovery", "step", "long_step", "flip_recovery"}
MASS_FACTORS = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
MOTOR_TAUS = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]


def available_controllers():
    names = list(CLASSICAL)
    names += [n for n, p in POLICIES.items() if p.exists()]
    return names


def make_controller(name, params):
    if name in POLICIES:
        return PolicyController.from_checkpoint(params, POLICIES[name], name=name)
    return ALL_CONTROLLERS[name](params)


def sweep_scenario(kind, value):
    base = SCENARIOS["figure8"]
    if kind == "mass":
        return Scenario(f"mass_{value}", "", base.trajectory, base.duration, base.initial_state, mass_factor=value)
    return Scenario(f"lag_{value}", "", base.trajectory, base.duration, base.initial_state,
                    plant=PlantConfig(motor_tau=value))


def run_job(job):
    torch.set_num_threads(1)
    kind, key, controller, policies = job
    POLICIES.update(policies)
    params = QuadParams()
    scenario = SCENARIOS[key] if kind == "scenario" else sweep_scenario(kind, key)
    result = simulate(make_controller(controller, params), scenario, params)
    return kind, key, result, summarize(result, scenario)


def fmt(row, scenario):
    if row["status"] != "ok":
        return "crash"
    v = row["rmse_cm"]
    cell = f"{v:.2f}" if v >= 0.01 else "<0.01"
    if scenario in SETTLING and np.isfinite(row["settle_s"]):
        cell += f" ({row['settle_s']:.2f} s)"
    return cell


def write_tables(rows, controllers, out):
    with open(out / "benchmark.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    by = {(r["scenario"], r["controller"]): r for r in rows}
    lines = ["| Scenario | " + " | ".join(controllers) + " |", "|---|" + "---:|" * len(controllers)]
    for s in SCENARIOS:
        lines.append(f"| `{s}` | " + " | ".join(fmt(by[(s, c)], s) if (s, c) in by else "" for c in controllers) + " |")
    compute = {c: np.mean([r["compute_ms"] for r in rows if r["controller"] == c]) for c in controllers}
    lines += ["", "Mean compute time per control step (Python, CPU, single thread): "
              + ", ".join(f"{c} {compute[c]:.2f} ms" for c in controllers) + "."]
    (out / "benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plots-only", action="store_true")
    ap.add_argument("--dr-model", default=str(POLICIES["RL (DR)"]))
    ap.add_argument("--nominal-model", default=str(POLICIES["RL (nominal)"]))
    ap.add_argument("--out", default=str(RESULTS))
    ap.add_argument("--workers", type=int, default=6, help="each worker loads PyTorch (about 0.5 GB of RAM)")
    args = ap.parse_args()
    out = Path(args.out)
    POLICIES.update({"RL (DR)": Path(args.dr_model), "RL (nominal)": Path(args.nominal_model)})
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw.pkl"
    controllers = available_controllers()

    if not args.plots_only:
        jobs = [("scenario", s, c) for s in SCENARIOS for c in controllers]
        jobs += [("mass", m, c) for m in MASS_FACTORS for c in controllers]
        jobs += [("lag", tau, c) for tau in MOTOR_TAUS for c in controllers]
        jobs = [(*job, dict(POLICIES)) for job in jobs]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            outputs = list(pool.map(run_job, jobs, chunksize=2))
        with open(raw, "wb") as fh:
            pickle.dump(outputs, fh)
    with open(raw, "rb") as fh:
        outputs = pickle.load(fh)

    results = {(key, r.controller): r for kind, key, r, _ in outputs if kind == "scenario"}
    rows = [row for kind, _, _, row in outputs if kind == "scenario"]
    write_tables(rows, controllers, out)

    sweeps = [{"sweep": kind, "value": key, "controller": row["controller"], "rmse_cm": row["rmse_cm"],
               "status": row["status"]} for kind, key, _, row in outputs if kind != "scenario"]
    with open(out / "sweeps.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(sweeps[0].keys()))
        writer.writeheader()
        writer.writerows(sweeps)

    from quadrl import plotting
    plotting.make_all(results, rows, sweeps, controllers, out / "figures", RESULTS / "training")


if __name__ == "__main__":
    main()

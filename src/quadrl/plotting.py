"""Figures for the README, drawn for GitHub's light and dark themes.

Line charts show four controllers in a fixed color order that passes the
color-vision-deficiency checks on both backgrounds. "RL (nominal)" is also
dashed, so it never relies on color alone. The heatmap covers every controller.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

from quadbench.dynamics import P, Q  # noqa: E402
from quadbench.metrics import position_error  # noqa: E402
from quadbench.rotations import quat_to_rotmat  # noqa: E402
from quadbench.scenarios import SCENARIOS  # noqa: E402

LINE_SET = ["RL (DR)", "Geometric", "PID", "RL (nominal)"]
DASHED = {"RL (nominal)"}

THEMES = {
    "light": {
        "surface": "#ffffff", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e8e7e3",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
        "ramp": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
        "crash": "#f0efec",
    },
    "dark": {
        "surface": "#0d1117", "text": "#ffffff", "muted": "#c3c2b7", "grid": "#262b33",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500"],
        "ramp": ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"],
        "crash": "#383835",
    },
}


def _apply(theme):
    th = THEMES[theme]
    plt.rcParams.update({
        "figure.facecolor": th["surface"], "axes.facecolor": th["surface"],
        "savefig.facecolor": th["surface"], "text.color": th["text"],
        "axes.labelcolor": th["muted"], "axes.edgecolor": th["grid"],
        "xtick.color": th["muted"], "ytick.color": th["muted"],
        "axes.grid": True, "grid.color": th["grid"], "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "legend.frameon": False, "lines.linewidth": 1.6, "lines.solid_capstyle": "round",
    })
    return th


def _style(th, name):
    return {"color": th["series"][LINE_SET.index(name)], "linestyle": "--" if name in DASHED else "-"}


def _save(fig, out_dir, stem, theme):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}_{theme}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _bottom_legend(fig, ax, th, extra_rect=0.08):
    handles, labels = ax.get_legend_handles_labels()
    fig.tight_layout(rect=(0, extra_rect, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9.5, labelcolor=th["text"])


def plot_summary(rows, controllers, out_dir, theme):
    th = _apply(theme)
    names = list(SCENARIOS)
    grid = np.full((len(names), len(controllers)), np.nan)
    crashed = np.zeros_like(grid, dtype=bool)
    for r in rows:
        if r["controller"] in controllers:
            i, j = names.index(r["scenario"]), controllers.index(r["controller"])
            grid[i, j], crashed[i, j] = r["rmse_cm"], r["status"] != "ok"
    finite = grid[np.isfinite(grid) & ~crashed]
    norm = LogNorm(vmin=max(finite.min(), 1e-2), vmax=finite.max())
    cmap = LinearSegmentedColormap.from_list("ramp", th["ramp"])

    fig, ax = plt.subplots(figsize=(1.35 * len(controllers) + 2.2, 0.45 * len(names) + 1.4))
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(len(names)):
        for j in range(len(controllers)):
            v = grid[i, j]
            if crashed[i, j] or not np.isfinite(v):
                face, label, ink = th["crash"], "crash", th["text"]
            else:
                face = cmap(norm(max(v, norm.vmin)))
                label = f"{v:.1f}" if v >= 10 else f"{v:.2f}" if v >= 0.01 else "<0.01"
                ink = "#0b0b0b" if np.dot(face[:3], [0.2126, 0.7152, 0.0722]) > 0.45 else "#ffffff"
            ax.add_patch(plt.Rectangle((j + 0.02, i + 0.04), 0.96, 0.92, facecolor=face, edgecolor="none"))
            ax.text(j + 0.5, i + 0.5, label, ha="center", va="center", color=ink, fontsize=9.5)
    ax.set_xlim(0, len(controllers))
    ax.set_ylim(len(names), 0)
    ax.set_xticks(np.arange(len(controllers)) + 0.5, controllers)
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(names)) + 0.5, names)
    ax.tick_params(length=0, labelsize=9.5)
    for label in ax.get_xticklabels():
        label.set_color(th["text"])
        label.set_fontweight("bold")
    ax.set_title("Position RMSE (cm), lower is better", loc="left", pad=26, color=th["text"])
    fig.tight_layout()
    _save(fig, out_dir, "summary_rmse", theme)


def plot_sweeps(sweeps, controllers, out_dir, theme):
    th = _apply(theme)
    shown = [c for c in LINE_SET if c in controllers]
    crashes = {"mass": [], "lag": []}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    panels = [("mass", "Vehicle mass / model mass", axes[0], 1.0),
              ("lag", "Motor time constant (ms)", axes[1], 1000.0)]
    for kind, xlabel, ax, scale in panels:
        for c in shown:
            pts = sorted((r["value"], r["rmse_cm"], r["status"]) for r in sweeps
                         if r["sweep"] == kind and r["controller"] == c)
            xs = np.array([p[0] for p in pts]) * scale
            ok = np.array([p[2] == "ok" for p in pts])
            ys = np.array([p[1] for p in pts])
            ax.plot(xs, np.where(ok, ys, np.nan), marker="o", markersize=5, label=c, **_style(th, c))
            crashes[kind].extend((x, c) for x in xs[~ok])
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("figure-8 RMSE (cm, log scale)")
    for kind, _, ax, _ in panels:
        # Crashed runs have no RMSE: mark them with an x along the top edge instead of leaving a gap.
        top = ax.get_ylim()[1]
        stacked = {}
        for x, c in crashes[kind]:
            k = stacked.get(x, 0)                     # several crashes at one x: stack them downwards
            stacked[x] = k + 1
            ax.scatter([x], [top * 0.72 ** k], marker="x", s=55, linewidths=2, color=_style(th, c)["color"],
                       clip_on=False, zorder=5)
        if crashes[kind]:
            ax.text(1.0, 1.02, "x = crashed", transform=ax.transAxes, ha="right", va="bottom",
                    color=th["muted"], fontsize=9)
    axes[0].set_title("Wrong mass", loc="left")
    axes[1].set_title("Unmodelled motor lag", loc="left")
    _bottom_legend(fig, axes[0], th, 0.1)
    _save(fig, out_dir, "robustness_sweeps", theme)


def plot_training(training_dir, out_dir, theme):
    th = _apply(theme)
    runs = {"RL (DR)": training_dir / "ppo_dr.csv", "RL (nominal)": training_dir / "ppo_nominal.csv"}
    runs = {k: v for k, v in runs.items() if v.exists()}
    if not runs:
        return
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    metrics = [("episode_return", "Episode return"), ("crash_rate", "Crash rate"),
               ("episode_mean_error_m", "Mean position error (m)")]
    for name, path in runs.items():
        with open(path) as fh:
            rows = list(csv.DictReader(fh))
        steps = np.array([float(r["steps"]) for r in rows]) / 1e6
        for ax, (key, title) in zip(axes, metrics):
            ax.plot(steps, np.array([float(r[key]) for r in rows]), label=name, **_style(th, name))
            ax.set_title(title, loc="left")
            ax.set_xlabel("environment steps (millions)")
    axes[2].set_ylim(bottom=0)
    _bottom_legend(fig, axes[0], th, 0.12)
    _save(fig, out_dir, "training_curves", theme)


def plot_scenario(results, name, out_dir, theme):
    th = _apply(theme)
    shown = [c for c in LINE_SET if (name, c) in results]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for c in shown:
        r = results[(name, c)]
        ax.plot(r.t, np.maximum(100 * position_error(r), 1e-3), label=c + (" (crashed)" if r.crashed else ""),
                **_style(th, c))
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-3)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("position error (cm, log scale)")
    ax.set_title(f"{name}: {SCENARIOS[name].description}", loc="left", fontsize=10)
    _bottom_legend(fig, ax, th, 0.12)
    _save(fig, out_dir, name, theme)


def make_side_gif(results, name, controllers, out_dir, theme, seconds=3.0, fps=25, title="3 m step",
                  axis=0, span=(-0.4, 3.5), height=(-0.05, 1.45), row=False):
    """Side view of one horizontal axis (0 = x, 1 = y) against z, one panel per controller."""
    th = _apply(theme)
    runs = [results[(name, c)] for c in controllers if (name, c) in results]
    target = runs[0].ref_pos[-1]
    if row:
        fig, axes = plt.subplots(1, len(runs), figsize=(10, 3.1))
        fig.subplots_adjust(left=0.06, right=0.99, top=0.80, bottom=0.17, wspace=0.28)
        label_y, label_x = axes[:1], axes
    else:
        fig, axes = plt.subplots(2, 2, figsize=(10, 3.9))
        fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.12, hspace=0.35, wspace=0.12)
        axes = axes.ravel()
        label_y, label_x = axes[::2], axes[2:]
    header = fig.suptitle("", color=th["text"], fontsize=11)

    def draw(frame):
        t_now = frame / fps
        header.set_text(f"{title}    t = {t_now:.2f} s")
        for ax, r in zip(axes, runs):
            ax.cla()
            color = _style(th, r.controller)["color"] if r.controller in LINE_SET else th["muted"]
            ax.set_xlim(*span)
            ax.set_ylim(*height)
            ax.set_aspect("equal")
            ax.axhline(0, color=th["muted"], lw=1.2)
            ax.scatter([target[axis]], [target[2]], color=th["muted"], marker="x", s=30, zorder=3)
            i = min(np.searchsorted(r.t, t_now), len(r.t) - 1)
            ax.plot(r.x[: i + 1, axis], r.x[: i + 1, 2], color=color, lw=1.2, alpha=0.7)
            R = quat_to_rotmat(r.x[i, Q])
            c = r.x[i, P]
            # The body axis lying in the viewing plane, and the thrust axis.
            arm, tip = 0.16 * R[:, axis], c + 0.16 * R[:, 2]
            ax.plot([c[axis] - arm[axis], c[axis] + arm[axis]], [c[2] - arm[2], c[2] + arm[2]],
                    color=th["text"], lw=2.4)
            ax.plot([c[axis], tip[axis]], [c[2], tip[2]], color=color, lw=2.0)
            tilt = np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))
            crashed = r.crashed and t_now >= r.t[-1]
            ax.text(0.01, 0.92, r.controller, transform=ax.transAxes, color=th["text"], fontsize=10,
                    fontweight="bold", va="top")
            ax.text(0.99, 0.04, "CRASHED" if crashed else f"tilt {tilt:4.0f} deg", transform=ax.transAxes,
                    color=th["text"], fontsize=9.5, ha="right", va="bottom")
        for ax in label_y:
            ax.set_ylabel("z (m)")
        for ax in label_x:
            ax.set_xlabel("xy"[axis] + " (m)")
        return []

    anim = FuncAnimation(fig, draw, frames=int(seconds * fps), blit=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    anim.save(out_dir / f"{name}_{theme}.gif", writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)


def make_all(results, rows, sweeps, controllers, out_dir: Path, training_dir: Path):
    for theme in THEMES:
        plot_summary(rows, controllers, out_dir, theme)
        plot_sweeps(sweeps, controllers, out_dir, theme)
        plot_training(training_dir, out_dir, theme)
        for name in SCENARIOS:
            plot_scenario(results, name, out_dir, theme)
        gif_set = [c for c in LINE_SET if ("flip_recovery", c) in results]
        if len(gif_set) == 4:
            make_side_gif(results, "flip_recovery", gif_set, out_dir, theme, seconds=2.5,
                          title="Recovery from 170 deg roll", axis=1, span=(-0.8, 0.8), height=(2.1, 3.4),
                          row=True)
    print(f"figures written to {out_dir}")

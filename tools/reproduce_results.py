#!/usr/bin/env python3
"""Regenerate the README results (tables + figure) from the saved run folders.

For each reported run (10 = FEP, 31 = Adam, 37 = Yama) this script loads the
saved ``init_param.mat`` (warm start) and ``optimal_param.mat`` (trained
weights), simulates the 9-point m2/l2 sweep of ``src/visualization.m`` with
``pendubot_sim.py`` and prints settling time, overshoot, offset from upright
and torque statistics. It also writes ``docs/figures/parameter_sweep_x1.png``.

Torque de-normalisation ranges
------------------------------
The network output is scaled by the range stored in
``test_drive/normalization_data.mat``. That file was regenerated several times
and is not saved per run, so the range each controller was trained with was
recovered by least squares from MATLAB's own saved rollouts (residual ~1e-13):

* run 31: from ``results/31/x_stack.mat`` / ``u_stack.mat`` (26 Jul 2025)
* run 37: from ``results/37/x_stack.mat`` / ``u_stack.mat`` (23 Aug 2025)
* run 10: no rollout is stored in the folder; every saved run-10 figure
  (Optimization_*.fig, Testing_*.fig, 26 Jul 2025) uses the run-37 range, so
  that range is used here.
* "current" is the committed ``normalization_data.mat`` (23 Aug 2025, 18:58);
  it is what ``scripts/main.m`` with ``flag_mode = 1`` uses today.

Usage:  python tools/reproduce_results.py [--plant coded|documented] [--no-plot]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig_metrics as fm  # noqa: E402
import pendubot_sim as ps  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "robotics_result" / "robust_optimization" / "test_drive" / "results"

NORM_RUN31 = (-103.14538630530888, 95.14078656412474)
NORM_RUN37 = (-74.36120540244652, 68.69836805883476)
NORM_CURRENT = (-69.3866936064508, 64.06446952934277)

RUNS = [  # label, folder, as-trained normalisation
    ("Adam (run 31)", "31", NORM_RUN31),
    ("Yama (run 37)", "37", NORM_RUN37),
    ("FEP (run 10)", "10", NORM_RUN37),
]


def load_init(folder: str) -> np.ndarray:
    import scipy.io as sio
    return np.asarray(sio.loadmat(RESULTS / folder / "init_param.mat", squeeze_me=True)["init_param"], float).ravel()


def evaluate(w, norm, plant):
    runs = ps.sweep(w, norm, plant)
    metrics = [fm.run_metrics(x[:, 0], u) for x, u in runs]
    stable = 0
    for m in metrics:
        if m["diverged"]:
            break
        stable += 1
    return runs, metrics, stable


def fmt_row(label, weights, norm_name, metrics, contiguous):
    n = metrics[0]
    total = sum(not m["diverged"] for m in metrics)
    upto = f"+{(contiguous - 1) * 10}%" if contiguous else "-"
    if n["diverged"]:
        nominal = "diverges | - | - | - | -"
    else:
        nominal = f"{n['offset_from_pi']:+.3f} | {n['Ts']:.2f} | {n['OS']:.1f} | {n['rms_u']:.4f} | {n['max_u']:.3f}"
    return f"| {label} | {weights} | {norm_name} | {total}/9 | {upto} | {nominal} |"


def plot(panels, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 3.5), sharey=True)
    cmap = plt.get_cmap("viridis")
    t = np.arange(ps.N) * ps.DT
    for ax, (title, runs, metrics) in zip(axes, panels):
        for i, ((xs, _), m) in enumerate(zip(runs, metrics)):
            colour = cmap(i / 8)
            x1 = xs[:, 0]
            if m["diverged"]:
                bad = np.flatnonzero(~np.isfinite(x1) | (np.abs(x1 - np.pi) > 0.5))
                stop = bad[0] if bad.size else x1.size
                ax.plot(t[:stop], x1[:stop], color=colour, lw=1, ls="--")
                ax.plot(t[max(stop - 1, 0)], x1[max(stop - 1, 0)], "x", color=colour, ms=6)
            else:
                ax.plot(t, x1, color=colour, lw=1.3)
        ax.axhline(np.pi, color="0.35", lw=0.8, ls=":")
        stable = sum(not m["diverged"] for m in metrics)
        ax.set_title(f"{title}\n{stable}/9 sweep runs stable", fontsize=10)
        ax.set_xlabel("time [s]")
        ax.set_ylim(2.85, 3.25)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"$x_1$ (link-1 angle) [rad]")
    handles = [plt.Line2D([], [], color=cmap(i / 8), lw=1.5) for i in range(9)]
    fig.legend(handles, [f"{1 + 0.1 * i:.1f}x" for i in range(9)], title="m2, l2 scale",
               loc="center right", fontsize=8, title_fontsize=8)
    fig.text(0.01, 0.01, "dotted: upright target x1 = pi;  dashed + x: run diverges (|x| > 1e4 later)",
             fontsize=7, color="0.3")
    fig.tight_layout(rect=(0, 0.03, 0.9, 1))
    fig.savefig(out, dpi=150)
    print(f"\nsaved {out.relative_to(ROOT)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--plant", choices=ps.PLANTS, default="coded")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    header = ("| Controller | Weights | Normalisation | Stable runs | Stable up to | "
              "Nominal pi - x1 [rad] | Ts [s] | %OS | RMS(u) [N m] | max abs u [N m] |")
    print(f"plant = {args.plant}\n")
    print(header)
    print("|---" * (header.count("|") - 1) + "|")
    panels = []
    for label, folder, norm in RUNS:
        optimal = ps.load_params(RESULTS / folder / "optimal_param.mat")
        warm = load_init(folder)
        for weights_name, w in (("warm start", warm), ("trained", optimal)):
            for norm_name, nrm in (("as trained", norm), ("current file", NORM_CURRENT)):
                runs, metrics, contiguous = evaluate(w, nrm, args.plant)
                print(fmt_row(label, weights_name, norm_name, metrics, contiguous))
                if weights_name == "trained" and norm_name == "as trained":
                    panels.append((label, runs, metrics))
    if not args.no_plot and args.plant == "coded":
        plot(panels, ROOT / "docs" / "figures" / "parameter_sweep_x1.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())

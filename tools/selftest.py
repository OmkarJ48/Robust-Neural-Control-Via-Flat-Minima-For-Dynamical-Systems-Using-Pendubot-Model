#!/usr/bin/env python3
"""Consistency checks for the repository (run locally or in GitHub Actions).

1. The NumPy re-simulation reproduces MATLAB's saved rollouts and figures.
2. Every ``map_2_assign_value`` in the active model configs gives each
   parameter its own value (the Pendubot swap fix), while the run snapshots
   in ``results/<n>/`` still contain the historical swap.
3. The headline robustness numbers in the README still hold.
4. No MATLAB file sits in the repository root.

Usage:  python tools/selftest.py      (exit code 0 = all checks passed)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig_metrics as fm  # noqa: E402
import pendubot_sim as ps  # noqa: E402
import reproduce_results as rr  # noqa: E402

ROOT = rr.ROOT
TEST_DRIVE = rr.RESULTS.parent
FIGS = ROOT / "docs" / "figures" / "matlab"
TOL = 1e-9
failures: list[str] = []


def check(ok: bool, message: str) -> None:
    print(("PASS  " if ok else "FAIL  ") + message)
    if not ok:
        failures.append(message)


def load_stack(path: Path, key: str) -> np.ndarray:
    import scipy.io as sio
    return np.asarray(sio.loadmat(path, squeeze_me=True)[key], float)


def params(run: str) -> np.ndarray:
    return ps.load_params(rr.RESULTS / run / "optimal_param.mat")


def run_norm(run: str):
    return rr.load_norm(rr.RESULTS / run / "normalization_data.mat")


# 1. Re-simulation against MATLAB output ------------------------------------
for run in ("31", "37"):
    xs, _ = ps.simulate(params(run), run_norm(run))
    err = np.abs(xs - load_stack(rr.RESULTS / run / "x_stack.mat", "x_stack")).max()
    check(err < TOL, f"run {run}: simulation matches results/{run}/x_stack.mat (max error {err:.1e})")

xs, _ = ps.simulate(params("31"), rr.NORM_CURRENT)
err = np.abs(xs - load_stack(ROOT / "docs" / "data" / "x_stack.mat", "x_stack")).max()
check(err < TOL, f"run 31 with current normalisation matches docs/data/x_stack.mat (max error {err:.1e})")

for run, norm, fig in (("10", run_norm("10"), "Testing_2.fig"),
                       ("10", run_norm("10"), "Optimization_2.fig"),
                       ("37", run_norm("37"), "Adam_optimization_3.fig"),
                       ("37", rr.NORM_CURRENT, "Yama_testing_2.fig")):
    err = ps.validate(params(run), norm, FIGS / fig)
    check(err < TOL, f"run {run}: simulation matches {fig} (max error {err:.1e})")


# 2. Parameter maps ----------------------------------------------------------
def map_mismatches(path: Path) -> list[str] | None:
    text = re.sub(r"%.*", "", path.read_text(errors="ignore")).replace("...\n", " ")
    fp = re.search(r"fixed_parameters\s*=\s*\[([^\]]*)\]", text)
    v = re.search(r"v_nominal\s*=\s*\[([^\]]*)\]", text)
    m = re.search(r"containers\.Map\(\{([^}]*)\}\s*,\s*\{([^}]*)\}", text)
    if not (fp and v and m):
        return None
    split = lambda s: [t for t in re.split(r"[\s,;]+", s.strip()) if t]  # noqa: E731
    fixed, unc = split(fp.group(1)), split(v.group(1))
    keys, vals = re.findall(r"'([^']+)'", m.group(1)), re.findall(r"'([^']+)'", m.group(2))
    bad = []
    for key, val in zip(keys, vals):
        source, idx = val.split("_")
        got = (fixed if source == "fp" else unc)[int(idx) - 1]
        if got != key:
            bad.append(f"{key} <- {got}")
    return bad


active = sorted(TEST_DRIVE.glob("*.m")) + sorted((ROOT / "src").rglob("*Conf.m")) + sorted((ROOT / "config_templates").glob("*.m"))
for path in active:
    bad = map_mismatches(path)
    if bad is not None:
        check(not bad, f"{path.relative_to(ROOT)}: every parameter gets its own value" + (f" ({', '.join(bad)})" if bad else ""))
for run in ("10", "31", "37"):
    bad = map_mismatches(rr.RESULTS / run / "PendubotModelConf.m")
    check(bool(bad), f"results/{run}/PendubotModelConf.m still records the historical swap ({', '.join(bad or [])})")


# 3. Headline results --------------------------------------------------------
expected = {  # stable sweep runs with the trained weights (README, section 4.2)
    ("31", "as-run"): 1, ("37", "as-run"): 4, ("10", "as-run"): 9,
    ("31", "corrected"): 0, ("37", "corrected"): 0, ("10", "corrected"): 0,
}
for (run, plant), n_stable in expected.items():
    _, metrics, _ = rr.evaluate(params(run), run_norm(run), plant)
    got = sum(not m["diverged"] for m in metrics)
    check(got == n_stable, f"run {run}, {plant} plant: {got}/9 sweep runs stable (README: {n_stable}/9)")


# LQR teacher (Q = diag(5, 5, 0.5, 0.5), R = 20, |u| <= 0.5) as in mainTeachingByDemonstration.m
def lqr_rollout(plant: str) -> np.ndarray:
    from scipy.linalg import solve_continuous_are
    (m1, l1), split, scalar_friction = ps.PLANTS[plant]
    m2, l2 = split(ps.V_NOMINAL)
    f = lambda x, u: ps.xdot(x, u, m1, l1, m2, l2, scalar_friction)  # noqa: E731
    x_des, eps = np.array([np.pi, 0.0, 0.0, 0.0]), 1e-6
    a = np.stack([(f(x_des + eps * e, 0.0) - f(x_des - eps * e, 0.0)) / (2 * eps) for e in np.eye(4)], 1)
    b = ((f(x_des, eps) - f(x_des, -eps)) / (2 * eps))[:, None]
    r = np.array([[20.0]])
    k = np.linalg.solve(r, b.T @ solve_continuous_are(a, b, np.diag([5, 5, 0.5, 0.5]), r)).ravel()
    x, xs = ps.X_INIT.copy(), []
    for _ in range(ps.N):
        xs.append(x.copy())  # perturbedTrajectoriesTbD stores the state before each step
        x = x + f(x, float(np.clip(k @ (x_des - x), -0.5, 0.5))) * ps.DT
    return np.array(xs)


teacher = np.stack(fm.read_lines(FIGS / "Teacher_controller_optimized.fig"), 1)
err = np.abs(lqr_rollout("as-run") - teacher).max()
check(err < 1e-8, f"LQR teacher on the as-run plant matches Teacher_controller_optimized.fig (max error {err:.1e})")
final = lqr_rollout("corrected")[-1]
check(abs(final[0] - np.pi) < 1e-3 and np.abs(final[1:]).max() < 1e-3,
      f"LQR teacher stabilises the corrected plant (final x = {np.round(final, 4).tolist()})")


# 4. Repository layout --------------------------------------------------------
root_m = sorted(p.name for p in ROOT.glob("*.m"))
check(not root_m, "no MATLAB files in the repository root" + (f" ({', '.join(root_m)})" if root_m else ""))

print(f"\n{'all checks passed' if not failures else f'{len(failures)} check(s) failed'}")
sys.exit(1 if failures else 0)

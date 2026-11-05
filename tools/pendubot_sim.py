#!/usr/bin/env python3
"""Re-simulate trained Pendubot controllers without MATLAB.

A NumPy re-implementation of the closed loop used by ``scripts/main.m``:

* ``src/+model/Pendubot.m`` dynamics (friction as in the selected plant, see below),
* explicit Euler integration ``x+ = x + f(x, u) * DT`` (``src/integra.m``, zero noise),
* the 4-10-1 tanh network of ``src/+controller/neuralController.m`` with output
  de-normalisation ``tau = (y + 1) / 2 * (max - min) + min``,
* the parameter sweep of ``src/visualization.m``: ``v_nominal .* (1 + (ii-1)*0.1)``.

It reproduces MATLAB to round-off: for run 31 it matches the saved
``docs/data/x_stack.mat`` to about 1e-13, and the trajectories stored in the
run-10 and run-37 figures to about 1e-12 (see ``--validate``).

Plant parameters
----------------
``PendubotModelConf.m`` routes its values through ``map_2_assign_value``. The
copy used for runs 10, 31 and 37 (saved in ``results/<n>/``) assigned
``fixed_parameters(1)`` (= m_1) to ``l_1``, ``fixed_parameters(2)`` (= l_1) to
``m_1``, ``v(1)`` (= m_2) to ``l_2`` and ``v(2)`` (= l_2) to ``m_2``, so those
runs simulated m1 = 0.1032, l1 = 0.193, m2 = 0.1065 and l2 = 0.073. Their
``Pendubot.m`` also applied friction as ``[mu_1;mu_2]' * q_dot``, a scalar
subtracted from both joint equations.

* ``--plant as-run`` (the default) reproduces runs 10, 31 and 37.
* ``--plant corrected`` matches the current code: m1 = 0.193, l1 = 0.1032,
  m2 = 0.073, l2 = 0.1065 and per-joint friction ``diag(mu_1, mu_2) * q_dot``.

Examples
--------
    python tools/pendubot_sim.py robotics_result/robust_optimization/test_drive/results/10/optimal_param.mat \
        --norm -74.36120540244652 68.69836805883476 --sweep
    python tools/pendubot_sim.py robotics_result/robust_optimization/test_drive/results/10/optimal_param.mat \
        --norm -74.36120540244652 68.69836805883476 --validate docs/figures/matlab/Testing_2.fig
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig_metrics as fm  # noqa: E402

DT, N = 0.01, 800
X_INIT = np.array([2.98, 0.1, 0.3, 0.3])
A1, A2, MU, G = 0.1483, 0.1804, 0.01, 9.8          # link lengths, friction, gravity
V_NOMINAL = np.array([0.073, 0.1065])              # PendubotModelConf.m: [m_2, l_2]
PLANTS = {
    # (m1, l1), a function v -> (m2, l2), scalar friction?
    "as-run": ((0.1032, 0.193), lambda v: (v[1], v[0]), True),       # runs 10, 31, 37
    "corrected": ((0.193, 0.1032), lambda v: (v[0], v[1]), False),   # current code
}


def load_params(path: Path) -> np.ndarray:
    """Load ``optimal_param`` from a .mat file (plain double or a saved casadi.DM)."""
    try:
        import scipy.io as sio
        val = sio.loadmat(path, squeeze_me=True)["optimal_param"]
        arr = np.asarray(val, dtype=float).ravel()
        if arr.size:
            return arr
    except (TypeError, ValueError, KeyError):
        pass
    # casadi.DM objects are saved via CasADi's string serialisation:
    # one letter a..p per nibble, low nibble first; the values are the trailing doubles.
    raw = path.read_bytes()
    blobs = [m for blk in fm._compressed_blocks(raw) for m in re.findall(rb"[a-p]{200,}", blk)]
    if not blobs:
        raise ValueError(f"cannot read optimal_param from {path}")
    s = max(blobs, key=len).decode()
    data = bytes((ord(s[i]) - 97) + 16 * (ord(s[i + 1]) - 97) for i in range(0, len(s) - 1, 2))
    return np.frombuffer(data[-61 * 8:], "<f8").copy()


def controller(w: np.ndarray, norm: tuple[float, float]):
    w1 = w[:40].reshape(4, 10, order="F")
    w2, b1, b2 = w[40:50], w[50:60], w[60]
    lo, hi = norm
    return lambda x: (np.tanh(x @ w1 + b1) @ w2 + b2 + 1) / 2 * (hi - lo) + lo


def xdot(x, u, m1, l1, m2, l2, scalar_friction=True):
    i1, i2 = m1 * A1 ** 2 / 12, m2 * A2 ** 2 / 12
    k1 = i1 + i2 + m1 * l1 ** 2 + m2 * (A1 ** 2 + l2 ** 2)
    k2 = i2 + m2 * l2 ** 2
    k3 = m2 * A1 * l2
    k4 = m2 * A1 + m1 * l1
    k5 = m2 * l2
    c, s = np.cos(x[1]), np.sin(x[1])
    mass = np.array([[k1 + 2 * k3 * c, k2 + k3 * c], [k2 + k3 * c, k2]])
    cor = np.array([[-k3 * s * x[3], -k3 * s * (x[2] + x[3])], [k3 * s * x[2], 0.0]])
    grav = np.array([k4 * G * np.sin(x[0]) + k5 * G * np.sin(x[0] + x[1]), k5 * G * np.sin(x[0] + x[1])])
    if scalar_friction:
        friction = MU * x[2] + MU * x[3]      # [mu_1;mu_2]' * qd: one scalar on both rows (runs 10, 31, 37)
    else:
        friction = MU * x[2:]                 # diag(mu_1, mu_2) * qd (current Pendubot.m)
    qdd = np.linalg.solve(mass, -cor @ x[2:] - grav + np.array([u, 0.0]) - friction)
    return np.r_[x[2:], qdd]


def simulate(w, norm, scale=1.0, plant="as-run"):
    """Return (X, U): states after each step (like x_stack) and torques (u_stack)."""
    (m1, l1), split, scalar_friction = PLANTS[plant]
    m2, l2 = split(V_NOMINAL * scale)
    ctrl = controller(w, norm)
    x, xs, us = X_INIT.copy(), [], []
    with np.errstate(all="ignore"):
        for _ in range(N):
            u = ctrl(x)
            x = x + xdot(x, u, m1, l1, m2, l2, scalar_friction) * DT
            xs.append(x.copy())
            us.append(u)
    return np.array(xs), np.array(us)


def sweep(w, norm, plant="as-run", runs=9):
    return [simulate(w, norm, 1 + fm.SWEEP_STEP * i, plant) for i in range(runs)]


def validate(w, norm, fig: Path, plant="as-run") -> float:
    """Max abs difference to a figure, over the runs that did not diverge."""
    chans = fm.split_channels(fm.read_lines(fig), 5)
    worst = 0.0
    for r, (xs, us) in enumerate(sweep(w, norm, plant, runs=len(chans[0]))):
        ref = np.stack([chans[c][r] for c in range(5)], 1)
        sim = np.c_[xs, us]
        if np.all(np.isfinite(ref)) and np.abs(ref).max() < fm.DIVERGENCE:
            worst = max(worst, float(np.abs(ref - sim).max()))
    return worst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("params", type=Path, help="optimal_param.mat of a run")
    ap.add_argument("--norm", nargs=2, type=float, required=True, metavar=("MIN", "MAX"),
                    help="torque de-normalisation range the controller was trained with")
    ap.add_argument("--plant", choices=PLANTS, default="as-run")
    ap.add_argument("--sweep", action="store_true", help="run the 9-point m2/l2 sweep")
    ap.add_argument("--validate", type=Path, help="compare against a .fig produced by visualization.m")
    args = ap.parse_args(argv)

    w = load_params(args.params)
    norm = tuple(args.norm)
    if args.validate:
        print(f"max |sim - fig| over non-divergent runs: {validate(w, norm, args.validate, args.plant):.2e}")
        return 0
    results = []
    for i, (xs, us) in enumerate(sweep(w, norm, args.plant, 9 if args.sweep else 1)):
        m = fm.run_metrics(xs[:, 0], us)
        m["scale"] = 1 + fm.SWEEP_STEP * i
        results.append(m)
    fm.print_table(f"{args.params} (plant={args.plant}, norm={norm})", results)
    return 0


if __name__ == "__main__":
    sys.exit(main())

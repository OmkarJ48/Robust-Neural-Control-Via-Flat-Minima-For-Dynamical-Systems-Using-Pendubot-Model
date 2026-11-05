#!/usr/bin/env python3
"""Extract closed-loop trajectories from MATLAB .fig files and compute metrics.

The figures written by ``src/visualization.m`` / ``src/visualizationAux.m`` contain one
subplot per state (x1..x4) plus one for the torque u1. A nominal run stores
5 lines; the parameter sweep (v_nominal scaled by 1.0, 1.1, ..., 1.8) stores
9 runs x 5 channels = 45 lines. This script recovers those lines without
MATLAB and reports, per run:

* final x1 and its offset from the upright target (pi),
* settling time Ts (x1 stays within 2 % of the step size |x1_final - x1_0|),
* percentage overshoot of x1 relative to the step size,
* RMS and peak of the torque u1,
* divergence (any |x| > 1e4, the same threshold visualizationAux.m uses).

Usage
-----
    python tools/fig_metrics.py docs/figures/matlab/*.fig
    python tools/fig_metrics.py A.fig B.fig --plot sweep.png --labels "A" "B"

Limitations
-----------
MATLAB R2014b+ stores figures as opaque MCOS objects. The parser does not
decode those objects: it scans the compressed MAT-file blocks for double
arrays of the trajectory length and relies on MATLAB writing plot children in
reverse creation order. For the figures in docs/figures/matlab this was
cross-checked two ways: the nominal trace of Optimization_1.fig equals the
1.0x run of Optimization_2.fig exactly, and every settled run satisfies
x1 + x2 = pi (link 2 vertical), as the Pendubot equilibria require. It is
still a heuristic, not a general .fig reader.
"""
from __future__ import annotations

import argparse
import struct
import sys
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

MI_DOUBLE = 9
MI_COMPRESSED = 15
DT = 0.01               # integration step used in experimentEnvOptConf.m
DIVERGENCE = 1e4        # same threshold as visualizationAux.m
SWEEP_STEP = 0.1        # visualization.m: v_nominal .* (1 + (ii-1)*0.1)


def _compressed_blocks(raw: bytes):
    pos = 128  # skip MAT v5 header
    while pos + 8 <= len(raw):
        dtype, nbytes = struct.unpack("<II", raw[pos:pos + 8])
        if dtype == MI_COMPRESSED:
            try:
                yield zlib.decompress(raw[pos + 8:pos + 8 + nbytes])
            except zlib.error:
                pass
            pos += 8 + nbytes
        else:
            pos += 8 + nbytes + ((8 - nbytes % 8) % 8)


def _double_arrays(block: bytes, min_len: int = 50):
    i = block.find(struct.pack("<I", MI_DOUBLE))
    while i != -1:
        nbytes = struct.unpack("<I", block[i + 4:i + 8])[0] if i + 8 <= len(block) else 0
        n = nbytes // 8
        if nbytes % 8 == 0 and n >= min_len and i + 8 + nbytes <= len(block):
            yield np.frombuffer(block[i + 8:i + 8 + nbytes], "<f8")
        i = block.find(struct.pack("<I", MI_DOUBLE), i + 4)


def read_lines(path: Path) -> list[np.ndarray]:
    """Return the YData arrays of all plotted lines, in creation order."""
    raw = path.read_bytes()
    arrays = [a for blk in _compressed_blocks(raw) for a in _double_arrays(blk)]
    if not arrays:
        return []
    length = Counter(a.size for a in arrays).most_common(1)[0][0]
    xdata = np.arange(1, length + 1)
    lines = [a for a in arrays if a.size == length and not np.array_equal(a, xdata)]
    return lines[::-1]  # MATLAB stores children newest-first


def split_channels(lines: list[np.ndarray], n_channels: int) -> list[list[np.ndarray]]:
    if len(lines) % n_channels:
        raise ValueError(f"{len(lines)} lines cannot be split into {n_channels} channels")
    runs = len(lines) // n_channels
    return [lines[c * runs:(c + 1) * runs] for c in range(n_channels)]


def run_metrics(x1: np.ndarray, u: np.ndarray | None) -> dict:
    finite = np.all(np.isfinite(x1)) and np.max(np.abs(x1)) < DIVERGENCE
    if not finite:
        return {"diverged": True}
    x_final, x_start = x1[-1], x1[0]
    step = abs(x_final - x_start)
    band = max(0.02 * step, 1e-3)
    outside = np.flatnonzero(np.abs(x1 - x_final) > band)
    ts = (outside[-1] + 1) * DT if outside.size else 0.0
    peak = np.max(x1) - x_final if x_final >= x_start else x_final - np.min(x1)
    out = {
        "diverged": False,
        "x1_final": x_final,
        "offset_from_pi": np.pi - x_final,
        "Ts": ts,
        "OS": max(peak, 0.0) / max(step, 1e-3) * 100.0,
    }
    if u is not None:
        out["rms_u"] = float(np.sqrt(np.mean(u ** 2)))
        out["max_u"] = float(np.max(np.abs(u)))
    return out


def analyse(path: Path, n_channels: int, horizon: int):
    lines = read_lines(path)
    if not lines:
        return None
    if lines[0].size != horizon or len(lines) % 4 and len(lines) % n_channels:
        return "history", lines
    if len(lines) % n_channels:
        # figures without a torque subplot (e.g. teacher/student state plots)
        n_channels = 4 if len(lines) % 4 == 0 else n_channels
    chans = split_channels(lines, n_channels)
    has_u = n_channels >= 5
    results = []
    for r in range(len(chans[0])):
        m = run_metrics(chans[0][r], chans[4][r] if has_u else None)
        m["scale"] = 1.0 + SWEEP_STEP * r if len(chans[0]) > 1 else 1.0
        results.append(m)
    return chans, results


def print_table(name: str, results: list[dict]) -> None:
    print(f"\n{name}")
    print(f"{'scale':>6} {'x1_final':>9} {'pi-x1':>8} {'Ts[s]':>6} {'%OS':>6} {'RMS(u)':>8} {'max|u|':>8}")
    for m in results:
        if m["diverged"]:
            print(f"{m['scale']:>6.1f} {'DIVERGED':>9}")
            continue
        print(f"{m['scale']:>6.1f} {m['x1_final']:>9.4f} {m['offset_from_pi']:>8.4f} {m['Ts']:>6.2f} "
              f"{m['OS']:>6.1f} {m.get('rms_u', float('nan')):>8.4f} {m.get('max_u', float('nan')):>8.3f}")


def plot(files, labels, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(files), figsize=(4.2 * len(files), 3.4), sharey=True, squeeze=False)
    cmap = plt.get_cmap("viridis")
    for ax, (chans, results), label in zip(axes[0], files, labels):
        runs = len(chans[0])
        t = np.arange(chans[0][0].size) * DT
        for r in range(runs):
            x1 = chans[0][r]
            colour = cmap(r / max(runs - 1, 1))
            scale = results[r]["scale"]
            if results[r]["diverged"]:
                ok = np.isfinite(x1) & (np.abs(x1) < 3.6)
                last = np.flatnonzero(~ok)
                stop = last[0] if last.size else x1.size
                ax.plot(t[:stop], x1[:stop], color=colour, lw=1, ls="--")
                ax.plot(t[stop - 1], x1[stop - 1], "x", color=colour)
            else:
                ax.plot(t, x1, color=colour, lw=1.2, label=f"x{scale:.1f}")
        ax.axhline(np.pi, color="0.4", lw=0.8, ls=":")
        stable = sum(not m["diverged"] for m in results)
        ax.set_title(f"{label}\n{stable}/{runs} runs stable")
        ax.set_xlabel("time [s]")
        ax.set_ylim(2.85, 3.25)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel(r"$x_1$ (link-1 angle) [rad]")
    handles = [plt.Line2D([], [], color=cmap(i / 8), lw=1.5) for i in range(9)]
    fig.legend(handles, [f"{1 + 0.1 * i:.1f}x" for i in range(9)], title="m2, l2 scale",
               loc="center right", fontsize=8, title_fontsize=8)
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    fig.savefig(out, dpi=150)
    print(f"\nsaved {out}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("figs", nargs="+", type=Path)
    ap.add_argument("--channels", type=int, default=5, help="subplots per run (default 5: x1..x4, u1)")
    ap.add_argument("--horizon", type=int, default=800,
                    help="rollout length N; figures with other line lengths are treated as optimisation histories")
    ap.add_argument("--plot", type=Path, help="save an x1 comparison plot to this PNG")
    ap.add_argument("--labels", nargs="*", help="panel titles for --plot (default: file names)")
    args = ap.parse_args(argv)

    analysed = []
    for f in args.figs:
        res = analyse(f, args.channels, args.horizon)
        if res is None:
            print(f"\n{f.name}: no line data found")
            continue
        if res[0] == "history":
            print(f"\n{f.name}: {len(res[1])} optimisation-history curve(s)")
            for i, y in enumerate(res[1], 1):
                print(f"   curve {i}: {y.size} iterations, first {y[0]:.4g}, last {y[-1]:.4g}, "
                      f"min {np.min(y):.4g}, max {np.max(y):.4g}")
            continue
        print_table(f.name, res[1])
        analysed.append((f, res))

    if args.plot:
        labels = args.labels or [f.stem for f, _ in analysed]
        plot([r for _, r in analysed], labels, args.plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())

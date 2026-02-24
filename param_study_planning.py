#!/usr/bin/env python3
"""
Parameter-sensitivity study for PRM and RRT planners.

Sweeps:
  PRM  -- n_samples x k_neighbors
  RRT  -- step_size x goal_sample_rate

Each configuration is evaluated over multiple trials on the default
Environment(seed=42).  Metrics (success rate, mean path length, mean time,
mean smoothness) are logged to CSV and visualised as line / heatmap plots.

All outputs land in  planning_param_studies/ .
"""

import os
import sys
import csv
import time
import contextlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from environment import Environment
from motion_planning import PRM, RRT

# ── output ──────────────────────────────────────────────────────────────────
OUT = "planning_param_studies"
os.makedirs(OUT, exist_ok=True)

# ── trial helpers ───────────────────────────────────────────────────────────

@contextlib.contextmanager
def _quiet():
    """Suppress planner stdout."""
    old = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = old


def _path_len(p):
    if p is None or len(p) < 2:
        return 0.0
    return float(sum(np.linalg.norm(p[i+1] - p[i]) for i in range(len(p)-1)))


def _smoothness(p):
    if p is None or len(p) < 3:
        return 0.0
    angles = []
    for i in range(1, len(p)-1):
        v1, v2 = p[i] - p[i-1], p[i+1] - p[i]
        c = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)+1e-12),
                     -1, 1)
        angles.append(abs(np.arccos(c)))
    return float(np.mean(angles))

# ── single-run wrappers ────────────────────────────────────────────────────

def _run_prm(env, start, goal, seed, n_samples, k_neighbors):
    np.random.seed(seed)
    planner = PRM(env, n_samples=n_samples, k_neighbors=k_neighbors)
    t0 = time.perf_counter()
    with _quiet():
        path = planner.plan(start, goal)
    dt = time.perf_counter() - t0
    ok = path is not None
    return dict(
        success=ok,
        path_length=_path_len(path) if ok else float("nan"),
        time_s=dt,
        smoothness=_smoothness(path) if ok else float("nan"),
        nodes=len(planner.nodes),
    )


def _run_rrt(env, start, goal, seed,
             step_size, goal_sample_rate, max_iter=3000, goal_tol=0.5):
    np.random.seed(seed)
    planner = RRT(env, max_iter=max_iter, step_size=step_size,
                  goal_sample_rate=goal_sample_rate)
    t0 = time.perf_counter()
    with _quiet():
        path = planner.plan(start, goal, goal_tolerance=goal_tol)
    dt = time.perf_counter() - t0
    ok = path is not None
    return dict(
        success=ok,
        path_length=_path_len(path) if ok else float("nan"),
        time_s=dt,
        smoothness=_smoothness(path) if ok else float("nan"),
        nodes=len(planner.nodes),
    )

# ── aggregation helper ─────────────────────────────────────────────────────

def _aggregate(results):
    """Return summary dict from a list of single-run dicts."""
    n = len(results)
    oks = [r for r in results if r["success"]]
    return dict(
        trials=n,
        successes=len(oks),
        success_rate=len(oks) / n * 100 if n else 0,
        mean_length=float(np.nanmean([r["path_length"] for r in oks])) if oks else float("nan"),
        std_length=float(np.nanstd([r["path_length"] for r in oks])) if oks else float("nan"),
        mean_time=float(np.mean([r["time_s"] for r in results])),
        std_time=float(np.std([r["time_s"] for r in results])),
        mean_smoothness=float(np.nanmean([r["smoothness"] for r in oks])) if oks else float("nan"),
    )

# ═══════════════════════════════════════════════════════════════════════════
#  PRM sweep
# ═══════════════════════════════════════════════════════════════════════════

def sweep_prm(env, start, goal, n_trials=5):
    n_samples_vals   = [100, 200, 400, 800]
    k_neighbors_vals = [5, 10, 15, 20]

    rows = []
    print("\n" + "=" * 65)
    print("  PRM Parameter Sweep")
    print("=" * 65)

    for ns in n_samples_vals:
        for kn in k_neighbors_vals:
            results = []
            for t in range(n_trials):
                seed = 7000 + ns + kn * 10 + t
                results.append(_run_prm(env, start, goal, seed, ns, kn))
            agg = _aggregate(results)
            agg["n_samples"] = ns
            agg["k_neighbors"] = kn
            rows.append(agg)
            print(f"  n_samples={ns:4d}  k={kn:2d}  "
                  f"success={agg['success_rate']:5.1f}%  "
                  f"len={agg['mean_length']:7.2f}  "
                  f"time={agg['mean_time']:.3f}s")

    # save CSV
    fp = os.path.join(OUT, "prm_sweep.csv")
    _write_csv(rows, fp)
    return rows


def plot_prm(rows):
    """
    Produce two figures for PRM:
      1. Line plots of success rate & path length vs n_samples (one curve
         per k_neighbors).
      2. Heatmaps of success rate & mean path length over the full
         (n_samples x k_neighbors) grid.
    """
    ns_vals = sorted(set(r["n_samples"] for r in rows))
    kn_vals = sorted(set(r["k_neighbors"] for r in rows))
    lookup = {(r["n_samples"], r["k_neighbors"]): r for r in rows}

    # ── Figure 1: line plots ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    cmap = plt.cm.viridis
    colours = [cmap(i / max(len(kn_vals)-1, 1)) for i in range(len(kn_vals))]

    for ci, kn in enumerate(kn_vals):
        sr   = [lookup[(ns, kn)]["success_rate"]  for ns in ns_vals]
        ml   = [lookup[(ns, kn)]["mean_length"]   for ns in ns_vals]
        mt   = [lookup[(ns, kn)]["mean_time"]     for ns in ns_vals]
        axes[0].plot(ns_vals, sr, "o-", color=colours[ci], label=f"k={kn}")
        axes[1].plot(ns_vals, ml, "s-", color=colours[ci], label=f"k={kn}")
        axes[2].plot(ns_vals, mt, "^-", color=colours[ci], label=f"k={kn}")

    axes[0].set_ylabel("Success Rate (%)")
    axes[0].set_ylim(0, 110)
    axes[1].set_ylabel("Mean Path Length (m)")
    axes[2].set_ylabel("Mean Time (s)")
    for ax in axes:
        ax.set_xlabel("n_samples")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    axes[0].set_title("PRM: Success Rate")
    axes[1].set_title("PRM: Path Length")
    axes[2].set_title("PRM: Computation Time")
    fig.suptitle("PRM -- Effect of n_samples and k_neighbors",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fname = os.path.join(OUT, "prm_line_plots.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  -> {fname}")

    # ── Figure 2: heatmaps ──────────────────────────────────────────────
    sr_grid = np.array([[lookup[(ns, kn)]["success_rate"]
                         for kn in kn_vals] for ns in ns_vals])
    ml_grid = np.array([[lookup[(ns, kn)]["mean_length"]
                         for kn in kn_vals] for ns in ns_vals])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, title, cbar_label, cmap_name in [
        (axes[0], sr_grid, "Success Rate (%)",    "Success %",       "YlGn"),
        (axes[1], ml_grid, "Mean Path Length (m)", "Path Length (m)", "YlOrRd_r"),
    ]:
        im = ax.imshow(data, aspect="auto", origin="lower",
                       cmap=cmap_name, interpolation="nearest")
        ax.set_xticks(range(len(kn_vals)))
        ax.set_xticklabels(kn_vals)
        ax.set_yticks(range(len(ns_vals)))
        ax.set_yticklabels(ns_vals)
        ax.set_xlabel("k_neighbors")
        ax.set_ylabel("n_samples")
        ax.set_title(title, fontweight="bold")
        # annotate cells
        for i in range(len(ns_vals)):
            for j in range(len(kn_vals)):
                val = data[i, j]
                txt = f"{val:.1f}" if not np.isnan(val) else "N/A"
                ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                        color="white" if val > np.nanmax(data)*0.6 else "black")
        fig.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)

    fig.suptitle("PRM -- Parameter Heatmaps", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fname = os.path.join(OUT, "prm_heatmaps.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  -> {fname}")


# ═══════════════════════════════════════════════════════════════════════════
#  RRT sweep
# ═══════════════════════════════════════════════════════════════════════════

def sweep_rrt(env, start, goal, n_trials=5):
    step_sizes         = [0.2, 0.5, 1.0]
    goal_sample_rates  = [0.0, 0.1, 0.3, 0.7]

    rows = []
    print("\n" + "=" * 65)
    print("  RRT Parameter Sweep")
    print("=" * 65)

    for ss in step_sizes:
        for gsr in goal_sample_rates:
            results = []
            for t in range(n_trials):
                seed = 8000 + int(ss*100) + int(gsr*100)*10 + t
                results.append(_run_rrt(env, start, goal, seed, ss, gsr))
            agg = _aggregate(results)
            agg["step_size"] = ss
            agg["goal_sample_rate"] = gsr
            rows.append(agg)
            print(f"  step={ss:.1f}  gsr={gsr:.1f}  "
                  f"success={agg['success_rate']:5.1f}%  "
                  f"len={agg['mean_length']:7.2f}  "
                  f"time={agg['mean_time']:.3f}s")

    fp = os.path.join(OUT, "rrt_sweep.csv")
    _write_csv(rows, fp)
    return rows


def plot_rrt(rows):
    """
    Produce two figures for RRT:
      1. Line plots of success rate & path length vs step_size (one curve
         per goal_sample_rate).
      2. Heatmaps of success rate & mean path length over the full
         (step_size x goal_sample_rate) grid.
    """
    ss_vals  = sorted(set(r["step_size"] for r in rows))
    gsr_vals = sorted(set(r["goal_sample_rate"] for r in rows))
    lookup   = {(r["step_size"], r["goal_sample_rate"]): r for r in rows}

    # ── Figure 1: line plots ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    cmap = plt.cm.plasma
    colours = [cmap(i / max(len(gsr_vals)-1, 1)) for i in range(len(gsr_vals))]

    for ci, gsr in enumerate(gsr_vals):
        sr = [lookup[(ss, gsr)]["success_rate"]  for ss in ss_vals]
        ml = [lookup[(ss, gsr)]["mean_length"]   for ss in ss_vals]
        mt = [lookup[(ss, gsr)]["mean_time"]     for ss in ss_vals]
        axes[0].plot(ss_vals, sr, "o-", color=colours[ci], label=f"gsr={gsr}")
        axes[1].plot(ss_vals, ml, "s-", color=colours[ci], label=f"gsr={gsr}")
        axes[2].plot(ss_vals, mt, "^-", color=colours[ci], label=f"gsr={gsr}")

    axes[0].set_ylabel("Success Rate (%)")
    axes[0].set_ylim(0, 110)
    axes[1].set_ylabel("Mean Path Length (m)")
    axes[2].set_ylabel("Mean Time (s)")
    for ax in axes:
        ax.set_xlabel("step_size")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    axes[0].set_title("RRT: Success Rate")
    axes[1].set_title("RRT: Path Length")
    axes[2].set_title("RRT: Computation Time")
    fig.suptitle("RRT -- Effect of step_size and goal_sample_rate",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fname = os.path.join(OUT, "rrt_line_plots.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  -> {fname}")

    # ── Figure 2: heatmaps ──────────────────────────────────────────────
    sr_grid = np.array([[lookup[(ss, gsr)]["success_rate"]
                         for gsr in gsr_vals] for ss in ss_vals])
    ml_grid = np.array([[lookup[(ss, gsr)]["mean_length"]
                         for gsr in gsr_vals] for ss in ss_vals])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, data, title, cbar_label, cmap_name in [
        (axes[0], sr_grid, "Success Rate (%)",    "Success %",       "YlGn"),
        (axes[1], ml_grid, "Mean Path Length (m)", "Path Length (m)", "YlOrRd_r"),
    ]:
        im = ax.imshow(data, aspect="auto", origin="lower",
                       cmap=cmap_name, interpolation="nearest")
        ax.set_xticks(range(len(gsr_vals)))
        ax.set_xticklabels(gsr_vals)
        ax.set_yticks(range(len(ss_vals)))
        ax.set_yticklabels(ss_vals)
        ax.set_xlabel("goal_sample_rate")
        ax.set_ylabel("step_size")
        ax.set_title(title, fontweight="bold")
        for i in range(len(ss_vals)):
            for j in range(len(gsr_vals)):
                val = data[i, j]
                txt = f"{val:.1f}" if not np.isnan(val) else "N/A"
                ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                        color="white" if val > np.nanmax(data)*0.6 else "black")
        fig.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)

    fig.suptitle("RRT -- Parameter Heatmaps", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fname = os.path.join(OUT, "rrt_heatmaps.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  -> {fname}")


# ── CSV writer ──────────────────────────────────────────────────────────────

def _write_csv(rows, filepath):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(filepath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {filepath}")


# ── summary table printer ──────────────────────────────────────────────────

def _print_table(rows, param_cols, title):
    print(f"\n{'='*75}")
    print(f"  {title}")
    print(f"{'='*75}")
    hdr_parts = [f"{c:>15s}" for c in param_cols]
    hdr = "".join(hdr_parts) + f"{'Success%':>10s}{'Len':>10s}{'Time(s)':>10s}{'Smooth':>10s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        vals = "".join(f"{str(r[c]):>15s}" for c in param_cols)
        sr  = f"{r['success_rate']:>9.1f}%"
        ml  = f"{r['mean_length']:>10.2f}" if not np.isnan(r['mean_length']) else f"{'N/A':>10s}"
        mt  = f"{r['mean_time']:>10.3f}"
        ms  = f"{r['mean_smoothness']:>10.4f}" if not np.isnan(r['mean_smoothness']) else f"{'N/A':>10s}"
        print(vals + sr + ml + mt + ms)
    print("=" * 75)


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    env   = Environment(seed=42)
    start = np.array([1.0, 1.0])
    goal  = np.array([17.0, 17.0])

    print("\n" + "=" * 75)
    print("  PRM & RRT Parameter Sensitivity Studies")
    print(f"  Environment: {len(env.landmarks)} obstacles, "
          f"start={start}, goal={goal}")
    print("=" * 75)

    # ── PRM ──
    prm_rows = sweep_prm(env, start, goal, n_trials=5)
    _print_table(prm_rows, ["n_samples", "k_neighbors"],
                 "PRM Sweep Summary")
    plot_prm(prm_rows)

    # ── RRT ──
    rrt_rows = sweep_rrt(env, start, goal, n_trials=5)
    _print_table(rrt_rows, ["step_size", "goal_sample_rate"],
                 "RRT Sweep Summary")
    plot_rrt(rrt_rows)

    print("\n" + "=" * 75)
    print(f"  All outputs in:  {OUT}/")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()

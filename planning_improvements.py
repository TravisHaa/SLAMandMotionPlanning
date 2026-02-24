#!/usr/bin/env python3
"""
Path post-processing improvements for PRM and RRT.

PRM improvement -- greedy shortcutting:
    Walk along the path and try to skip intermediate waypoints by connecting
    non-consecutive nodes with a straight collision-free segment.

RRT improvement -- random shortcutting (smoothing):
    Repeatedly pick two random waypoints on the path; if the straight line
    between them is collision-free, replace the sub-path with that segment.

Outputs (in  planning_improvements/ ):
    * combined_before_after.png   -- 2x2 panel (PRM / RRT) x (before / after)
    * improvement_metrics.csv     -- per-trial metrics for all four variants
    * improvement_comparison.png  -- bar charts comparing baseline vs improved
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
from matplotlib.patches import Circle

from environment import Environment
from motion_planning import PRM, RRT

# ── output directory ────────────────────────────────────────────────────────
OUT = "planning_improvements"
os.makedirs(OUT, exist_ok=True)

# ── colours ─────────────────────────────────────────────────────────────────
CLR_BEFORE = "#cc4444"
CLR_AFTER  = "#22aa22"
CLR_OBS    = "#888888"
CLR_PRM    = "#4c9fd1"
CLR_RRT    = "#e07b54"


# ═══════════════════════════════════════════════════════════════════════════
#  Post-processing algorithms
# ═══════════════════════════════════════════════════════════════════════════

def shortcut_path(path, env):
    """
    Greedy shortcutting for PRM paths.

    Starting from the first waypoint, try to directly connect to the farthest
    reachable waypoint (collision-free straight line).  This eliminates
    unnecessary detours while respecting obstacles.
    """
    if path is None or len(path) < 3:
        return path

    pts = list(path)
    shortened = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        # try to jump as far ahead as possible
        best = i + 1
        for j in range(len(pts) - 1, i + 1, -1):
            if env.is_path_collision_free(pts[i], pts[j]):
                best = j
                break
        shortened.append(pts[best])
        i = best

    return np.array(shortened)


def smooth_path(path, env, iterations=200):
    """
    Random shortcutting (smoothing) for RRT paths.

    Each iteration picks two random indices on the current path; if the
    straight segment between them is collision-free, the intermediate
    waypoints are removed.
    """
    if path is None or len(path) < 3:
        return path

    pts = list(path)
    for _ in range(iterations):
        if len(pts) < 3:
            break
        i = np.random.randint(0, len(pts) - 2)
        j = np.random.randint(i + 2, len(pts))
        if env.is_path_collision_free(pts[i], pts[j]):
            pts = pts[:i + 1] + pts[j:]

    return np.array(pts)


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

@contextlib.contextmanager
def _quiet():
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
    for i in range(1, len(p) - 1):
        v1, v2 = p[i] - p[i-1], p[i+1] - p[i]
        c = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-12),
                     -1, 1)
        angles.append(abs(np.arccos(c)))
    return float(np.mean(angles))


def _metrics(path, elapsed):
    ok = path is not None
    return dict(
        success=ok,
        path_length=_path_len(path) if ok else float("nan"),
        waypoints=len(path) if ok else 0,
        time_s=elapsed,
        smoothness=_smoothness(path) if ok else float("nan"),
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Single-run wrappers (return path + planner for visualisation)
# ═══════════════════════════════════════════════════════════════════════════

def run_prm_pair(env, start, goal, seed, n_samples=400, k_neighbors=12):
    """Run PRM, then shortcut.  Return (baseline_path, improved_path, prm, metrics_base, metrics_imp)."""
    np.random.seed(seed)
    prm = PRM(env, n_samples=n_samples, k_neighbors=k_neighbors)
    t0 = time.perf_counter()
    with _quiet():
        base_path = prm.plan(start, goal)
    t_base = time.perf_counter() - t0

    t0 = time.perf_counter()
    imp_path = shortcut_path(base_path, env) if base_path is not None else None
    t_imp = t_base + (time.perf_counter() - t0)

    return base_path, imp_path, prm, _metrics(base_path, t_base), _metrics(imp_path, t_imp)


def run_rrt_pair(env, start, goal, seed,
                 max_iter=3000, step_size=0.5, goal_sample_rate=0.15,
                 goal_tol=0.5, smooth_iters=200):
    """Run RRT, then smooth.  Return (baseline_path, improved_path, rrt, metrics_base, metrics_imp)."""
    np.random.seed(seed)
    rrt = RRT(env, max_iter=max_iter, step_size=step_size,
              goal_sample_rate=goal_sample_rate)
    t0 = time.perf_counter()
    with _quiet():
        base_path = rrt.plan(start, goal, goal_tolerance=goal_tol)
    t_base = time.perf_counter() - t0

    np.random.seed(seed + 99999)  # separate seed for smooth randomness
    t0 = time.perf_counter()
    imp_path = smooth_path(base_path, env, iterations=smooth_iters) if base_path is not None else None
    t_imp = t_base + (time.perf_counter() - t0)

    return base_path, imp_path, rrt, _metrics(base_path, t_base), _metrics(imp_path, t_imp)


# ═══════════════════════════════════════════════════════════════════════════
#  Visualisation
# ═══════════════════════════════════════════════════════════════════════════

def _draw_obs(ax, env):
    for pos in env.landmarks.values():
        ax.add_patch(Circle((pos[0], pos[1]), env.obstacle_radius,
                            color=CLR_OBS, alpha=0.45, linewidth=0))


def _setup(ax, env, title):
    ax.set_xlim(env.x_min, env.x_max)
    ax.set_ylim(env.y_min, env.y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.25)


def plot_before_after(env, start, goal,
                      prm_base, prm_imp, prm_obj,
                      rrt_base, rrt_imp, rrt_obj):
    """
    2-row x 2-col figure:
        row 0 = PRM  (before | after)
        row 1 = RRT  (before | after)
    Both paths drawn on each panel so the improvement is obvious.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # ── helper to draw one panel ──
    def _panel(ax, env, start, goal, base_path, imp_path, bg_struct, title, is_prm):
        _setup(ax, env, title)
        _draw_obs(ax, env)

        # faded roadmap / tree
        if is_prm and bg_struct is not None:
            nodes, edges = bg_struct.nodes, bg_struct.edges
            for i in range(len(nodes)):
                for j in edges[i]:
                    if j > i and j < len(nodes):
                        ax.plot([nodes[i][0], nodes[j][0]],
                                [nodes[i][1], nodes[j][1]],
                                "c-", lw=0.2, alpha=0.1)
        elif not is_prm and bg_struct is not None:
            for i, pidx in bg_struct.parents.items():
                if pidx is not None:
                    ax.plot([bg_struct.nodes[i][0], bg_struct.nodes[pidx][0]],
                            [bg_struct.nodes[i][1], bg_struct.nodes[pidx][1]],
                            "b-", lw=0.2, alpha=0.1)

        if base_path is not None:
            pa = np.array(base_path)
            ax.plot(pa[:, 0], pa[:, 1], color=CLR_BEFORE, lw=2.5, alpha=0.7,
                    label=f"Before ({_path_len(base_path):.1f} m, {len(base_path)} pts)",
                    zorder=5)
        if imp_path is not None:
            pa = np.array(imp_path)
            ax.plot(pa[:, 0], pa[:, 1], color=CLR_AFTER, lw=2.5,
                    label=f"After  ({_path_len(imp_path):.1f} m, {len(imp_path)} pts)",
                    zorder=6)

        ax.plot(*start, "ko", markersize=12, zorder=7)
        ax.plot(*start, "go", markersize=9, label="Start", zorder=8)
        ax.plot(*goal,  "r*", markersize=16, label="Goal",  zorder=8)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # row 0: PRM
    _panel(axes[0, 0], env, start, goal, prm_base, prm_imp, prm_obj,
           "PRM -- Before Shortcutting", is_prm=True)
    _panel(axes[0, 1], env, start, goal, prm_base, prm_imp, prm_obj,
           "PRM -- After Shortcutting", is_prm=True)

    # row 1: RRT
    _panel(axes[1, 0], env, start, goal, rrt_base, rrt_imp, rrt_obj,
           "RRT -- Before Smoothing", is_prm=False)
    _panel(axes[1, 1], env, start, goal, rrt_base, rrt_imp, rrt_obj,
           "RRT -- After Smoothing", is_prm=False)

    fig.suptitle("Path Improvements: Before vs After",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fname = os.path.join(OUT, "combined_before_after.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  -> {fname}")


def plot_metric_bars(records):
    """
    Grouped bar charts comparing baseline vs improved for both PRM and RRT
    across four metrics: path length, waypoints, time, smoothness.
    """
    groups = {
        "PRM baseline":  [r for r in records if r["label"] == "PRM baseline"],
        "PRM improved":  [r for r in records if r["label"] == "PRM improved"],
        "RRT baseline":  [r for r in records if r["label"] == "RRT baseline"],
        "RRT improved":  [r for r in records if r["label"] == "RRT improved"],
    }

    def _mean(grp, key):
        vals = [r[key] for r in grp if r["success"]]
        return float(np.mean(vals)) if vals else 0

    labels = list(groups.keys())
    x = np.arange(len(labels))
    width = 0.6

    metrics = [
        ("path_length", "Path Length (m)"),
        ("waypoints",   "Waypoints"),
        ("time_s",      "Time (s)"),
        ("smoothness",  "Smoothness (rad)"),
    ]

    colours = [CLR_PRM, CLR_PRM, CLR_RRT, CLR_RRT]
    hatches = ["", "//", "", "//"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (key, ylabel) in zip(axes.flat, metrics):
        vals = [_mean(groups[l], key) for l in labels]
        bars = ax.bar(x, vals, width, color=colours, edgecolor="black")
        # add hatch for improved variants
        for bar, h in zip(bars, hatches):
            bar.set_hatch(h)
        # annotate
        for bar, v in zip(bars, vals):
            fmt = f"{v:.2f}" if key != "waypoints" else f"{v:.0f}"
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(vals+[1]),
                    fmt, ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=10, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Baseline vs Improved -- Quantitative Comparison",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fname = os.path.join(OUT, "improvement_comparison.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  -> {fname}")


# ═══════════════════════════════════════════════════════════════════════════
#  Multi-trial experiment
# ═══════════════════════════════════════════════════════════════════════════

def run_trials(env, start, goal, n_trials=5):
    records = []
    print(f"\n  Running {n_trials} trials ...")

    for t in range(n_trials):
        seed = 9000 + t

        # PRM
        _, _, _, mb, mi = run_prm_pair(env, start, goal, seed)
        mb["label"] = "PRM baseline"
        mi["label"] = "PRM improved"
        records.extend([mb, mi])

        # RRT
        _, _, _, rb, ri = run_rrt_pair(env, start, goal, seed)
        rb["label"] = "RRT baseline"
        ri["label"] = "RRT improved"
        records.extend([rb, ri])

        print(f"    Trial {t+1}/{n_trials}  "
              f"PRM {mb['path_length']:.1f}->{mi['path_length']:.1f}  "
              f"RRT {rb['path_length']:.1f}->{ri['path_length']:.1f}")

    return records


def print_summary(records):
    labels = ["PRM baseline", "PRM improved", "RRT baseline", "RRT improved"]
    print(f"\n{'='*78}")
    print("  Baseline vs Improved -- Summary")
    print(f"{'='*78}")
    hdr = f"{'Variant':<18s}{'Succ%':>8s}{'Length':>10s}{'Wpts':>8s}{'Time(s)':>10s}{'Smooth':>10s}"
    print(hdr)
    print("-" * len(hdr))
    for lab in labels:
        sub = [r for r in records if r["label"] == lab]
        ok  = [r for r in sub if r["success"]]
        n   = len(sub)
        sr  = len(ok) / n * 100 if n else 0
        ml  = np.mean([r["path_length"] for r in ok]) if ok else float("nan")
        mw  = np.mean([r["waypoints"]   for r in ok]) if ok else 0
        mt  = np.mean([r["time_s"]      for r in sub])
        ms  = np.mean([r["smoothness"]  for r in ok]) if ok else float("nan")
        print(f"{lab:<18s}{sr:>7.1f}%{ml:>10.2f}{mw:>8.1f}{mt:>10.4f}{ms:>10.4f}")
    print("=" * 78)


def save_csv(records, fname):
    if not records:
        return
    keys = list(dict.fromkeys(k for r in records for k in r))
    fp = os.path.join(OUT, fname)
    with open(fp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)
    print(f"  -> {fp}")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    env   = Environment(seed=42)
    start = np.array([1.0, 1.0])
    goal  = np.array([17.0, 17.0])

    print("\n" + "=" * 78)
    print("  Path Improvement Study -- PRM Shortcutting & RRT Smoothing")
    print(f"  Environment: {len(env.landmarks)} obstacles, start={start}, goal={goal}")
    print("=" * 78)

    # ── single illustrated run for the before/after figure ──────────────
    print("\n  Single illustrated run (seed=42) ...")
    prm_base, prm_imp, prm_obj, _, _ = run_prm_pair(env, start, goal, seed=42)
    rrt_base, rrt_imp, rrt_obj, _, _ = run_rrt_pair(env, start, goal, seed=42)

    print(f"    PRM: {_path_len(prm_base):.2f} m -> {_path_len(prm_imp):.2f} m  "
          f"({len(prm_base)} -> {len(prm_imp)} waypoints)")
    print(f"    RRT: {_path_len(rrt_base):.2f} m -> {_path_len(rrt_imp):.2f} m  "
          f"({len(rrt_base)} -> {len(rrt_imp)} waypoints)")

    plot_before_after(env, start, goal,
                      prm_base, prm_imp, prm_obj,
                      rrt_base, rrt_imp, rrt_obj)

    # ── multi-trial quantitative comparison ─────────────────────────────
    records = run_trials(env, start, goal, n_trials=5)
    save_csv(records, "improvement_metrics.csv")
    print_summary(records)
    plot_metric_bars(records)

    print("\n" + "=" * 78)
    print(f"  Done.  All outputs in:  {OUT}/")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()

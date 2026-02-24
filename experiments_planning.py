#!/usr/bin/env python3
"""
Scenario-based comparison of PRM and RRT on two custom obstacle fields.

Scenario 1 -- Bottleneck: a horizontal wall with a single narrow gap forces
              both planners through the only feasible corridor.
Scenario 2 -- Zigzag:     three vertical baffle walls create an S-shaped
              passage requiring several tight turns.

For every (scenario x algorithm) combination the script saves:
    * roadmap / tree-growth visualisation
    * final-path visualisation
    * per-trial metrics CSV
    * comparison bar / box plots

Reuses Environment, PRM, and RRT from the existing codebase.
"""

import os
import sys
import csv
import time
import copy
import contextlib
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless runs
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from collections import defaultdict

from environment import Environment
from motion_planning import PRM, RRT

# ── output directory ────────────────────────────────────────────────────────
OUTPUT_DIR = "planning_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── colours used throughout ─────────────────────────────────────────────────
CLR_PRM  = "#4c9fd1"
CLR_RRT  = "#e07b54"
CLR_OBS  = "#888888"
CLR_PATH = "#22aa22"

# ═══════════════════════════════════════════════════════════════════════════
#  Custom-environment factories
# ═══════════════════════════════════════════════════════════════════════════

def _make_env_shell():
    """Return a bare Environment with empty landmarks (we fill them in)."""
    env = Environment(seed=42)
    env.landmarks = {}
    return env


def make_bottleneck_env():
    """
    Scenario 1 -- Bottleneck.

    A horizontal wall of tightly-spaced obstacles at y = 10 spans the full
    width of the arena.  A single gap near x = 10 is the only passage.
    A handful of extra obstacles above and below add clutter.
    """
    env = _make_env_shell()
    env.obstacle_radius = 0.8
    lid = 0

    # Horizontal wall at y = 10, spacing 1.5 m (edges overlap at r = 0.8)
    wall_y = 10.0
    for x in np.arange(-1.5, 21.0, 1.5):
        # Leave a gap centred at x ~ 9.75 (skip x = 9.0 and x = 10.5)
        if 8.5 < x < 11.0:
            continue
        env.landmarks[lid] = np.array([x, wall_y])
        lid += 1

    # Scatter a few obstacles above and below for visual interest
    extras = [
        (3.0, 4.0), (7.0, 5.5), (14.0, 4.5), (17.0, 6.0),
        (2.0, 15.0), (6.5, 16.0), (13.0, 14.5), (17.5, 15.5),
    ]
    for pos in extras:
        env.landmarks[lid] = np.array(pos)
        lid += 1

    return env


def make_zigzag_env():
    """
    Scenario 2 -- Zigzag (tight turns).

    Three vertical baffle walls at x = 5, 10, 15 force the path into an
    S-shaped route with at least four sharp direction changes.
      Wall A (x = 5):  runs from y = -1.5 to y = 13.5  -> gap at top
      Wall B (x = 10): runs from y =  6.5 to y = 21.0  -> gap at bottom
      Wall C (x = 15): runs from y = -1.5 to y = 13.5  -> gap at top
    """
    env = _make_env_shell()
    env.obstacle_radius = 0.8
    lid = 0
    spacing = 1.5

    # Wall A -- gap at top (y > ~14.3)
    for y in np.arange(-1.5, 14.0, spacing):
        env.landmarks[lid] = np.array([5.0, y])
        lid += 1

    # Wall B -- gap at bottom (y < ~5.7)
    for y in np.arange(6.5, 21.5, spacing):
        env.landmarks[lid] = np.array([10.0, y])
        lid += 1

    # Wall C -- gap at top (y > ~14.3)
    for y in np.arange(-1.5, 14.0, spacing):
        env.landmarks[lid] = np.array([15.0, y])
        lid += 1

    return env


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def compute_path_length(path):
    """Total Euclidean length of a waypoint sequence."""
    if path is None or len(path) < 2:
        return 0.0
    return float(sum(
        np.linalg.norm(path[i + 1] - path[i]) for i in range(len(path) - 1)
    ))


def compute_smoothness(path):
    """Mean absolute turning angle (rad); lower = smoother."""
    if path is None or len(path) < 3:
        return 0.0
    angles = []
    for i in range(1, len(path) - 1):
        v1 = path[i] - path[i - 1]
        v2 = path[i + 1] - path[i]
        cos_a = np.clip(
            np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12),
            -1.0, 1.0,
        )
        angles.append(abs(np.arccos(cos_a)))
    return float(np.mean(angles))


@contextlib.contextmanager
def suppress_stdout():
    """Silence print() calls from the planner internals."""
    old = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = old


# ═══════════════════════════════════════════════════════════════════════════
#  Visualisation helpers (save to file -- no GUI)
# ═══════════════════════════════════════════════════════════════════════════

def _draw_obstacles(ax, env):
    """Draw all circular obstacles onto an axes."""
    for pos in env.landmarks.values():
        ax.add_patch(Circle(
            (pos[0], pos[1]), env.obstacle_radius,
            color=CLR_OBS, alpha=0.45, linewidth=0,
        ))


def _setup_ax(ax, env, title):
    """Common axes setup."""
    ax.set_xlim(env.x_min, env.x_max)
    ax.set_ylim(env.y_min, env.y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.25)


def save_prm_roadmap(env, prm, start, goal, path, tag):
    """
    Save two PNGs for a PRM run:
      {tag}_prm_roadmap.png  -- full roadmap (nodes + edges)
      {tag}_prm_path.png     -- roadmap faded, final path highlighted
    """
    nodes = prm.nodes
    edges = prm.edges

    # ---- roadmap visualisation ----
    fig, ax = plt.subplots(figsize=(10, 9))
    _setup_ax(ax, env, f"{tag} -- PRM Roadmap")
    _draw_obstacles(ax, env)

    # Draw edges
    for i in range(len(nodes)):
        for j in edges[i]:
            if j > i and j < len(nodes):
                ax.plot([nodes[i][0], nodes[j][0]],
                        [nodes[i][1], nodes[j][1]],
                        "c-", linewidth=0.4, alpha=0.35)
    # Draw nodes
    if nodes:
        na = np.array(nodes)
        ax.plot(na[:, 0], na[:, 1], "c.", markersize=3, alpha=0.6,
                label=f"Roadmap nodes ({len(nodes)})")

    ax.plot(*start, "go", markersize=14, label="Start", zorder=7)
    ax.plot(*goal, "r*", markersize=18, label="Goal", zorder=7)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fname = os.path.join(OUTPUT_DIR, f"{tag}_prm_roadmap.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  -> {fname}")

    # ---- final-path visualisation ----
    fig, ax = plt.subplots(figsize=(10, 9))
    _setup_ax(ax, env, f"{tag} -- PRM Final Path")
    _draw_obstacles(ax, env)

    # Faded roadmap in background
    for i in range(len(nodes)):
        for j in edges[i]:
            if j > i and j < len(nodes):
                ax.plot([nodes[i][0], nodes[j][0]],
                        [nodes[i][1], nodes[j][1]],
                        "c-", linewidth=0.25, alpha=0.15)

    # Path
    if path is not None:
        pa = np.array(path)
        ax.plot(pa[:, 0], pa[:, 1], color=CLR_PATH, linewidth=3,
                label=f"Path (len={compute_path_length(path):.2f} m)", zorder=6)

    ax.plot(*start, "go", markersize=14, label="Start", zorder=7)
    ax.plot(*goal, "r*", markersize=18, label="Goal", zorder=7)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fname = os.path.join(OUTPUT_DIR, f"{tag}_prm_path.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  -> {fname}")


def save_rrt_tree(env, rrt, start, goal, path, tag):
    """
    Save two PNGs for an RRT run:
      {tag}_rrt_tree.png   -- full exploration tree
      {tag}_rrt_path.png   -- tree faded, final path highlighted
    """
    nodes = rrt.nodes
    parents = rrt.parents

    # ---- tree-growth visualisation ----
    fig, ax = plt.subplots(figsize=(10, 9))
    _setup_ax(ax, env, f"{tag} -- RRT Tree Growth")
    _draw_obstacles(ax, env)

    # Draw tree edges
    for i, pidx in parents.items():
        if pidx is not None:
            ax.plot([nodes[i][0], nodes[pidx][0]],
                    [nodes[i][1], nodes[pidx][1]],
                    "b-", linewidth=0.6, alpha=0.35)
    # Draw tree nodes
    if nodes:
        na = np.array(nodes)
        ax.plot(na[:, 0], na[:, 1], "b.", markersize=2, alpha=0.5,
                label=f"Tree nodes ({len(nodes)})")

    ax.plot(*start, "go", markersize=14, label="Start", zorder=7)
    ax.plot(*goal, "r*", markersize=18, label="Goal", zorder=7)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fname = os.path.join(OUTPUT_DIR, f"{tag}_rrt_tree.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  -> {fname}")

    # ---- final-path visualisation ----
    fig, ax = plt.subplots(figsize=(10, 9))
    _setup_ax(ax, env, f"{tag} -- RRT Final Path")
    _draw_obstacles(ax, env)

    # Faded tree in background
    for i, pidx in parents.items():
        if pidx is not None:
            ax.plot([nodes[i][0], nodes[pidx][0]],
                    [nodes[i][1], nodes[pidx][1]],
                    "b-", linewidth=0.3, alpha=0.12)

    # Path
    if path is not None:
        pa = np.array(path)
        ax.plot(pa[:, 0], pa[:, 1], color=CLR_PATH, linewidth=3,
                label=f"Path (len={compute_path_length(path):.2f} m)", zorder=6)

    ax.plot(*start, "go", markersize=14, label="Start", zorder=7)
    ax.plot(*goal, "r*", markersize=18, label="Goal", zorder=7)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fname = os.path.join(OUTPUT_DIR, f"{tag}_rrt_path.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  -> {fname}")


# ═══════════════════════════════════════════════════════════════════════════
#  Single-trial runners (return metrics dict + planner object + path)
# ═══════════════════════════════════════════════════════════════════════════

def run_prm(env, start, goal, seed, n_samples=400, k_neighbors=12):
    """Run one PRM trial; return (metrics_dict, prm_object, path)."""
    np.random.seed(seed)
    prm = PRM(env, n_samples=n_samples, k_neighbors=k_neighbors)
    t0 = time.perf_counter()
    with suppress_stdout():
        path = prm.plan(start, goal)
    elapsed = time.perf_counter() - t0
    ok = path is not None
    metrics = {
        "algorithm":     "PRM",
        "seed":          seed,
        "success":       ok,
        "path_length":   compute_path_length(path) if ok else float("nan"),
        "waypoints":     len(path) if ok else 0,
        "nodes_created": len(prm.nodes),
        "time_s":        elapsed,
        "smoothness":    compute_smoothness(path) if ok else float("nan"),
    }
    return metrics, prm, path


def run_rrt(env, start, goal, seed,
            max_iter=3000, step_size=0.5, goal_sample_rate=0.15,
            goal_tolerance=0.5):
    """Run one RRT trial; return (metrics_dict, rrt_object, path)."""
    np.random.seed(seed)
    rrt = RRT(env, max_iter=max_iter, step_size=step_size,
              goal_sample_rate=goal_sample_rate)
    t0 = time.perf_counter()
    with suppress_stdout():
        path = rrt.plan(start, goal, goal_tolerance=goal_tolerance)
    elapsed = time.perf_counter() - t0
    ok = path is not None
    metrics = {
        "algorithm":     "RRT",
        "seed":          seed,
        "success":       ok,
        "path_length":   compute_path_length(path) if ok else float("nan"),
        "waypoints":     len(path) if ok else 0,
        "nodes_created": len(rrt.nodes),
        "time_s":        elapsed,
        "smoothness":    compute_smoothness(path) if ok else float("nan"),
    }
    return metrics, rrt, path


# ═══════════════════════════════════════════════════════════════════════════
#  Multi-trial experiment
# ═══════════════════════════════════════════════════════════════════════════

def run_multi_trial(env, start, goal, n_trials, tag):
    """
    Run n_trials of both PRM and RRT, collect metrics, and return
    a list of record dicts.
    """
    records = []
    for t in range(n_trials):
        seed = 5000 + t
        m_prm, _, _ = run_prm(env, start, goal, seed)
        m_rrt, _, _ = run_rrt(env, start, goal, seed)
        m_prm["scenario"] = tag
        m_rrt["scenario"] = tag
        records.extend([m_prm, m_rrt])
        print(f"    Trial {t+1:3d}/{n_trials}  "
              f"PRM={'OK' if m_prm['success'] else 'FAIL':>4s} "
              f"len={m_prm['path_length']:7.2f}   "
              f"RRT={'OK' if m_rrt['success'] else 'FAIL':>4s} "
              f"len={m_rrt['path_length']:7.2f}")
    return records


# ═══════════════════════════════════════════════════════════════════════════
#  CSV / table / plot output
# ═══════════════════════════════════════════════════════════════════════════

def save_csv(records, filename):
    """Dump records to CSV."""
    if not records:
        return
    keys = list(dict.fromkeys(k for r in records for k in r))
    fp = os.path.join(OUTPUT_DIR, filename)
    with open(fp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)
    print(f"  -> {fp}")


def print_summary(records, heading):
    """Pretty-print a comparison table to stdout."""
    prm = [r for r in records if r["algorithm"] == "PRM"]
    rrt = [r for r in records if r["algorithm"] == "RRT"]

    def _s(grp):
        n = len(grp)
        ok = [r for r in grp if r["success"]]
        lens  = [r["path_length"] for r in ok]
        times = [r["time_s"]      for r in ok]
        smth  = [r["smoothness"]  for r in ok]
        nodes = [r["nodes_created"] for r in grp]
        return {
            "n": n, "ok": len(ok),
            "rate":  len(ok) / n * 100 if n else 0,
            "lm":   np.mean(lens)  if lens  else float("nan"),
            "ls":   np.std(lens)   if lens  else float("nan"),
            "tm":   np.mean(times) if times else float("nan"),
            "ts":   np.std(times)  if times else float("nan"),
            "sm":   np.mean(smth)  if smth  else float("nan"),
            "nm":   np.mean(nodes),
        }

    sp, sr = _s(prm), _s(rrt)

    print(f"\n{'='*65}")
    print(f"  {heading}")
    print(f"{'='*65}")
    h = f"{'Metric':<28s} {'PRM':>15s} {'RRT':>15s}"
    print(h)
    print("-" * len(h))
    print(f"{'Trials':<28s} {sp['n']:>15d} {sr['n']:>15d}")
    print(f"{'Successes':<28s} {sp['ok']:>15d} {sr['ok']:>15d}")
    print(f"{'Success rate (%)':<28s} {sp['rate']:>14.1f}% {sr['rate']:>14.1f}%")
    print(f"{'Mean path length (m)':<28s} {sp['lm']:>15.2f} {sr['lm']:>15.2f}")
    print(f"{'Std  path length (m)':<28s} {sp['ls']:>15.2f} {sr['ls']:>15.2f}")
    print(f"{'Mean time (s)':<28s} {sp['tm']:>15.4f} {sr['tm']:>15.4f}")
    print(f"{'Std  time (s)':<28s} {sp['ts']:>15.4f} {sr['ts']:>15.4f}")
    print(f"{'Mean smoothness (rad)':<28s} {sp['sm']:>15.4f} {sr['sm']:>15.4f}")
    print(f"{'Mean nodes created':<28s} {sp['nm']:>15.1f} {sr['nm']:>15.1f}")
    print("=" * 65)


def plot_comparison(records, tag):
    """
    Four-panel figure: success-rate bar, path-length box, time box,
    smoothness box.  Saved as {tag}_comparison.png.
    """
    prm = [r for r in records if r["algorithm"] == "PRM"]
    rrt = [r for r in records if r["algorithm"] == "RRT"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # -- success rate --
    pr = sum(r["success"] for r in prm) / max(len(prm), 1) * 100
    rr = sum(r["success"] for r in rrt) / max(len(rrt), 1) * 100
    bars = axes[0, 0].bar(["PRM", "RRT"], [pr, rr],
                           color=[CLR_PRM, CLR_RRT], edgecolor="black", width=0.5)
    for b, v in zip(bars, [pr, rr]):
        axes[0, 0].text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                        f"{v:.1f}%", ha="center", va="bottom", fontweight="bold")
    axes[0, 0].set_ylabel("Success rate (%)")
    axes[0, 0].set_ylim(0, 115)
    axes[0, 0].set_title("Success Rate")
    axes[0, 0].grid(axis="y", alpha=0.3)

    def _box(ax, prm_vals, rrt_vals, ylabel, title):
        if not prm_vals:
            prm_vals = [0]
        if not rrt_vals:
            rrt_vals = [0]
        bp = ax.boxplot([prm_vals, rrt_vals], tick_labels=["PRM", "RRT"],
                        patch_artist=True, widths=0.45,
                        medianprops=dict(color="black", linewidth=1.5))
        bp["boxes"][0].set_facecolor(CLR_PRM)
        bp["boxes"][1].set_facecolor(CLR_RRT)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    # -- path length --
    _box(axes[0, 1],
         [r["path_length"] for r in prm if r["success"]],
         [r["path_length"] for r in rrt if r["success"]],
         "Path length (m)", "Path Length")

    # -- time --
    _box(axes[1, 0],
         [r["time_s"] for r in prm if r["success"]],
         [r["time_s"] for r in rrt if r["success"]],
         "Time (s)", "Computation Time")

    # -- smoothness --
    _box(axes[1, 1],
         [r["smoothness"] for r in prm if r["success"]],
         [r["smoothness"] for r in rrt if r["success"]],
         "Mean turn angle (rad)", "Smoothness (lower = smoother)")

    fig.suptitle(f"{tag} -- PRM vs RRT Comparison",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fname = os.path.join(OUTPUT_DIR, f"{tag}_comparison.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  -> {fname}")


# ═══════════════════════════════════════════════════════════════════════════
#  Per-scenario driver
# ═══════════════════════════════════════════════════════════════════════════

def run_scenario(env, start, goal, tag, n_trials=20):
    """
    Full pipeline for one scenario:
      1. single PRM run  -> roadmap + path PNGs
      2. single RRT run  -> tree   + path PNGs
      3. multi-trial run -> CSV, table, comparison plot
    """
    print(f"\n{'#'*65}")
    print(f"  {tag}")
    print(f"  start={start}  goal={goal}  trials={n_trials}")
    print(f"{'#'*65}")

    # ---- single illustrated run (seed = 42) ----
    print("\n  [PRM] single illustrated run ...")
    m_prm, prm_obj, prm_path = run_prm(env, start, goal, seed=42)
    save_prm_roadmap(env, prm_obj, start, goal, prm_path, tag)
    status = "OK" if m_prm["success"] else "FAIL"
    print(f"    PRM: {status}, length={m_prm['path_length']:.2f} m, "
          f"nodes={m_prm['nodes_created']}, time={m_prm['time_s']:.4f} s")

    print("\n  [RRT] single illustrated run ...")
    m_rrt, rrt_obj, rrt_path = run_rrt(env, start, goal, seed=42)
    save_rrt_tree(env, rrt_obj, start, goal, rrt_path, tag)
    status = "OK" if m_rrt["success"] else "FAIL"
    print(f"    RRT: {status}, length={m_rrt['path_length']:.2f} m, "
          f"nodes={m_rrt['nodes_created']}, time={m_rrt['time_s']:.4f} s")

    # ---- multi-trial metrics ----
    print(f"\n  Running {n_trials} trials for each algorithm ...")
    records = run_multi_trial(env, start, goal, n_trials, tag)
    save_csv(records, f"{tag}_metrics.csv")
    print_summary(records, f"{tag} -- Multi-Trial Summary")
    plot_comparison(records, tag)

    return records, (prm_obj, prm_path), (rrt_obj, rrt_path)


# ═══════════════════════════════════════════════════════════════════════════
#  Combined 2x2 final-path figure
# ═══════════════════════════════════════════════════════════════════════════

def plot_combined_paths(scenarios):
    """
    Build a 2-row x 2-col figure:
        col 0 = PRM final path,  col 1 = RRT final path
        row 0 = Scenario 1,      row 1 = Scenario 2

    Parameters
    ----------
    scenarios : list of dicts, each containing
        env, start, goal, tag, prm_obj, prm_path, rrt_obj, rrt_path
    """
    n_rows = len(scenarios)
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, n_rows * 7.5))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, sc in enumerate(scenarios):
        env       = sc["env"]
        start     = sc["start"]
        goal      = sc["goal"]
        tag       = sc["tag"]
        prm_obj   = sc["prm_obj"]
        prm_path  = sc["prm_path"]
        rrt_obj   = sc["rrt_obj"]
        rrt_path  = sc["rrt_path"]

        # ── left column: PRM ──
        ax = axes[row, 0]
        _setup_ax(ax, env, f"{tag} -- PRM Path")
        _draw_obstacles(ax, env)
        # faded roadmap
        nodes, edges = prm_obj.nodes, prm_obj.edges
        for i in range(len(nodes)):
            for j in edges[i]:
                if j > i and j < len(nodes):
                    ax.plot([nodes[i][0], nodes[j][0]],
                            [nodes[i][1], nodes[j][1]],
                            "c-", linewidth=0.25, alpha=0.15)
        if prm_path is not None:
            pa = np.array(prm_path)
            ax.plot(pa[:, 0], pa[:, 1], color=CLR_PATH, linewidth=3,
                    label=f"Path ({compute_path_length(prm_path):.2f} m)",
                    zorder=6)
        ax.plot(*start, "go", markersize=13, label="Start", zorder=7)
        ax.plot(*goal,  "r*", markersize=17, label="Goal",  zorder=7)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

        # ── right column: RRT ──
        ax = axes[row, 1]
        _setup_ax(ax, env, f"{tag} -- RRT Path")
        _draw_obstacles(ax, env)
        # faded tree
        rnodes, rparents = rrt_obj.nodes, rrt_obj.parents
        for i, pidx in rparents.items():
            if pidx is not None:
                ax.plot([rnodes[i][0], rnodes[pidx][0]],
                        [rnodes[i][1], rnodes[pidx][1]],
                        "b-", linewidth=0.3, alpha=0.12)
        if rrt_path is not None:
            pa = np.array(rrt_path)
            ax.plot(pa[:, 0], pa[:, 1], color=CLR_PATH, linewidth=3,
                    label=f"Path ({compute_path_length(rrt_path):.2f} m)",
                    zorder=6)
        ax.plot(*start, "go", markersize=13, label="Start", zorder=7)
        ax.plot(*goal,  "r*", markersize=17, label="Goal",  zorder=7)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.suptitle("PRM vs RRT Final Paths -- Both Scenarios",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fname = os.path.join(OUTPUT_DIR, "combined_final_paths.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  -> {fname}")


# ═══════════════════════════════════════════════════════════════════════════
#  Grand comparison across both scenarios
# ═══════════════════════════════════════════════════════════════════════════

def plot_grand_comparison(all_records):
    """
    Grouped bar chart: success rate and mean path length for each
    (scenario x algorithm) combination.
    """
    scenarios = sorted(set(r["scenario"] for r in all_records))
    algos     = ["PRM", "RRT"]
    colours   = {"PRM": CLR_PRM, "RRT": CLR_RRT}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(scenarios))
    width = 0.35

    for ai, algo in enumerate(algos):
        rates, lengths = [], []
        for sc in scenarios:
            sub = [r for r in all_records
                   if r["scenario"] == sc and r["algorithm"] == algo]
            rates.append(sum(r["success"] for r in sub) / max(len(sub), 1) * 100)
            ok = [r["path_length"] for r in sub if r["success"]]
            lengths.append(np.mean(ok) if ok else 0)

        offset = -width / 2 + ai * width
        axes[0].bar(x + offset, rates, width, label=algo,
                    color=colours[algo], edgecolor="black")
        axes[1].bar(x + offset, lengths, width, label=algo,
                    color=colours[algo], edgecolor="black")

    for ax, ylabel, title in [
        (axes[0], "Success rate (%)", "Success Rate by Scenario"),
        (axes[1], "Mean path length (m)", "Mean Path Length by Scenario"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylim(0, 115)
    fig.suptitle("Grand Comparison -- Both Scenarios",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fname = os.path.join(OUTPUT_DIR, "grand_comparison.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  -> {fname}")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    N_TRIALS = 5

    print("\n" + "=" * 65)
    print("  PRM vs RRT -- Scenario-Based Planning Experiments")
    print("=" * 65)

    # ── Scenario 1: Bottleneck ──────────────────────────────────────────
    env1   = make_bottleneck_env()
    start1 = np.array([3.0, 2.0])
    goal1  = np.array([16.0, 18.0])
    recs1, (prm1, prm_path1), (rrt1, rrt_path1) = run_scenario(
        env1, start1, goal1, tag="Scenario1_Bottleneck", n_trials=N_TRIALS)

    # ── Scenario 2: Zigzag ──────────────────────────────────────────────
    env2   = make_zigzag_env()
    start2 = np.array([1.0, 10.0])
    goal2  = np.array([19.0, 10.0])
    recs2, (prm2, prm_path2), (rrt2, rrt_path2) = run_scenario(
        env2, start2, goal2, tag="Scenario2_Zigzag", n_trials=N_TRIALS)

    # ── Combined 2x2 final-path figure ──────────────────────────────────
    plot_combined_paths([
        dict(env=env1, start=start1, goal=goal1,
             tag="Scenario 1: Bottleneck",
             prm_obj=prm1, prm_path=prm_path1,
             rrt_obj=rrt1, rrt_path=rrt_path1),
        dict(env=env2, start=start2, goal=goal2,
             tag="Scenario 2: Zigzag",
             prm_obj=prm2, prm_path=prm_path2,
             rrt_obj=rrt2, rrt_path=rrt_path2),
    ])

    # ── Grand comparison ────────────────────────────────────────────────
    all_recs = recs1 + recs2
    save_csv(all_recs, "all_scenario_metrics.csv")
    plot_grand_comparison(all_recs)

    print("\n" + "=" * 65)
    print(f"  Done. All outputs in:  {OUTPUT_DIR}/")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()

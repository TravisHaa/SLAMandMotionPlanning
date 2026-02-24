#!/usr/bin/env python3
"""
Noise sensitivity experiment for EKF-SLAM using the landmark-hop controller.

Runs two experiment sweeps:
  A) Vary process noise R (low / medium / high) with measurement noise Q fixed.
  B) Vary measurement noise Q (low / medium / high) with process noise R fixed.

For each configuration the same landmark-hop control sequence is executed
(deterministic via a fixed random seed), and per-step metrics are collected:
  - Pose error (Euclidean distance between true and estimated position)
  - Pose uncertainty (sqrt of trace of the 2x2 pose covariance)
  - Average landmark uncertainty (mean trace of all landmark covariances)

Outputs:
  - noise_sensitivity_process.png   — 3-subplot figure for varying R
  - noise_sensitivity_measurement.png — 3-subplot figure for varying Q
  - noise_sensitivity_summary.csv   — one-row-per-experiment summary table
  - noise_sensitivity_combined.png  — combined 2×3 comparison figure
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless rendering
import matplotlib.pyplot as plt

from environment import Environment, RobotSimulator, wrap_angle
from slam import EKFSLAM

# test_EKF-SLAM.py has a hyphenated name so normal import doesn't work.
import importlib, importlib.util
_spec = importlib.util.spec_from_file_location(
    "test_ekf_slam",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_EKF-SLAM.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
LandmarkHopController = _mod.LandmarkHopController
collect_step_metrics = _mod.collect_step_metrics


# ===================================================================
# Experiment configuration
# ===================================================================

NUM_STEPS = 360
SEED = 42

# "Medium" baseline noise (same as the original demo)
R_MED = np.diag([0.1, 0.1, np.deg2rad(3.0)]) ** 2
Q_MED = np.diag([0.5, np.deg2rad(8.0)]) ** 2

# Process noise sweep (scale the diagonal entries)
R_LOW  = np.diag([0.02, 0.02, np.deg2rad(0.5)]) ** 2
R_HIGH = np.diag([0.3,  0.3,  np.deg2rad(10.0)]) ** 2

# Measurement noise sweep
Q_LOW  = np.diag([0.1,  np.deg2rad(2.0)]) ** 2
Q_HIGH = np.diag([1.5,  np.deg2rad(20.0)]) ** 2

PROCESS_SWEEP = [
    ("Low R",    R_LOW,  Q_MED),
    ("Med R",    R_MED,  Q_MED),
    ("High R",   R_HIGH, Q_MED),
]

MEASUREMENT_SWEEP = [
    ("Low Q",    R_MED,  Q_LOW),
    ("Med Q",    R_MED,  Q_MED),
    ("High Q",   R_MED,  Q_HIGH),
]


# ===================================================================
# Single-experiment runner
# ===================================================================

def run_experiment(label, R, Q, seed=SEED, num_steps=NUM_STEPS):
    """
    Execute one full EKF-SLAM run with the landmark-hop controller.

    Returns:
        metrics_log: list of per-step metric dicts
    """
    np.random.seed(seed)

    env = Environment(seed=seed)
    init_pose = [0.0, 0.0, 0.0]

    # Both the EKF and the simulator use the *same* R / Q so the filter
    # is tuned to the actual noise it experiences.
    ekf = EKFSLAM(init_pose=init_pose, R=R, Q=Q)
    simulator = RobotSimulator(
        environment=env,
        init_pose=init_pose,
        process_noise=R,
        measurement_noise=Q,
    )

    hop = LandmarkHopController(
        base_speed=1.0, gain=2.0, orbit_radius=3.5, orbit_steps=100,
        env_bounds=(env.x_min + 1, env.x_max - 1,
                    env.y_min + 1, env.y_max - 1),
    )

    metrics_log = []
    for t in range(num_steps):
        v, omega = hop(ekf, simulator.true_pose)
        control = (v, omega)
        measurements = simulator.step(control)
        ekf.step(control, measurements, dt=simulator.dt)
        step_m = collect_step_metrics(ekf, simulator, env, t, measurements)
        metrics_log.append(step_m)

    print(f"  [{label}] done — final pose err {metrics_log[-1]['pose_error_m']:.4f} m, "
          f"avg LM unc {metrics_log[-1]['avg_landmark_uncertainty']:.4f}")
    return metrics_log


# ===================================================================
# Plotting helpers
# ===================================================================

COLORS = ["#2196F3", "#4CAF50", "#F44336"]  # blue, green, red for low/med/high


def plot_sweep(sweep_results, sweep_name, filename):
    """
    Create a 3-subplot figure comparing low / medium / high noise for
    one sweep (either process or measurement).

    sweep_results: list of (label, metrics_log) tuples
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"EKF-SLAM Noise Sensitivity — {sweep_name}",
                 fontsize=14, fontweight="bold")

    for idx, (label, mlog) in enumerate(sweep_results):
        steps = [m["step"] for m in mlog]
        pose_err = [m["pose_error_m"] for m in mlog]
        pose_unc = [m["pose_uncertainty"] for m in mlog]
        avg_lm   = [m["avg_landmark_uncertainty"] for m in mlog]
        c = COLORS[idx]

        axes[0].plot(steps, pose_err, color=c, linewidth=1.2, label=label)
        axes[1].plot(steps, pose_unc, color=c, linewidth=1.2, label=label)
        axes[2].plot(steps, avg_lm,   color=c, linewidth=1.2, label=label)

    axes[0].set_ylabel("Pose Error [m]")
    axes[0].set_title("Position Error  (‖true − est‖)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("Pose Uncertainty [m]")
    axes[1].set_title("Pose Uncertainty  (√trace of pose covariance)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_ylabel("Avg LM Uncertainty")
    axes[2].set_title("Average Landmark Uncertainty  (mean trace of LM covariances)")
    axes[2].set_xlabel("Time Step")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"  Saved {filename}")


def plot_combined(process_results, measurement_results, filename):
    """
    2×3 combined figure — left column: varying R, right column: varying Q.
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 11), sharex=True)
    fig.suptitle("EKF-SLAM Noise Sensitivity — Landmark Hop Controller",
                 fontsize=15, fontweight="bold")

    titles_y = [
        ("Pose Error [m]", "Position Error"),
        ("Pose Uncertainty [m]", "Pose Uncertainty"),
        ("Avg LM Uncertainty", "Avg Landmark Uncertainty"),
    ]
    keys = ["pose_error_m", "pose_uncertainty", "avg_landmark_uncertainty"]

    for col, (sweep_label, results) in enumerate([
        ("Varying Process Noise R", process_results),
        ("Varying Measurement Noise Q", measurement_results),
    ]):
        for row in range(3):
            ax = axes[row, col]
            ylabel, title = titles_y[row]
            key = keys[row]
            for idx, (label, mlog) in enumerate(results):
                steps = [m["step"] for m in mlog]
                vals  = [m[key] for m in mlog]
                ax.plot(steps, vals, color=COLORS[idx], linewidth=1.2, label=label)
            ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(f"{sweep_label}\n{title}", fontsize=11)
            else:
                ax.set_title(title, fontsize=10)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
            if row == 2:
                ax.set_xlabel("Time Step")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"  Saved {filename}")


def build_summary_table(all_results, csv_path):
    """
    Write a summary CSV with one row per experiment configuration,
    reporting mean / final values of each key metric.
    """
    rows = []
    for label, mlog in all_results:
        pose_errors = [m["pose_error_m"] for m in mlog]
        pose_uncs   = [m["pose_uncertainty"] for m in mlog]
        avg_lm_uncs = [m["avg_landmark_uncertainty"] for m in mlog]

        rows.append({
            "config":                label,
            "mean_pose_error_m":     round(np.mean(pose_errors), 4),
            "final_pose_error_m":    round(pose_errors[-1], 4),
            "max_pose_error_m":      round(np.max(pose_errors), 4),
            "mean_pose_uncertainty": round(np.mean(pose_uncs), 4),
            "final_pose_uncertainty":round(pose_uncs[-1], 4),
            "mean_avg_lm_unc":       round(np.mean(avg_lm_uncs), 4),
            "final_avg_lm_unc":      round(avg_lm_uncs[-1], 4),
            "final_n_landmarks":     mlog[-1]["n_landmarks"],
        })

    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Summary table saved to {csv_path}")
    return rows


def print_summary_table(rows):
    """Pretty-print the summary table to the console."""
    print("\n" + "=" * 100)
    print("  NOISE SENSITIVITY — SUMMARY TABLE")
    print("=" * 100)
    header = (f"{'Config':<12s} {'MeanPosErr':>11s} {'FinalPosErr':>12s} "
              f"{'MaxPosErr':>10s} {'MeanPoseUnc':>12s} {'FinalPoseUnc':>13s} "
              f"{'MeanAvgLMUnc':>13s} {'FinalAvgLMUnc':>14s} {'#LMs':>5s}")
    print(header)
    print("-" * 100)
    for r in rows:
        print(f"{r['config']:<12s} "
              f"{r['mean_pose_error_m']:>11.4f} "
              f"{r['final_pose_error_m']:>12.4f} "
              f"{r['max_pose_error_m']:>10.4f} "
              f"{r['mean_pose_uncertainty']:>12.4f} "
              f"{r['final_pose_uncertainty']:>13.4f} "
              f"{r['mean_avg_lm_unc']:>13.4f} "
              f"{r['final_avg_lm_unc']:>14.4f} "
              f"{r['final_n_landmarks']:>5d}")
    print("=" * 100)


# ===================================================================
# Main
# ===================================================================

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print("\n" + "=" * 60)
    print("  EKF-SLAM NOISE SENSITIVITY EXPERIMENT")
    print("  Controller: Landmark Hop")
    print("=" * 60)

    # ── Sweep A: Varying process noise R ──────────────────────────
    print("\n▸ Sweep A — Varying Process Noise R (Q fixed at medium)")
    process_results = []
    for label, R, Q in PROCESS_SWEEP:
        mlog = run_experiment(label, R, Q)
        process_results.append((label, mlog))

    plot_sweep(process_results, "Varying Process Noise R (Q fixed)",
               os.path.join(base_dir, "noise_sensitivity_process.png"))

    # ── Sweep B: Varying measurement noise Q ──────────────────────
    print("\n▸ Sweep B — Varying Measurement Noise Q (R fixed at medium)")
    measurement_results = []
    for label, R, Q in MEASUREMENT_SWEEP:
        mlog = run_experiment(label, R, Q)
        measurement_results.append((label, mlog))

    plot_sweep(measurement_results, "Varying Measurement Noise Q (R fixed)",
               os.path.join(base_dir, "noise_sensitivity_measurement.png"))

    # ── Combined figure ───────────────────────────────────────────
    plot_combined(process_results, measurement_results,
                  os.path.join(base_dir, "noise_sensitivity_combined.png"))

    # ── Summary table ─────────────────────────────────────────────
    all_results = process_results + measurement_results
    csv_path = os.path.join(base_dir, "noise_sensitivity_summary.csv")
    rows = build_summary_table(all_results, csv_path)
    print_summary_table(rows)

    print("\nDone. Figures and CSV written to the project directory.\n")


if __name__ == "__main__":
    main()

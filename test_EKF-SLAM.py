#!/usr/bin/env python3
"""
Demonstration of uncertainty visualization in EKF-SLAM.
Shows how uncertainty evolves over time as landmarks are observed.
"""

import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from environment import Environment, RobotSimulator, wrap_angle
from slam import EKFSLAM, get_covariance_ellipse


def visualize_uncertainty_evolution(ax, env, ekf, true_path, current_pose, step, total_steps):
    """Visualize SLAM with focus on uncertainty."""
    ax.clear()
    ax.set_title(f"EKF-SLAM Uncertainty Evolution - Step {step}/{total_steps}", 
                 fontsize=14, fontweight='bold')
    ax.set_xlim(env.x_min, env.x_max)
    ax.set_ylim(env.y_min, env.y_max)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Draw true obstacles (very faded)
    for lm_id, lm_pos in env.landmarks.items():
        circle = Circle((lm_pos[0], lm_pos[1]), env.obstacle_radius, 
                       color='gray', alpha=0.1)
        ax.add_patch(circle)
        ax.plot(lm_pos[0], lm_pos[1], 'k+', markersize=8, alpha=0.3)
    
    # Get estimated landmarks and their covariances
    est_landmarks = ekf.get_estimated_landmarks()
    landmark_covs = ekf.get_landmark_covariances()
    
    # Visualize landmarks with uncertainty
    max_uncertainty = 0
    min_uncertainty = float('inf')
    
    for lm_id, lm_pos in est_landmarks.items():
        if lm_id not in landmark_covs:
            continue
            
        cov = landmark_covs[lm_id]
        
        # Calculate uncertainty magnitude (trace of covariance)
        uncertainty = np.trace(cov)
        max_uncertainty = max(max_uncertainty, uncertainty)
        min_uncertainty = min(min_uncertainty, uncertainty)
        
        # Color code by uncertainty (blue = low, red = high)
        if max_uncertainty > min_uncertainty:
            normalized_unc = (uncertainty - min_uncertainty) / (max_uncertainty - min_uncertainty)
        else:
            normalized_unc = 0
        color = plt.cm.RdYlBu_r(normalized_unc)
        
        # Draw landmark estimate
        ax.plot(lm_pos[0], lm_pos[1], 'o', color=color, markersize=10, 
               markeredgecolor='black', markeredgewidth=1, zorder=5)
        
        # Draw 1-sigma ellipse (68% confidence)
        width_1, height_1, angle = get_covariance_ellipse(cov, n_std=1.0)
        ellipse_1 = Ellipse((lm_pos[0], lm_pos[1]), width_1, height_1, angle=angle,
                           facecolor=color, edgecolor='black', linewidth=1,
                           alpha=0.3, zorder=4)
        ax.add_patch(ellipse_1)
        
        # Draw 2-sigma ellipse (95% confidence)
        width_2, height_2, angle = get_covariance_ellipse(cov, n_std=2.0)
        ellipse_2 = Ellipse((lm_pos[0], lm_pos[1]), width_2, height_2, angle=angle,
                           facecolor='none', edgecolor=color, linewidth=2,
                           linestyle='--', alpha=0.7, zorder=4)
        ax.add_patch(ellipse_2)
        
        # Label
        ax.text(lm_pos[0] + 0.4, lm_pos[1] + 0.4, f"L{lm_id}", 
               fontsize=7, color='black', fontweight='bold')
    
    # Draw paths (faded)
    if len(true_path) > 1:
        ax.plot(true_path[:, 0], true_path[:, 1], 'k-', 
               linewidth=1, alpha=0.2, label='True path')
    
    est_path = ekf.get_estimated_path()
    if len(est_path) > 1:
        ax.plot(est_path[:, 0], est_path[:, 1], 'b--', 
               linewidth=1, alpha=0.3, label='Est. path')
    
    # Draw current robot pose (true)
    x_r, y_r, th_r = current_pose
    ax.plot(x_r, y_r, 'ko', markersize=10, label='True robot', zorder=6)
    ax.arrow(x_r, y_r, 0.8 * np.cos(th_r), 0.8 * np.sin(th_r),
             head_width=0.3, length_includes_head=True, color='black', 
             linewidth=2, zorder=6)
    
    # Draw current robot pose (estimated) with uncertainty
    x_e, y_e, th_e = ekf.get_current_pose()
    ax.plot(x_e, y_e, 'bo', markersize=10, label='Est. robot', zorder=6)
    ax.arrow(x_e, y_e, 0.8 * np.cos(th_e), 0.8 * np.sin(th_e),
             head_width=0.3, length_includes_head=True, color='blue', 
             linewidth=2, zorder=6)
    
    # Draw robot pose uncertainty ellipses
    pose_cov = ekf.get_pose_covariance()
    
    # 1-sigma
    width_1, height_1, angle = get_covariance_ellipse(pose_cov, n_std=1.0)
    pose_ellipse_1 = Ellipse((x_e, y_e), width_1, height_1, angle=angle,
                            facecolor='blue', edgecolor='darkblue', linewidth=1,
                            alpha=0.3, zorder=5, label='Robot 1σ')
    ax.add_patch(pose_ellipse_1)
    
    # 2-sigma
    width_2, height_2, angle = get_covariance_ellipse(pose_cov, n_std=2.0)
    pose_ellipse_2 = Ellipse((x_e, y_e), width_2, height_2, angle=angle,
                            facecolor='none', edgecolor='blue', linewidth=2,
                            linestyle='--', alpha=0.7, zorder=5, label='Robot 2σ')
    ax.add_patch(pose_ellipse_2)
    
    # Info box with statistics
    pose_error = np.linalg.norm([x_r - x_e, y_r - y_e])
    pose_uncertainty = np.sqrt(np.trace(pose_cov))
    
    if len(est_landmarks) > 0:
        avg_landmark_unc = np.mean([np.trace(landmark_covs[lm_id]) 
                                    for lm_id in landmark_covs])
        max_landmark_unc = np.max([np.trace(landmark_covs[lm_id]) 
                                   for lm_id in landmark_covs]) if landmark_covs else 0
    else:
        avg_landmark_unc = 0
        max_landmark_unc = 0
    
    info_text = (f"Landmarks: {len(est_landmarks)}/{len(env.landmarks)}\n"
                f"Pose error: {pose_error:.3f}m\n"
                f"Pose unc: {pose_uncertainty:.3f}m\n"
                f"Avg LM unc: {avg_landmark_unc:.3f}\n"
                f"Max LM unc: {max_landmark_unc:.3f}")
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    


def collect_step_metrics(ekf, simulator, env, step, measurements):
    """
    Compute numerical metrics for the current EKF-SLAM state.

    Returns a dict with pose errors, uncertainties, landmark stats, etc.
    """
    # --- Robot pose error ---
    true_x, true_y, true_th = simulator.true_pose
    est_x, est_y, est_th = ekf.get_current_pose()

    pos_error = np.linalg.norm([true_x - est_x, true_y - est_y])
    heading_error = abs(wrap_angle(true_th - est_th))

    # --- Robot pose uncertainty (sqrt of trace of 2x2 position cov) ---
    pose_cov = ekf.get_pose_covariance()
    pose_uncertainty = np.sqrt(np.trace(pose_cov))

    # --- Landmark statistics ---
    est_landmarks = ekf.get_estimated_landmarks()
    landmark_covs = ekf.get_landmark_covariances()
    n_landmarks = len(est_landmarks)

    # Average / max landmark uncertainty (trace of 2x2 cov)
    if n_landmarks > 0:
        lm_traces = [np.trace(landmark_covs[lid]) for lid in landmark_covs]
        avg_lm_unc = float(np.mean(lm_traces))
        max_lm_unc = float(np.max(lm_traces))
    else:
        avg_lm_unc = 0.0
        max_lm_unc = 0.0

    # Average landmark position error against ground truth
    lm_errors = []
    for lm_id, est_pos in est_landmarks.items():
        if lm_id in env.landmarks:
            true_pos = env.landmarks[lm_id]
            lm_errors.append(np.linalg.norm(est_pos - true_pos))
    avg_lm_error = float(np.mean(lm_errors)) if lm_errors else 0.0
    max_lm_error = float(np.max(lm_errors)) if lm_errors else 0.0

    # NEES (Normalized Estimation Error Squared) for robot pose — a
    # consistency check: values near the state dimension (2) indicate a
    # well-calibrated filter.
    pose_err_vec = np.array([true_x - est_x, true_y - est_y])
    try:
        nees_pose = float(pose_err_vec @ np.linalg.inv(pose_cov) @ pose_err_vec)
    except np.linalg.LinAlgError:
        nees_pose = float('nan')

    return {
        "step": step,
        "n_measurements": len(measurements),
        "n_landmarks": n_landmarks,
        "pose_error_m": round(pos_error, 6),
        "heading_error_rad": round(heading_error, 6),
        "pose_uncertainty": round(pose_uncertainty, 6),
        "nees_pose": round(nees_pose, 6),
        "avg_landmark_error_m": round(avg_lm_error, 6),
        "max_landmark_error_m": round(max_lm_error, 6),
        "avg_landmark_uncertainty": round(avg_lm_unc, 6),
        "max_landmark_uncertainty": round(max_lm_unc, 6),
    }


def save_metrics_csv(metrics_log, filepath):
    """Write the list of per-step metric dicts to a CSV file."""
    if not metrics_log:
        return
    fieldnames = list(metrics_log[0].keys())
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_log)
    print(f"  Metrics saved to {filepath}")


def print_final_summary(metrics_log, ekf, env):
    """Print a comprehensive summary table of final EKF-SLAM performance."""
    # Convert lists for easy stats
    pose_errors = [m["pose_error_m"] for m in metrics_log]
    heading_errors = [m["heading_error_rad"] for m in metrics_log]
    nees_values = [m["nees_pose"] for m in metrics_log if not np.isnan(m["nees_pose"])]

    print("\n" + "=" * 60)
    print("  NUMERICAL METRICS SUMMARY")
    print("=" * 60)

    # Robot pose accuracy
    print("\n  Robot Pose Error (metres):")
    print(f"    Mean : {np.mean(pose_errors):.4f}")
    print(f"    Std  : {np.std(pose_errors):.4f}")
    print(f"    Max  : {np.max(pose_errors):.4f}")
    print(f"    Final: {pose_errors[-1]:.4f}")

    print(f"\n  Heading Error (rad):")
    print(f"    Mean : {np.mean(heading_errors):.4f}")
    print(f"    Max  : {np.max(heading_errors):.4f}")
    print(f"    Final: {heading_errors[-1]:.4f}")

    # NEES consistency
    if nees_values:
        print(f"\n  NEES (pose, ideal ~2.0):")
        print(f"    Mean : {np.mean(nees_values):.4f}")
        print(f"    Max  : {np.max(nees_values):.4f}")

    # Per-landmark breakdown
    est_landmarks = ekf.get_estimated_landmarks()
    landmark_covs = ekf.get_landmark_covariances()
    if est_landmarks:
        print(f"\n  Per-Landmark Errors ({len(est_landmarks)}/{len(env.landmarks)} observed):")
        print(f"    {'ID':>4s}  {'PosErr(m)':>10s}  {'Trace(cov)':>10s}")
        print(f"    {'----':>4s}  {'----------':>10s}  {'----------':>10s}")
        for lm_id in sorted(est_landmarks.keys()):
            est_pos = est_landmarks[lm_id]
            true_pos = env.landmarks[lm_id]
            err = np.linalg.norm(est_pos - true_pos)
            tr = np.trace(landmark_covs[lm_id])
            print(f"    {lm_id:>4d}  {err:>10.4f}  {tr:>10.4f}")

    print("=" * 60)


def plot_final_slam_state(env, ekf, simulator):
    """
    Static summary figure: true vs estimated robot trajectory and
    all landmark estimates with 1-sigma / 2-sigma uncertainty ellipses.
    """
    fig, ax = plt.subplots(figsize=(14, 11))
    ax.set_title("EKF-SLAM — Final State: Trajectories & Landmark Estimates",
                 fontsize=14, fontweight='bold')
    ax.set_xlim(env.x_min, env.x_max)
    ax.set_ylim(env.y_min, env.y_max)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # --- True landmark positions (ground truth) ---
    for lm_id, lm_pos in env.landmarks.items():
        circle = Circle((lm_pos[0], lm_pos[1]), env.obstacle_radius,
                         color='gray', alpha=0.15)
        ax.add_patch(circle)
        ax.plot(lm_pos[0], lm_pos[1], 'k+', markersize=10, alpha=0.5)

    # --- Estimated landmarks with covariance ellipses ---
    est_landmarks = ekf.get_estimated_landmarks()
    landmark_covs = ekf.get_landmark_covariances()

    for lm_id, lm_pos in est_landmarks.items():
        if lm_id not in landmark_covs:
            continue
        cov = landmark_covs[lm_id]

        # Landmark marker
        ax.plot(lm_pos[0], lm_pos[1], 'o', color='red', markersize=8,
                markeredgecolor='black', markeredgewidth=0.8, zorder=5)

        # 1-sigma ellipse (68 % confidence)
        w1, h1, angle = get_covariance_ellipse(cov, n_std=1.0)
        ell_1 = Ellipse((lm_pos[0], lm_pos[1]), w1, h1, angle=angle,
                         facecolor='salmon', edgecolor='red', linewidth=1,
                         alpha=0.25, zorder=4)
        ax.add_patch(ell_1)

        # 2-sigma ellipse (95 % confidence)
        w2, h2, angle = get_covariance_ellipse(cov, n_std=2.0)
        ell_2 = Ellipse((lm_pos[0], lm_pos[1]), w2, h2, angle=angle,
                         facecolor='none', edgecolor='red', linewidth=1.5,
                         linestyle='--', alpha=0.6, zorder=4)
        ax.add_patch(ell_2)

        # Label each landmark
        ax.text(lm_pos[0] + 0.4, lm_pos[1] + 0.4, f"L{lm_id}",
                fontsize=7, color='black', fontweight='bold')

    # --- True vs estimated robot trajectories ---
    true_path = simulator.get_true_path()
    est_path = ekf.get_estimated_path()

    if len(true_path) > 1:
        ax.plot(true_path[:, 0], true_path[:, 1], 'k-',
                linewidth=1.2, label='True trajectory')
    if len(est_path) > 1:
        ax.plot(est_path[:, 0], est_path[:, 1], 'b--',
                linewidth=1.2, label='Estimated trajectory')

    # --- Robot pose uncertainty ellipse at final position ---
    x_e, y_e, th_e = ekf.get_current_pose()
    pose_cov = ekf.get_pose_covariance()
    w_p, h_p, a_p = get_covariance_ellipse(pose_cov, n_std=2.0)
    pose_ell = Ellipse((x_e, y_e), w_p, h_p, angle=a_p,
                        facecolor='blue', edgecolor='darkblue',
                        linewidth=1.5, alpha=0.2, zorder=5,
                        label='Robot 2σ')
    ax.add_patch(pose_ell)

    # Start / end markers
    ax.plot(true_path[0, 0], true_path[0, 1], 'gs', markersize=12,
            label='Start', zorder=6)
    ax.plot(true_path[-1, 0], true_path[-1, 1], 'ko', markersize=10,
            label='True end', zorder=6)
    ax.plot(est_path[-1, 0], est_path[-1, 1], 'bo', markersize=10,
            label='Est. end', zorder=6)

    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout()
    return fig


def plot_metrics_over_time(metrics_log):
    """
    Time-series subplots of key EKF-SLAM metrics:
      1. Pose error (metres)
      2. Pose uncertainty (sqrt of trace of pose covariance)
      3. Average landmark uncertainty (mean trace of landmark covariances)
    """
    steps = [m["step"] for m in metrics_log]
    pose_errors = [m["pose_error_m"] for m in metrics_log]
    pose_uncs = [m["pose_uncertainty"] for m in metrics_log]
    avg_lm_uncs = [m["avg_landmark_uncertainty"] for m in metrics_log]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("EKF-SLAM Metrics Over Time", fontsize=14, fontweight='bold')

    # --- Subplot 1: Pose error ---
    axes[0].plot(steps, pose_errors, 'r-', linewidth=1)
    axes[0].set_ylabel("Pose Error [m]")
    axes[0].set_title("Position Error  (||true_xy − est_xy||)")
    axes[0].grid(True, alpha=0.3)

    # --- Subplot 2: Pose uncertainty ---
    axes[1].plot(steps, pose_uncs, 'b-', linewidth=1)
    axes[1].set_ylabel("Pose Uncertainty [m]")
    axes[1].set_title("Pose Uncertainty  (√trace of pose covariance)")
    axes[1].grid(True, alpha=0.3)

    # --- Subplot 3: Average landmark uncertainty ---
    axes[2].plot(steps, avg_lm_uncs, 'g-', linewidth=1)
    axes[2].set_ylabel("Avg LM Uncertainty")
    axes[2].set_title("Average Landmark Uncertainty  (mean trace of LM covariances)")
    axes[2].set_xlabel("Time Step")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


class LandmarkHopController:
    """
    Explore-and-wander controller.

    Behaviour cycle:
      1. WANDER  — drive in a random direction until a previously
                   unvisited landmark is detected by the EKF.
      2. SEEK    — steer toward that landmark.
      3. EXPLORE — orbit the landmark to gather close-range
                   measurements and reduce its covariance.
      4. Return to WANDER with a fresh random heading.

    Args:
        base_speed:   forward velocity (m/s)
        gain:         proportional heading-correction gain
        orbit_radius: desired distance to maintain while orbiting (m)
        orbit_steps:  how many time-steps to orbit before moving on
        env_bounds:   (xmin, xmax, ymin, ymax) used for wall avoidance
    """

    def __init__(self, base_speed=1.0, gain=2.0, orbit_radius=3.5,
                 orbit_steps=100, env_bounds=(0.0, 18.0, 0.0, 18.0)):
        self.base_speed = base_speed
        self.gain = gain
        self.orbit_radius = orbit_radius
        self.orbit_steps = orbit_steps
        self.env_bounds = env_bounds

        # Internal state machine
        self.mode = "wander"               # "wander" | "seek" | "explore"
        self.target_id = None              # id of the landmark being targeted
        self.wander_heading = np.random.uniform(-np.pi, np.pi)
        self.explore_counter = 0           # steps spent orbiting so far
        self.visited = set()               # landmark ids already explored
        self._known_ids = set()            # landmarks the EKF knew last call

    # ------------------------------------------------------------------
    # Public interface — matches the signature expected by the main loop
    # ------------------------------------------------------------------
    def __call__(self, ekf, robot_pose):
        """
        Compute (v, omega) for the current time-step.

        Args:
            ekf:        EKFSLAM instance (read-only access)
            robot_pose: current true pose [x, y, theta]

        Returns:
            (v, omega) control tuple
        """
        x, y, theta = robot_pose
        est_landmarks = ekf.get_estimated_landmarks()
        current_ids = set(est_landmarks.keys())

        # Detect landmarks that were just added to the EKF and
        # have not been explored yet.
        new_ids = (current_ids - self._known_ids) - self.visited
        self._known_ids = current_ids.copy()

        # ---- state transitions ----
        self._update_state(new_ids, current_ids, est_landmarks,
                           x, y, theta)

        # ---- produce control output based on current mode ----
        if self.mode == "seek":
            return self._seek_control(est_landmarks, robot_pose)
        if self.mode == "explore":
            return self._explore_control(est_landmarks, robot_pose)
        return self._wander_control(robot_pose)

    # ------------------------------------------------------------------
    # State-machine transitions
    # ------------------------------------------------------------------
    def _update_state(self, new_ids, current_ids, est_landmarks, x, y, theta):
        """Advance the internal state machine."""

        # --- WANDER → SEEK when a new or unvisited landmark is available ---
        if self.mode == "wander":
            target = self._pick_target(new_ids, current_ids,
                                       est_landmarks, x, y)
            if target is not None:
                self.target_id = target
                self.mode = "seek"

        # --- SEEK → EXPLORE once close enough, or back to WANDER if stale ---
        if self.mode == "seek":
            if self.target_id is None or self.target_id in self.visited:
                self._enter_wander()
            elif self.target_id in est_landmarks:
                tpos = est_landmarks[self.target_id]
                if np.hypot(tpos[0] - x, tpos[1] - y) < self.orbit_radius:
                    self.mode = "explore"
                    self.explore_counter = 0

        # --- EXPLORE → WANDER after enough orbiting ---
        if self.mode == "explore":
            self.explore_counter += 1
            if self.explore_counter >= self.orbit_steps:
                self.visited.add(self.target_id)
                self._enter_wander()

    def _pick_target(self, new_ids, current_ids, est_landmarks, x, y):
        """
        Choose the next landmark to visit.
        Prefer a brand-new detection; fall back to the closest unvisited
        landmark already in the map.
        """
        if new_ids:
            # Prefer the closest newly-detected landmark
            return min(new_ids,
                       key=lambda lid: np.hypot(
                           est_landmarks[lid][0] - x,
                           est_landmarks[lid][1] - y))

        unvisited = current_ids - self.visited
        if unvisited:
            return min(unvisited,
                       key=lambda lid: np.hypot(
                           est_landmarks[lid][0] - x,
                           est_landmarks[lid][1] - y))
        return None

    def _enter_wander(self):
        """Reset to the wander state with a fresh random heading."""
        self.mode = "wander"
        self.target_id = None
        self.wander_heading = np.random.uniform(-np.pi, np.pi)

    # ------------------------------------------------------------------
    # Low-level control laws
    # ------------------------------------------------------------------
    def _seek_control(self, est_landmarks, robot_pose):
        """Proportional steering toward the target landmark."""
        x, y, theta = robot_pose
        tpos = est_landmarks[self.target_id]
        dx, dy = tpos[0] - x, tpos[1] - y
        desired = np.arctan2(dy, dx)
        err = wrap_angle(desired - theta)
        omega = self.gain * err
        # Slow down during sharp turns to keep the path smooth
        v = self.base_speed * max(0.3, 1.0 - abs(err) / np.pi)
        return v, omega

    def _explore_control(self, est_landmarks, robot_pose):
        """
        Counter-clockwise orbit around the target landmark.
        A radial correction term keeps the robot near *orbit_radius*.
        """
        x, y, theta = robot_pose
        tpos = est_landmarks[self.target_id]
        dx, dy = tpos[0] - x, tpos[1] - y
        dist = max(np.hypot(dx, dy), 1e-3)

        angle_to_lm = np.arctan2(dy, dx)
        # Tangent + radial pull to maintain desired orbit distance
        radial_err = (dist - self.orbit_radius) / self.orbit_radius
        desired = angle_to_lm + np.pi / 2 - 0.5 * radial_err
        err = wrap_angle(desired - theta)
        omega = self.gain * err
        v = self.base_speed * 0.7
        return v, omega

    def _wander_control(self, robot_pose):
        """
        Drive along self.wander_heading.  If the robot drifts near
        the environment boundary, redirect toward the centre.
        """
        x, y, theta = robot_pose
        xmin, xmax, ymin, ymax = self.env_bounds
        margin = 2.0

        # Bounce off walls by steering toward the environment centre
        if (x < xmin + margin or x > xmax - margin or
                y < ymin + margin or y > ymax - margin):
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
            self.wander_heading = np.arctan2(cy - y, cx - x)

        err = wrap_angle(self.wander_heading - theta)
        omega = self.gain * err
        v = self.base_speed
        return v, omega


def greedy_uncertainty_control(ekf, robot_pose, base_speed=1.0, gain=2.0,
                               well_mapped_threshold=1.5):
    """
    Greedy controller that steers toward the landmark with the
    highest uncertainty (largest covariance trace).

    Landmarks with covariance trace below *well_mapped_threshold* are
    considered sufficiently mapped and ignored.  When every known
    landmark is well-mapped (or none have been seen yet), the robot
    switches to an exploration arc so it can discover new landmarks.

    Args:
        ekf:                   EKFSLAM instance
        robot_pose:            current true pose [x, y, theta]
        base_speed:            forward velocity (m/s)
        gain:                  proportional gain for heading correction
        well_mapped_threshold: covariance-trace below which a landmark
                               is considered "good enough"

    Returns:
        (v, omega) control tuple
    """
    est_landmarks = ekf.get_estimated_landmarks()
    landmark_covs = ekf.get_landmark_covariances()

    # Filter to landmarks that still need improvement
    uncertain = {lid: np.trace(c) for lid, c in landmark_covs.items()
                 if np.trace(c) > well_mapped_threshold}

    # If no landmarks yet, or all known ones are well-mapped,
    # steer toward the centre of the environment to maximise
    # landmark discovery, then arc left once we're near it.
    if not uncertain:
        cx, cy = 9.0, 9.0
        x, y, theta = robot_pose
        dx_c = cx - x
        dy_c = cy - y
        dist_to_centre = np.sqrt(dx_c * dx_c + dy_c * dy_c)

        if dist_to_centre > 3.0:
            desired = np.arctan2(dy_c, dx_c)
            err = wrap_angle(desired - theta)
            return base_speed, gain * err
        # Already near the centre — arc left to sweep the area
        return base_speed, 0.3

    # Target the most uncertain landmark that still needs work
    target_id = max(uncertain, key=uncertain.get)
    target_pos = est_landmarks[target_id]

    x, y, theta = robot_pose
    dx = target_pos[0] - x
    dy = target_pos[1] - y
    dist = np.sqrt(dx * dx + dy * dy)

    # If we're already close enough to observe it well, move on to
    # the next-most-uncertain one (or explore if none left).
    if dist < 3.0:
        remaining = {lid: tr for lid, tr in uncertain.items()
                     if lid != target_id}
        if remaining:
            target_id = max(remaining, key=remaining.get)
            target_pos = est_landmarks[target_id]
            dx = target_pos[0] - x
            dy = target_pos[1] - y
        else:
            return base_speed, 0.3

    # Proportional steering toward the target
    desired_heading = np.arctan2(dy, dx)
    heading_err = wrap_angle(desired_heading - theta)
    omega = gain * heading_err

    # Scale speed down when making sharp turns to keep the path smooth
    v = base_speed * max(0.3, 1.0 - abs(heading_err) / np.pi)

    return v, omega


def main():
    """Run uncertainty visualization demo."""

    # =================================================================
    # CONTROL STRATEGY TOGGLE
    # Change this variable to switch between strategies:
    #   "greedy"        — steers toward the most uncertain landmark
    #   "landmark_hop"  — wanders randomly, then orbits each new
    #                      landmark it detects before moving on
    # =================================================================
    CONTROL_MODE = "landmark_hop"

    print("\n" + "="*60)
    print("EKF-SLAM UNCERTAINTY VISUALIZATION")
    print("="*60)
    print(f"\n  Control strategy: {CONTROL_MODE}")
    print("\nThis demo focuses on uncertainty estimation in SLAM.")
    print("\nVisualization elements:")
    print("  • Ellipses show uncertainty regions (1σ and 2σ)")
    print("  • Colors indicate uncertainty level:")
    print("    - Blue = low uncertainty (well-observed)")
    print("    - Red = high uncertainty (fewer observations)")
    print("  • Ellipse size shows position uncertainty")
    print("  • Ellipse orientation shows correlation")
    print("\nWatch how:")
    print("  • Uncertainty decreases with repeated observations")
    print("  • New landmarks start with high uncertainty")
    print("  • Robot pose uncertainty grows between observations")
    print("="*60 + "\n")
    
    # Setup
    env = Environment(seed=42)
    
    # Higher noise for more visible uncertainty
    R = np.diag([0.1, 0.1, np.deg2rad(3.0)]) ** 2  # Process noise
    Q = np.diag([0.5, np.deg2rad(8.0)]) ** 2  # Measurement noise
    
    init_pose = [0.0, 0.0, 0.0]
    ekf = EKFSLAM(init_pose=init_pose, R=R, Q=Q)
    
    simulator = RobotSimulator(
        environment=env,
        init_pose=init_pose,
        process_noise=R,
        measurement_noise=Q
    )

    # Instantiate the landmark-hop controller (only used when selected)
    hop_controller = LandmarkHopController(
        base_speed=1.0, gain=2.0, orbit_radius=3.5, orbit_steps=100,
        env_bounds=(env.x_min + 1, env.x_max - 1,
                    env.y_min + 1, env.y_max - 1)
    )
    
    # Visualization setup
    plt.ion()
    fig, ax = plt.subplots(figsize=(14, 11))
    
    num_steps = 360
    update_interval = 5  # Update every N steps
    
    # Metrics log — one dict per time step
    metrics_log = []
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "ekf_slam_metrics.csv")

    print("Starting exploration with uncertainty tracking...")
    print("(Higher noise settings for clearer visualization)\n")
    
    try:
        for t in range(num_steps):
            # --- Select control based on the chosen strategy ---
            if CONTROL_MODE == "greedy":
                v, omega = greedy_uncertainty_control(
                    ekf, simulator.true_pose, base_speed=1.0, gain=2.0
                )
            else:
                v, omega = hop_controller(ekf, simulator.true_pose)
            control = (v, omega)
            
            # Simulate and update SLAM
            measurements = simulator.step(control)
            ekf.step(control, measurements, dt=simulator.dt)

            # Collect numerical metrics every step
            step_metrics = collect_step_metrics(ekf, simulator, env, t, measurements)
            metrics_log.append(step_metrics)
            
            # Update visualization
            if t % update_interval == 0:
                visualize_uncertainty_evolution(
                    ax, env, ekf,
                    simulator.get_true_path(),
                    simulator.true_pose,
                    t, num_steps
                )
                plt.pause(0.01)
            
            # Progress updates with richer metrics
            if (t + 1) % 50 == 0:
                m = step_metrics
                print(f"  Step {t+1}/{num_steps} — "
                      f"LMs: {m['n_landmarks']}, "
                      f"PoseErr: {m['pose_error_m']:.4f}m, "
                      f"HdgErr: {np.degrees(m['heading_error_rad']):.2f}°, "
                      f"AvgLMErr: {m['avg_landmark_error_m']:.4f}m, "
                      f"AvgLMUnc: {m['avg_landmark_uncertainty']:.4f}")
        
        print(f"\n✓ Exploration complete!")
        print(f"  Final landmarks: {len(ekf.landmark_indices)}/{len(env.landmarks)}")
        
        # Save per-step metrics to CSV
        save_metrics_csv(metrics_log, csv_path)

        # Print detailed summary table
        print_final_summary(metrics_log, ekf, env)

        # --- Static summary figures ---
        plt.ioff()

        # Figure 1: true vs estimated trajectory + landmark estimates with ellipses
        fig_state = plot_final_slam_state(env, ekf, simulator)
        fig_state.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "ekf_slam_final_state.png"), dpi=150)
        print("  Final-state figure saved to ekf_slam_final_state.png")

        # Figure 2: metrics over time (pose error, pose unc, avg LM unc)
        fig_metrics = plot_metrics_over_time(metrics_log)
        fig_metrics.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "ekf_slam_metrics_plot.png"), dpi=150)
        print("  Metrics-over-time figure saved to ekf_slam_metrics_plot.png")

        print("\nAll figures displayed. Close windows to exit.")
        plt.show()
        
    except KeyboardInterrupt:
        # Still save whatever metrics we collected before interruption
        if metrics_log:
            save_metrics_csv(metrics_log, csv_path)
            print_final_summary(metrics_log, ekf, env)
        print("\n\nDemo interrupted by user.")
        plt.close()


if __name__ == "__main__":
    main()

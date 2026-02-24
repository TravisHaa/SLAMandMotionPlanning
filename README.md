# CA3 Navigation — Added Files & Folders

This document describes all files and directories that were added on top of the original starter code (`environment.py`, `motion_planning.py`, `test_PathPlanning.py`, `slam.py`, `test_EKF-SLAM.py`).

---

## Scripts

| File | Purpose |
|---|---|
| `experiments_planning.py` | **Scenario-based comparison** of PRM vs RRT. Builds two custom obstacle environments (Bottleneck and Zigzag), runs both planners on each, saves roadmap/tree and final-path visualisations, collects multi-trial metrics, and produces comparison plots. |
| `param_study_planning.py` | **Parameter sensitivity study.** Sweeps PRM over `n_samples` × `k_neighbors` and RRT over `step_size` × `goal_sample_rate`. Logs per-config metrics to CSV and generates line plots + heatmaps. |
| `planning_improvements.py` | **Path post-processing improvements.** Implements greedy shortcutting for PRM and random shortcutting (smoothing) for RRT, then compares baseline vs improved paths both visually and quantitatively. |

### How to run

```bash
python3 experiments_planning.py      # -> planning_results/
python3 param_study_planning.py      # -> planning_param_studies/
python3 planning_improvements.py     # -> planning_improvements/
```

All scripts use `matplotlib` in headless mode (`Agg` backend) and require only `numpy` and `matplotlib`.

---

## Output Folders

### `planning_results/`

Scenario-based PRM vs RRT comparison outputs.

| File | Description |
|---|---|
| `Scenario1_Bottleneck_prm_roadmap.png` | Full PRM roadmap for the bottleneck environment |
| `Scenario1_Bottleneck_prm_path.png` | PRM final path (bottleneck) |
| `Scenario1_Bottleneck_rrt_tree.png` | Full RRT tree for the bottleneck environment |
| `Scenario1_Bottleneck_rrt_path.png` | RRT final path (bottleneck) |
| `Scenario1_Bottleneck_comparison.png` | 4-panel metric comparison (success rate, path length, time, smoothness) |
| `Scenario1_Bottleneck_metrics.csv` | Per-trial metrics for the bottleneck scenario |
| `Scenario2_Zigzag_prm_roadmap.png` | Full PRM roadmap for the zigzag environment |
| `Scenario2_Zigzag_prm_path.png` | PRM final path (zigzag) |
| `Scenario2_Zigzag_rrt_tree.png` | Full RRT tree for the zigzag environment |
| `Scenario2_Zigzag_rrt_path.png` | RRT final path (zigzag) |
| `Scenario2_Zigzag_comparison.png` | 4-panel metric comparison (zigzag) |
| `Scenario2_Zigzag_metrics.csv` | Per-trial metrics for the zigzag scenario |
| `combined_final_paths.png` | **2×2 panel** — PRM and RRT final paths for both scenarios side by side |
| `grand_comparison.png` | Grouped bar chart of success rate and path length across both scenarios |
| `all_scenario_metrics.csv` | Combined CSV of all scenario trial data |

### `planning_param_studies/`

Parameter sweep outputs.

| File | Description |
|---|---|
| `prm_line_plots.png` | Success rate, path length, and time vs `n_samples` (one curve per `k_neighbors`) |
| `prm_heatmaps.png` | Heatmaps of success rate and path length over the `n_samples` × `k_neighbors` grid |
| `prm_sweep.csv` | Raw PRM sweep data |
| `rrt_line_plots.png` | Success rate, path length, and time vs `step_size` (one curve per `goal_sample_rate`) |
| `rrt_heatmaps.png` | Heatmaps of success rate and path length over the `step_size` × `goal_sample_rate` grid |
| `rrt_sweep.csv` | Raw RRT sweep data |
| `combined_param_studies.png` | **All four plots stacked** into a single image |

### `planning_improvements/`

Path post-processing improvement outputs.

| File | Description |
|---|---|
| `combined_before_after.png` | **2×2 panel** — before (red) vs after (green) paths for PRM shortcutting and RRT smoothing |
| `improvement_comparison.png` | Grouped bar charts comparing baseline vs improved across path length, waypoints, time, and smoothness |
| `improvement_metrics.csv` | Per-trial metrics for all four variants (PRM baseline, PRM improved, RRT baseline, RRT improved) |

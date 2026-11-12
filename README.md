# Distributed Active Perception for Multi-Robot Radiation Source Localization in Nuclear Facilities

Reference implementation, simulation environment and experimental datasets for the manuscript:

> **Distributed Active Perception for Multi-Robot Radiation Source Localization in Nuclear Facilities**
> Dong Yan, Yufeng Xiao (corresponding author)
> Submitted to *Advanced Engineering Informatics*

The framework couples a **Voronoi-partitioned particle filter** (spatial-probabilistic
knowledge base), an **AOA-variance surrogate of directional mutual information**
(active trajectory optimization), and a **sequential source-subtraction loop with a
Neyman–Pearson residual test** (unknown source cardinality).

## Repository layout

```
.
├── ros_ws/                          # catkin workspace (ROS Noetic)
│   └── src/nuclear_search/
│       ├── launch/multi_robot_search.launch   # 3 Ackermann robots + Gazebo world + RViz
│       ├── scripts/
│       │   ├── multi_agent_aoa_voronoi.py     # main node: Voronoi-SPF + AOA active perception
│       │   └── data_recorder.py               # logs entropy / robot poses to CSV
│       ├── urdf/robot_sensors.xacro           # robot model with AOA + count-rate sensors
│       ├── worlds/nuclear_maze.world          # maze in which the sources are hidden
│       └── rviz/search_config.rviz            # visualization layout
├── analysis/                        # post-processing that generates the paper tables/figures
│   ├── analyze_trials.py            # RMSE / search-time statistics of field trials
│   ├── make_section5_tables.py      # builds the LaTeX tables of Section 5
│   ├── make_figure1.py              # vector version of the technical roadmap (Fig. 1)
│   ├── real_single_source.csv       # real single-source field trials
│   └── hil_multi_source.csv         # outdoor hardware-in-the-loop multi-source trials
└── data/
    ├── sim_logs/                    # raw Gazebo run logs (spatial entropy, discovered count, poses)
    └── figures/                     # representative RViz captures of the four search phases
```

## Requirements

* Ubuntu 20.04 + [ROS Noetic](http://wiki.ros.org/noetic/Installation/Ubuntu)
* Gazebo 11 (`gazebo_ros`), `rviz`, `robot_state_publisher`, `xacro`, `tf2_ros`
* Python 3.8 with `numpy`, `scipy`, `pandas`, `matplotlib`

## Simulation

```bash
# build the workspace
cd ros_ws
catkin_make
source devel/setup.bash

# launch Gazebo + 3 robots + the active-perception node + RViz
roslaunch nuclear_search multi_robot_search.launch
```

The three robots start at `(2,2)`, `(18,2)` and `(10,18)` in the 20 m × 20 m maze,
each carrying one directional AOA sensor and one omnidirectional count-rate sensor.
The search logs are written to `search_data.csv` in the working directory; run
`scripts/data_recorder.py` in parallel to record the per-iteration spatial entropy
and the robot trajectories.

## Reproducing the paper's tables and figures

```bash
cd analysis

# Section 5 tables (reads real_single_source.csv and hil_multi_source.csv in this folder)
python3 make_section5_tables.py

# statistics reported in the text (RMSE, search time, success rate)
python3 analyze_trials.py --csv real_single_source.csv
python3 analyze_trials.py --csv hil_multi_source.csv

# Figure 1 (technical roadmap, vector pdf)
python3 make_figure1.py
```

`analyze_trials.py` expects the columns
`method, trial, source, true_x, true_y, est_x, est_y, time_s, path_m, success`
(case-insensitive, extra columns ignored).

## Data

* `data/sim_logs/radiation_search_*.csv` – proposed method, columns
  `Iteration, Spatial_Entropy, Discovered_Count, Robot{1,2,3}_{X,Y}`.
* `data/sim_logs/blind_search_*.csv` – ablation baseline, same schema.
* `data/sim_logs/search_log_*.csv` – per-timestep traces of individual runs.
* Files containing only a header row correspond to runs that were aborted at start-up
  and are kept only for completeness.
* `analysis/*.csv` – field-trial logs from which Section 5 is computed.

## License

To be determined before public release. Please contact the corresponding author
(xiaoyufeng@swust.edu.cn) for the terms of use.

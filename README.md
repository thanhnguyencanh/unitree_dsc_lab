# Unitree DSC Lab — Stair Climbing on G1

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.2-silver)](https://isaac-sim.github.io/IsaacLab)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

Replication of *"Explicit Stair Geometry Conditioning for Robust Humanoid
Locomotion"* (arXiv:2605.09944v1) on the Unitree G1.

This repo is structured as a standalone Isaac Lab extension that lives **outside**
the `IsaacLab/` tree so updates to IsaacLab do not clobber the code. Layout and
tooling are modeled on
[`unitree_rl_lab`](https://github.com/unitreerobotics/unitree_rl_lab).

## What's inside

```
unitree_dsc_lab/
├── source/unitree_dsc_lab/                  # editable-install python package
│   └── unitree_dsc_lab/
│       ├── assets/robots/                   # USD/URDF asset registry (G1)
│       ├── tasks/stair_climb/
│       │   ├── agents/                      # rsl_rl PPO runner configs
│       │   ├── mdp/                         # observations, rewards, terminations, commands
│       │   ├── robots/g1_23dof/             # env cfg + gym.register
│       │   ├── terrains/                    # procedural staircase generator
│       │   ├── perception/                  # BEV builder, CNN student encoder, teacher
│       │   └── policy/                      # actor-critic, three-stage runner
│       └── utils/                           # cli parsing, deploy cfg export
├── scripts/
│   ├── list_envs.py
│   ├── rsl_rl/{train.py, play.py, cli_args.py, export_onnx.py}
│   └── stages/{train_stage1_policy.py, train_stage2_perception.py, train_stage3_joint.py}
├── deploy/                                  # on-board ROS2 / C++ runtime
├── assets/g1/                               # local URDF / MJCF copies (gitignored)
├── doc/
├── docker/
├── unitree_dsc_lab.sh                       # install / list / train / play helper
├── pyproject.toml
└── README.md
```

## Installation

- Install Isaac Lab 2.3.2 (Isaac Sim 5.1) — see
  [`STAIR_CLIMBING_REPLICATION_GUIDE.md`](../STAIR_CLIMBING_REPLICATION_GUIDE.md)
  sections 3–5.
- Clone this repo outside the `IsaacLab/` directory.
- Install in editable mode against the active Isaac Lab conda env:

  ```bash
  conda activate stairg1
  ./unitree_dsc_lab.sh -i
  ```

- Provide the Unitree G1 robot description. Either:
  - Set `UNITREE_ROS_DIR` in `source/unitree_dsc_lab/unitree_dsc_lab/assets/robots/unitree.py`
    to a checkout of [`unitree_ros`](https://github.com/unitreerobotics/unitree_ros), or
  - Set `UNITREE_MODEL_DIR` to a checkout of the
    [`unitree_model`](https://huggingface.co/datasets/unitreerobotics/unitree_model) USDs.

## Tasks

| Task ID                          | Stage  | Notes |
|----------------------------------|--------|-------|
| `Unitree-G1-23dof-StairClimb-v0` | 1/2/3  | PPO with explicit 4-D terrain token `z_t = [s_t, h_step, d_step, theta_yaw]` |

List, train, play:

```bash
./unitree_dsc_lab.sh -l
./unitree_dsc_lab.sh -t --task Unitree-G1-23dof-StairClimb-v0
./unitree_dsc_lab.sh -p --task Unitree-G1-23dof-StairClimb-v0
```

## Three-stage training

See `scripts/stages/` and §12 of the replication guide.

```bash
python scripts/stages/train_stage1_policy.py     --max_iterations 6000 --num_envs 4096 --headless
python scripts/stages/train_stage2_perception.py --policy_ckpt logs/stage1/best.pt --epochs 50
python scripts/stages/train_stage3_joint.py      --resume_policy logs/stage1/best.pt \
                                                 --resume_encoder logs/stage2/best.pt \
                                                 --max_iterations 3000 --num_envs 2048
```

## Deploy

Sim-to-real ONNX export and on-board ROS 2 runtime live under
[`scripts/rsl_rl/export_onnx.py`](scripts/rsl_rl/export_onnx.py) and
[`deploy/`](deploy/). See §14–15 of the replication guide.

## Acknowledgements

- [`unitree_rl_lab`](https://github.com/unitreerobotics/unitree_rl_lab) — repo layout and tooling
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
- Paper: arXiv:2605.09944v1

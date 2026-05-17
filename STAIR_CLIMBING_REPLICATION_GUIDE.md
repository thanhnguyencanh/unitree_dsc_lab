# Replicating "Explicit Stair Geometry Conditioning for Robust Humanoid Locomotion" on Unitree G1

A complete, end-to-end guide for replicating the paper (arXiv:2605.09944v1) and deploying the trained policy on a real Unitree G1 humanoid.

> **Target platform:** Ubuntu 22.04 LTS, NVIDIA RTX 3090/4090 (or A6000), CUDA 12.x, Unitree G1 (23 or 29 DoF variant) with Intel RealSense D435i / Livox MID-360 / equivalent depth sensor.

---

## Table of Contents

0. [Task Checklist (Done / To-Do)](#0-task-checklist-done--to-do)
1. [Paper Overview & What You Are Building](#1-paper-overview--what-you-are-building)
2. [Hardware & System Prerequisites](#2-hardware--system-prerequisites)
3. [Base OS Setup (Drivers, CUDA, Conda)](#3-base-os-setup-drivers-cuda-conda)
4. [Install Isaac Sim](#4-install-isaac-sim)
5. [Install Isaac Lab](#5-install-isaac-lab)
6. [Project Layout & Repository Setup](#6-project-layout--repository-setup)
7. [Unitree G1 Asset & MJCF/USD Setup](#7-unitree-g1-asset--mjcfusd-setup)
8. [Procedural Stair Terrain Generator](#8-procedural-stair-terrain-generator)
9. [BEV Perception Module](#9-bev-perception-module)
10. [Privileged Teacher & Student Encoder](#10-privileged-teacher--student-encoder)
11. [PPO Policy: Observations, Actions, Rewards](#11-ppo-policy-observations-actions-rewards)
12. [Three-Stage Training Pipeline](#12-three-stage-training-pipeline)
13. [Evaluation in Isaac Lab + MuJoCo](#13-evaluation-in-isaac-lab--mujoco)
14. [Sim-to-Real Bridge (ONNX Export)](#14-sim-to-real-bridge-onnx-export)
15. [Real-World Deployment on Unitree G1](#15-real-world-deployment-on-unitree-g1)
16. [Tuning, Safety Checklist, Troubleshooting](#16-tuning-safety-checklist-troubleshooting)
17. [References](#17-references)

---

## 0. Task Checklist (Done / To-Do)

A living view of where this replication stands. Tick items off as you finish them; the section numbers link to the relevant chapter.

### Done

- [x] **§6 — Repository scaffold.** `unitree_dsc_lab/` created at `~/humanoid_ws/unitree_dsc_lab`, modeled on `unitree_rl_lab` (editable `source/unitree_dsc_lab/` package, helper `./unitree_dsc_lab.sh`, `scripts/{rsl_rl,stages}/`, `deploy/`, `pyproject.toml`, `.gitignore`, `.flake8`). Initial git commit on `main`.
- [x] **§6 — Module stubs.** `tasks/stair_climb/{agents,mdp,robots/g1_23dof,terrains,perception,policy}/` and `scripts/{rsl_rl,stages}/` placeholders raise `NotImplementedError` and link back to the paper section / `unitree_rl_lab` analogue.
- [x] **§6 — Gym registration.** Task id `Unitree-G1-23dof-StairClimb-v0` registered (env cfg + PPO runner cfg are stubs).
- [x] **§8 — `StairsTerrainGenerator` implemented.** Procedural trimesh builder for `{flat, stairs-up, stairs-down}` tiles with paper-range randomization (`h_step`, `d_step`, yaw, lead-in) and `StairsTerrainTestCfg` for the OOD test range. Per-tile GT `(class_id, h_step, d_step, theta_yaw_terrain)` stored in `StairGTRegistry` keyed by IsaacLab's `dict_to_md5_hash(cfg)`. Smoke-tested: determinism + difficulty interpolation + correct tile geometry across all three classes.
- [x] **§10 — Privileged teacher wired.** `PrivilegedTeacher` holds per-env GPU buffers and exposes `.token(robot_yaw_world) -> z_t` (computes `theta_yaw_current = wrap(robot_yaw - terrain_yaw)`). `terrain_token_privileged` observation term reads from `env.privileged_teacher`. `refresh_from_cfgs(env_ids, [...])` repopulates the buffer on episode reset by looking up the registry.
- [x] **§8 follow-up — Stair scene + reset event wired.** `StairTerrainGenerator(TerrainGenerator)` subclass records per-`(row, col)` GT in `StairGTRegistry._by_row_col` at terrain build. `StairTerrainGeneratorCfg` (with `class_type=StairTerrainGenerator`) is used by `STAIR_TERRAIN_CFG` in [`tasks/stair_climb/robots/g1_23dof/stair_env_cfg.py`](source/unitree_dsc_lab/unitree_dsc_lab/tasks/stair_climb/robots/g1_23dof/stair_env_cfg.py), mixing the three class-locked presets `FlatLeadInCfg`/`StairsUpCfg`/`StairsDownCfg` via `proportion=(0.2, 0.4, 0.4)` across `num_rows=10 × num_cols=20`. Reset-mode event `mdp.reset_privileged_teacher` (a `ManagerTermBase`) attaches `env.privileged_teacher` on first call and on every reset calls `teacher.refresh_from_terrain(env_ids, env.scene.terrain)`. Smoke-tested: every tile has a registry entry, per-row class distribution matches proportions, h_step escalates with curriculum, teacher GT matches the assigned `(row, col)` per env.

### To-Do — environment

- [ ] **§3.1** — Install NVIDIA driver 550+, verify `nvidia-smi`.
- [ ] **§3.2** — Miniconda, create `stairg1` env (Python 3.11).
- [ ] **§3.3** — Git LFS installed.
- [ ] **§4** — `pip install isaacsim[all,extscache]==5.1.0`, accept EULA, smoke test.
- [ ] **§5** — `IsaacLab` checked out at `v2.3.2`, `./isaaclab.sh --install` succeeds.
- [ ] **§5** — Stock `Isaac-Velocity-Rough-G1-v0` runs at steady FPS / rising reward.
- [ ] **§6** — `./unitree_dsc_lab.sh -i` runs cleanly in `stairg1` env.

### To-Do — assets

- [ ] **§7** — Clone `unitree_ros` and `unitree_rl_gym` into `assets/g1/`.
- [ ] **§7** — Set `UNITREE_ROS_DIR` (or `UNITREE_MODEL_DIR`) so `G1_23DOF_CFG.spawn` resolves.
- [ ] **§7** — Verify 23-DoF joint config (freeze wrists if running 29-DoF G1).

### To-Do — simulation & training code

- [ ] **§9** — Implement `points_to_bev` (vectorized GPU scatter into 6 × 60 × 60).
- [ ] **§9** — Implement `BEVStudentEncoder` (Conv stack → 128 × 8 × 8 → 4 MLP heads).
- [x] **§11** — `G1StairClimbEnvCfg` scaffolded (scene with `StairTerrainGeneratorCfg`, proprio + `terrain_token_privileged` obs, joint-pos actions, forward-velocity command, IsaacLab-default rough-locomotion reward stack, reset/startup/interval events incl. `reset_privileged_teacher`, terminations, `terrain_levels_vel` curriculum). Stair-specific shaping (`swing_clearance_bonus`/`step_alignment_bonus`) is still pending.
- [ ] **§11** — Implement `swing_clearance_bonus` and `step_alignment_bonus` rewards.
- [ ] **§11** — Implement `StairClimbActorCritic` compatible with rsl_rl `OnPolicyRunner`.
- [ ] **§12** — Implement `ThreeStagePPORunner` (stage-1 PPO loop, stage-2 perception SL, stage-3 joint loss).
- [ ] **§12** — Stage 1 reaches `success_rate > 0.85` on training stair heights.
- [ ] **§12** — Stage 2 hits `MAE(h) < 1 cm`, `MAE(d) < 1 cm`, `class_acc > 99 %`.
- [ ] **§12** — Stage 3 joint fine-tune converges with LR × 0.3.

### To-Do — evaluation

- [ ] **§13** — `scripts/rsl_rl/play.py` reproduces Table I numbers in Isaac Lab.
- [ ] **§13** — Scaffold `scripts/rsl_rl/mujoco_eval.py`; cross-sim sanity check (Fig. 4).

### To-Do — sim-to-real & deploy

- [ ] **§14** — `scripts/rsl_rl/export_onnx.py` exports `encoder.onnx` + `policy.onnx`.
- [ ] **§14** — `utils/export_deploy_cfg.py` writes runtime YAML (joint order, PD gains, obs/action scales).
- [ ] **§14** — `onnxruntime` smoke test on CUDA EP.
- [ ] **§15.1** — Jetson Orin: ROS 2 Humble + `unitree_sdk2` + `onnxruntime-gpu`.
- [ ] **§15.2** — `deploy/pointcloud_to_bev.py` ROS 2 node implemented.
- [ ] **§15.2** — `deploy/g1_runtime.py` 50 Hz policy node implemented.
- [ ] **§15.2** — `deploy/robots/g1_23dof/` C++ controller compiles against `unitree_sdk2`.
- [ ] **§15.3** — Safety bring-up steps 1–6 passed three runs in a row.
- [ ] **§16** — Pre-flight checklist signed off on every run.

> Tip: keep this section honest. If a checkbox is ticked but a downstream stage fails, untick it and note why in the corresponding section.

---

## 1. Paper Overview & What You Are Building

The paper replaces implicit heightmap encoders with a **compact 4-D terrain token**:

```
z_t = [ s_t, h_step, d_step, theta_yaw_current ]
```

where:
- `s_t ∈ {flat, stairs-up, stairs-down}` — terrain class
- `h_step` — step height (m)
- `d_step` — step depth (m)
- `theta_yaw_current` — robot heading relative to terrain direction (rad)

**Three components** must be implemented:
1. **BEV Encoder (student):** 6-channel 60×60 BEV from on-board depth → CNN → `z_t`.
2. **Privileged Teacher (sim only):** ground-truth terrain + proprioception → supervises `z_t`.
3. **PPO Actor–Critic:** proprioception ⊕ `z_t` → joint targets at 50 Hz, wrapped by a PD controller.

Loss: `L_total = L_PPO + α · L_terrain`, with `L_terrain = 0.6·CE + 1·L1(h) + 1·L1(d)` and `α = 1`.

---

## 2. Hardware & System Prerequisites

### Workstation (training)
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU       | RTX 3080 12 GB | RTX 4090 24 GB / A6000 48 GB |
| RAM       | 32 GB | 64 GB |
| Disk      | 200 GB SSD | 1 TB NVMe |
| OS        | Ubuntu 22.04 | Ubuntu 22.04 |

### Robot side
- Unitree G1 (firmware ≥ 1.4)
- On-board Jetson Orin (or NUC i7) running Ubuntu 20.04/22.04 with ROS 2 Humble
- Depth source — RealSense D435i (paper-equivalent), or Livox MID-360 + projection
- Emergency-stop and a safety harness/gantry for first runs

---

## 3. Base OS Setup (Drivers, CUDA, Conda)

### 3.1 NVIDIA driver + CUDA 12.4
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git curl wget vim \
    libglu1-mesa libxi6 libxrandr2 libxinerama1 libxcursor1 \
    libgl1-mesa-glx libegl1 libgles2

# Driver 550+ ships CUDA 12.4 runtime
sudo ubuntu-drivers install nvidia:550
sudo reboot

# Verify after reboot
nvidia-smi
```

### 3.2 Miniconda
```bash
cd /tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
conda init zsh   # or bash
```

> **Note:** Isaac Sim 5.x requires **Python 3.11** — do not create a 3.10 env or `pip install isaacsim` will refuse.

### 3.3 Git LFS (required for USD assets)
```bash
sudo apt install -y git-lfs
git lfs install
```

---

## 4. Install Isaac Sim

> Isaac Lab 2.3+ pairs with **Isaac Sim 5.1** and installs it via pip. Support for Isaac Sim ≤ 4.2 has been dropped, and 5.1 is the recommended baseline as of May 2026.

```bash
conda create -n stairg1 python=3.11 -y
conda activate stairg1

# PyTorch CUDA 12.4 wheels (cu124 works with the NVIDIA 550+ driver shipped above)
pip install --upgrade pip
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Isaac Sim 5.1 (≈12 GB download)
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

# First-launch EULA accept (headless OK)
isaacsim --no-window
```

Smoke test:
```bash
python -c "from isaacsim.simulation_app import SimulationApp; \
  app = SimulationApp({'headless': True}); print('OK'); app.close()"
```

---

## 5. Install Isaac Lab

```bash
cd ~/humanoid_ws        # working dir already exists
# IsaacLab/ is already cloned in your workspace — fetch the matching tag
cd IsaacLab
git fetch --tags
git checkout v2.3.2     # latest stable on Isaac Sim 5.1 (May 2026)

# Pip-install all extensions in editable mode (uses the active conda env)
./isaaclab.sh --install
```

Verify with a stock training task:
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Rough-G1-v0 --num_envs 64 --headless
```

If you see steady FPS and rising reward, the stack is healthy.

---

## 6. Project Layout & Repository Setup

Create a new extension that lives **outside** the IsaacLab tree so updates to IsaacLab don't clobber your code. The repository is structured as a **standalone Isaac Lab extension** (editable-install python package under `source/`, helper `.sh`, separate `scripts/` and `deploy/`) — the same shape as [`unitree_rl_lab`](https://github.com/unitreerobotics/unitree_rl_lab), with stair-climbing-specific modules added under `tasks/stair_climb/`.

```bash
cd ~/humanoid_ws
# The unitree_dsc_lab/ repo is already scaffolded — it was created from this guide.
# To bootstrap from scratch:
git clone <your-fork>/unitree_dsc_lab.git    # or start a fresh one with:
# mkdir unitree_dsc_lab && cd unitree_dsc_lab && git init -b main
```

Top-level layout (as scaffolded):
```
unitree_dsc_lab/
├── source/unitree_dsc_lab/                  # editable-install python package
│   ├── setup.py
│   ├── config/extension.toml
│   └── unitree_dsc_lab/
│       ├── __init__.py
│       ├── assets/robots/unitree.py         # G1_23DOF_CFG, URDF/USD path overrides
│       ├── utils/{parser_cfg,export_deploy_cfg}.py
│       └── tasks/stair_climb/
│           ├── agents/rsl_rl_ppo_cfg.py     # PPO runner cfg
│           ├── mdp/                         # observations, rewards, terminations, commands
│           ├── robots/g1_23dof/             # env cfg + gym.register("Unitree-G1-23dof-StairClimb-v0")
│           ├── terrains/stair_generator.py  # procedural staircase
│           ├── perception/{bev,encoder,teacher}.py
│           └── policy/{actor_critic,ppo_runner}.py    # PPO actor-critic + three-stage runner
├── scripts/
│   ├── list_envs.py
│   ├── rsl_rl/{train,play,export_onnx,cli_args}.py
│   └── stages/{train_stage1_policy,train_stage2_perception,train_stage3_joint}.py
├── deploy/
│   ├── robots/g1_23dof/                     # C++ controller (mirrors unitree_rl_lab/deploy/...)
│   ├── g1_runtime.py                        # ROS2 node, 50 Hz inference
│   ├── pointcloud_to_bev.py                 # on-board BEV builder
│   └── safety_monitor.py
├── assets/g1/                               # local URDF / MJCF (gitignored)
├── doc/  docker/
├── unitree_dsc_lab.sh                       # install / list / train / play helper
├── pyproject.toml  .gitignore  .flake8  LICENCE
└── README.md
```

Install in editable mode against the Isaac Lab conda env:
```bash
conda activate stairg1
cd ~/humanoid_ws/unitree_dsc_lab
./unitree_dsc_lab.sh -i
# `./unitree_dsc_lab.sh -l` to list tasks once stubs are filled in.
```

> The Python files under `tasks/stair_climb/{terrains,perception,policy,mdp}/` and the `scripts/` entry points are scaffolded as stubs that raise `NotImplementedError` and cite the paper section / `unitree_rl_lab` analogue to follow when implementing. Sections 7–15 below fill them in.

---

## 7. Unitree G1 Asset & MJCF/USD Setup

```bash
cd ~/humanoid_ws/unitree_dsc_lab/assets/g1
git clone https://github.com/unitreerobotics/unitree_ros.git
git clone https://github.com/unitreerobotics/unitree_rl_gym.git
# G1 USD is bundled inside IsaacLab assets pack — pulled automatically by:
python -c "from isaaclab_assets.robots.unitree import G1_CFG; print(G1_CFG.spawn.usd_path)"
```

Both `unitree_ros/` and `unitree_model/` checkouts are `.gitignore`-d under `assets/g1/`. Point [`source/unitree_dsc_lab/unitree_dsc_lab/assets/robots/unitree.py`](source/unitree_dsc_lab/unitree_dsc_lab/assets/robots/unitree.py) at them via the `UNITREE_ROS_DIR` / `UNITREE_MODEL_DIR` env vars (or hard-code).

Confirm the joints match the paper's 23 DoF config — if you have the 29 DoF (with wrists), freeze the wrist joints in `tasks/stair_climb/robots/g1_23dof/stair_env_cfg.py::G1StairClimbEnvCfg` so the action dim stays at 23.

---

## 8. Procedural Stair Terrain Generator

**Status: implemented** — [`source/unitree_dsc_lab/unitree_dsc_lab/tasks/stair_climb/terrains/stair_generator.py`](source/unitree_dsc_lab/unitree_dsc_lab/tasks/stair_climb/terrains/stair_generator.py).

Randomization ranges:

| Parameter | Train range | Test range |
|-----------|-------------|------------|
| step height `h_step` | 0.12 – 0.16 m | 0.18 – 0.22 m |
| step depth `d_step`  | 0.26 – 0.32 m | same |
| yaw offset           | -25° – +25°   | up to ±35° |
| flat lead-in/out     | 1.0 – 2.0 m   | 1.0 – 2.0 m |

Module API:

```python
from unitree_dsc_lab.tasks.stair_climb.terrains import (
    StairsTerrainCfg,            # base config (paper train ranges)
    StairsTerrainTestCfg,        # OOD ranges (h ∈ [0.18, 0.22] m, yaw ±35°)
    FlatLeadInCfg,               # class-locked: flat only
    StairsUpCfg,                 # class-locked: stairs-up only
    StairsDownCfg,               # class-locked: stairs-down only
    stairs_terrain,              # IsaacLab function callback (difficulty, cfg) -> (meshes, origin)
    StairGTRegistry,             # process-local tile_hash + (row, col) -> StairGT lookup
    StairTerrainGenerator,       # TerrainGenerator subclass that records (row, col) GT
    StairTerrainGeneratorCfg,    # TerrainGeneratorCfg with class_type=StairTerrainGenerator
    lookup_tile_gt,              # helper to recover GT for a given (cfg, difficulty, seed)
)
```

Plug into an env via `TerrainImporterCfg` — mix the three class-locked presets by `proportion`:

```python
from isaaclab.terrains import TerrainImporterCfg
from unitree_dsc_lab.tasks.stair_climb.terrains import (
    StairTerrainGeneratorCfg, FlatLeadInCfg, StairsUpCfg, StairsDownCfg,
)

STAIR_TERRAIN_CFG = StairTerrainGeneratorCfg(
    size=(8.0, 4.0),
    num_rows=10, num_cols=20,     # 10 difficulty levels x 20 tiles per row
    curriculum=True,
    sub_terrains={
        "flat":        FlatLeadInCfg(proportion=0.2),
        "stairs_up":   StairsUpCfg(proportion=0.4),
        "stairs_down": StairsDownCfg(proportion=0.4),
    },
)

terrain = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=STAIR_TERRAIN_CFG,
    max_init_terrain_level=STAIR_TERRAIN_CFG.num_rows - 1,
    # ...
)
```

Two indexes are kept in sync during terrain build:

- `StairGTRegistry._store[tile_hash]` — populated by `stairs_terrain()` keyed by `dict_to_md5_hash(cfg.to_dict())` (same hash IsaacLab uses for its terrain cache).
- `StairGTRegistry._by_row_col[(row, col)]` — populated by `StairTerrainGenerator._add_sub_terrain()`. This is what the env-side teacher reads via `terrain.terrain_levels[env_id]` / `terrain.terrain_types[env_id]`, so no per-tile difficulty re-derivation is required.

A full reference wiring (terrain + observations + reset event + commands + rewards) lives in [`tasks/stair_climb/robots/g1_23dof/stair_env_cfg.py::G1StairClimbEnvCfg`](source/unitree_dsc_lab/unitree_dsc_lab/tasks/stair_climb/robots/g1_23dof/stair_env_cfg.py).

> The robot spawn origin is the centre of the lead-in flat patch, **after** the yaw rotation. On `stairs-down` tiles the spawn is elevated (top of the staircase); on `stairs-up` and `flat` it's at z = 0.

---

## 9. BEV Perception Module

**Status: stubs in place; bodies not yet implemented.**

[`tasks/stair_climb/perception/bev.py::points_to_bev`](source/unitree_dsc_lab/unitree_dsc_lab/tasks/stair_climb/perception/bev.py) — pure-tensor (GPU) projection. Per the paper:

- Region: 3 m × 3 m, resolution 0.05 m → grid 60 × 60
- Channels (6): `max(z)`, `min(z)`, `mean(z)`, `max - min`, `std(z)`, normalized point density
- Empty cells: zero-filled
- Frame: robot-centric, +X forward, +Y left

```python
def points_to_bev(points: torch.Tensor) -> torch.Tensor:  # (N, 3) -> (6, 60, 60)
    ...
```

[`tasks/stair_climb/perception/encoder.py::BEVStudentEncoder`](source/unitree_dsc_lab/unitree_dsc_lab/tasks/stair_climb/perception/encoder.py) — CNN matching paper specs:
- Multi-layer Conv2d + BatchNorm + ReLU, progressive downsampling
- Output `F_enc ∈ R^{128×8×8}` then MLP heads → `(logits_class[3], h_step, d_step, theta_yaw)`
- Replaces `mdp.terrain_token_privileged` in `ObservationsCfg.PolicyCfg` once Stage 2 kicks in.

---

## 10. Privileged Teacher & Student Encoder

**Status: implemented (teacher + observation term).** The teacher is **not a network** — it directly publishes the ground-truth `(s_t, h_step, d_step, theta_yaw_current)` from the terrain manager.

- [`tasks/stair_climb/perception/teacher.py::PrivilegedTeacher`](source/unitree_dsc_lab/unitree_dsc_lab/tasks/stair_climb/perception/teacher.py) holds per-env GPU buffers `(class_id, h_step, d_step, theta_yaw_terrain)` and exposes:
  - `.token(robot_yaw_world) -> (num_envs, 4)` — returns `z_t` with `theta_yaw_current = wrap(robot_yaw - terrain_yaw)`.
  - `.refresh_from_terrain(env_ids, terrain)` — **preferred** reset-time refresh. Reads `terrain.terrain_levels[env_ids]` / `terrain.terrain_types[env_ids]` and pulls from `StairGTRegistry._by_row_col` (populated by `StairTerrainGenerator`).
  - `.refresh_from_cfgs(env_ids, [(cfg, difficulty, seed), ...])` — escape hatch when you have explicit per-env cfg triples instead of a live `TerrainImporter`.
- [`tasks/stair_climb/mdp/observations.py::terrain_token_privileged`](source/unitree_dsc_lab/unitree_dsc_lab/tasks/stair_climb/mdp/observations.py) — observation term that reads from `env.privileged_teacher`.
- [`tasks/stair_climb/mdp/events.py::reset_privileged_teacher`](source/unitree_dsc_lab/unitree_dsc_lab/tasks/stair_climb/mdp/events.py) — `ManagerTermBase` event term that wires both halves: on construction it attaches `env.privileged_teacher`; on every reset it calls `teacher.refresh_from_terrain(env_ids, env.scene.terrain)`. Drop it into the env's `EventCfg`:

  ```python
  reset_privileged_teacher = EventTerm(
      func=mdp.reset_privileged_teacher,
      mode="reset",
  )
  ```

Student loss (Eq. 8 of the paper):
```python
L_terrain = 0.6 * F.cross_entropy(logits_s, s_gt) \
          + 1.0 * F.smooth_l1_loss(h_pred, h_gt) \
          + 1.0 * F.smooth_l1_loss(d_pred, d_gt)
# (yaw is part of the policy observation; not regressed)
```

---

## 11. PPO Policy: Observations, Actions, Rewards

**Status: env cfg scaffolded** — [`tasks/stair_climb/robots/g1_23dof/stair_env_cfg.py::G1StairClimbEnvCfg`](source/unitree_dsc_lab/unitree_dsc_lab/tasks/stair_climb/robots/g1_23dof/stair_env_cfg.py) wires the scene (`StairTerrainGeneratorCfg` + G1 articulation + contact sensor), observations (policy + critic groups), actions, commands, rewards, events (incl. `reset_privileged_teacher`), terminations, and `terrain_levels_vel` curriculum. The two paper-specific shaping rewards remain as stubs in [`mdp/rewards.py`](source/unitree_dsc_lab/unitree_dsc_lab/tasks/stair_climb/mdp/rewards.py).

**Observation** (per Eq. 1):
```
o_prop = [base_ang_vel, projected_gravity, velocity_commands,
          joint_pos - default, joint_vel, last_action]   # critic group also gets base_lin_vel
o_t    = concat(o_prop, z_t)                              # z_t = mdp.terrain_token_privileged
```

> The paper also lists `foot_contact_flags` in `o_prop`. Not yet wired — add a contact-flag obs term keyed to the ankle-roll bodies in [`mdp/observations.py`](source/unitree_dsc_lab/unitree_dsc_lab/tasks/stair_climb/mdp/observations.py) and a corresponding `ObsTerm` in `PolicyCfg`.

**Action:** `mdp.JointPositionActionCfg(joint_names=[".*"], scale=0.25, use_default_offset=True)` — target joint offsets from default, wrapped by IsaacLab's built-in PD (`Kp`/`Kd` come from `G1_23DOF_CFG.actuators`, sourced from the Unitree spec).

**Commands:** `mdp.UniformVelocityCommandCfg` with forward-only ranges `lin_vel_x ∈ [0.0, 0.7] m/s`, small ang_vel_z noise; resampling every 8–12 s.

**Rewards:** IsaacLab's default rough-locomotion stack (`track_lin_vel_xy_exp`, `track_ang_vel_z_exp`, `is_alive`, `lin_vel_z_l2`, `ang_vel_xy_l2`, `flat_orientation_l2`, `base_height_l2`, `joint_vel_l2`, `joint_acc_l2`, `action_rate_l2`, `joint_pos_limits`, arm/waist `joint_deviation_l1`, `undesired_contacts`). Plus two paper-specific shaping terms still to implement (stubs in [`mdp/rewards.py`](source/unitree_dsc_lab/unitree_dsc_lab/tasks/stair_climb/mdp/rewards.py)):
- `swing_clearance_bonus` — reward foot apex height ≥ `h_step + 5 cm` during stair-up
- `step_alignment_bonus`  — reward landing inside `d_step` window after stair edge

Once those land, register them in `RewardsCfg` alongside the existing terms.

---

## 12. Three-Stage Training Pipeline

The three-stage trainer is implemented in [`tasks/stair_climb/policy/ppo_runner.py::ThreeStagePPORunner`](source/unitree_dsc_lab/unitree_dsc_lab/tasks/stair_climb/policy/ppo_runner.py); each `scripts/stages/...` entry below delegates to it.

The Gym task id registered by `unitree_dsc_lab` is **`Unitree-G1-23dof-StairClimb-v0`** (see [`tasks/stair_climb/robots/g1_23dof/__init__.py`](source/unitree_dsc_lab/unitree_dsc_lab/tasks/stair_climb/robots/g1_23dof/__init__.py)).

### Stage 1 — Pre-train policy with ground-truth `z_t`
```bash
python scripts/stages/train_stage1_policy.py \
    --task Unitree-G1-23dof-StairClimb-v0 --num_envs 4096 \
    --max_iterations 6000 --headless \
    --logdir logs/stage1
# or equivalently:
./unitree_dsc_lab.sh -t --task Unitree-G1-23dof-StairClimb-v0
```
Targets: `success_rate > 0.85` on training stair heights.

### Stage 2 — Train student encoder under teacher supervision
Freeze the policy, run rollouts, store `(BEV, z_t_gt)` pairs, train CNN until:
- `MAE(h) < 1 cm`, `MAE(d) < 1 cm`, `class_acc > 99 %` (matches Table II of paper).

```bash
python scripts/stages/train_stage2_perception.py \
    --policy_ckpt logs/stage1/best.pt \
    --epochs 50 --batch_size 256 \
    --logdir logs/stage2
```

### Stage 3 — Joint fine-tuning
Unfreeze both. Loss = `L_PPO + 1.0 * L_terrain`. Reduce learning rate by ×0.3.
```bash
python scripts/stages/train_stage3_joint.py \
    --resume_policy logs/stage1/best.pt \
    --resume_encoder logs/stage2/best.pt \
    --max_iterations 3000 --num_envs 2048 \
    --logdir logs/stage3
```

Expected wall-clock on RTX 4090: ~6 h stage 1, ~1 h stage 2, ~3 h stage 3.

---

## 13. Evaluation in Isaac Lab + MuJoCo

### Isaac Lab playback
```bash
python scripts/rsl_rl/play.py --task Unitree-G1-23dof-StairClimb-v0 \
    --checkpoint logs/stage3/best.pt --num_play_envs 16
# or:
./unitree_dsc_lab.sh -p --task Unitree-G1-23dof-StairClimb-v0 \
    --checkpoint logs/stage3/best.pt
```
Reproduce **Table I** numbers: `E_vel`, `E_ang`, `M_terrain`, success rate.

### MuJoCo cross-sim sanity check (Fig. 4)
```bash
pip install mujoco==3.2.4 dm_control
python scripts/rsl_rl/mujoco_eval.py \
    --policy logs/stage3/best.pt \
    --xml assets/g1/unitree_ros/robots/g1_description/g1_23dof.xml \
    --vel_commands 0.0,0.4,0.7,0.5,0.0
```
(`mujoco_eval.py` is not yet scaffolded — add alongside `play.py` when needed.)

If MuJoCo fails but Isaac Lab passes, suspect joint-order mismatch or PD-gain mismatch.

---

## 14. Sim-to-Real Bridge (ONNX Export)

```bash
python scripts/rsl_rl/export_onnx.py \
    --policy logs/stage3/best.pt \
    --encoder logs/stage3/encoder.pt \
    --out_dir deploy \
    --opset 17
```

Verify with `onnxruntime`:
```python
import onnxruntime as ort
sess = ort.InferenceSession("deploy/policy.onnx",
                            providers=["CUDAExecutionProvider"])
print([i.name for i in sess.get_inputs()])
```

Export the **encoder** and **policy** as two separate ONNX files so the on-board node can run encoder at 10 Hz and policy at 50 Hz (matches Fig. 2). The runtime YAML consumed by the C++/Python deploy nodes is generated via [`utils/export_deploy_cfg.py`](source/unitree_dsc_lab/unitree_dsc_lab/utils/export_deploy_cfg.py).

---

## 15. Real-World Deployment on Unitree G1

### 15.1 On-board software
On the Jetson Orin:
```bash
# ROS 2 Humble
sudo apt install ros-humble-desktop ros-humble-realsense2-camera

# unitree_sdk2 + python bindings
git clone https://github.com/unitreerobotics/unitree_sdk2.git
cd unitree_sdk2 && mkdir build && cd build
cmake .. && make -j4 && sudo make install
pip install unitree_sdk2py onnxruntime-gpu
```

### 15.2 Runtime architecture
```
RealSense depth (30 Hz)
        │
        ▼
deploy/pointcloud_to_bev.py  ─► BEV 6×60×60 ──► encoder.onnx (10 Hz) ──► z_t
                                                                          │
unitree state subscriber (500 Hz) ──► proprio buffer ──┐                  │
                                                        ▼                  ▼
                                              deploy/g1_runtime.py (policy.onnx, 50 Hz)
                                                        │
                                                        ▼
                                         joint targets ─► unitree_sdk2 LowCmd
```
The C++ controller under `deploy/robots/g1_23dof/` mirrors `unitree_rl_lab/deploy/robots/g1_29dof/` and links against `unitree_sdk2` + `onnxruntime`.

### 15.3 Safety bring-up (mandatory order)
1. Robot on gantry, EMO in reach.
2. Run policy in **damping mode** — publish zero torques, log inferred actions for 30 s.
3. Switch to **low-gain track** (50 % Kp) for 1 m flat walk.
4. One-step climb on a 12 cm box.
5. Five-step indoor staircase (paper §IV-D.1).
6. Outdoor long staircase only after steps 1–5 succeed three runs in a row.

### 15.4 Launch
```bash
ros2 launch deploy/launch/g1_stair.launch.py \
    encoder_onnx:=/opt/unitree_dsc_lab/encoder.onnx \
    policy_onnx:=/opt/unitree_dsc_lab/policy.onnx \
    cmd_topic:=/cmd_vel
```
Send forward velocity command:
```bash
ros2 topic pub /cmd_vel geometry_msgs/Twist "{linear: {x: 0.3}}"
```

---

## 16. Tuning, Safety Checklist, Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Robot scuffs on step edge | swing clearance reward too low or `h_step` underestimated | Increase `swing_clearance_bonus`; check encoder MAE on real depth |
| Yaws away from stair | `theta_yaw` channel noisy | Add IMU yaw filter; widen domain rand on yaw init |
| Works in sim, falls in real | PD gain mismatch | Match `Kp/Kd` to `unitree_sdk2` defaults; verify joint order |
| Encoder predicts `h=0` on real stairs | BEV empty / depth out of range | Check depth min/max clip, extrinsic calibration, BEV frame |
| PPO reward plateaus low | reward weights | Re-read paper §III-C, `α=1`, `λ_cls=0.6`, `λ_h=λ_d=1` |

### Pre-flight checklist (every real run)
- [ ] Battery > 60 %
- [ ] EMO tested
- [ ] Encoder ONNX MAE checked against tape-measured stair
- [ ] Gantry/harness if novel geometry
- [ ] Logging on (`ros2 bag record -a`)

---

## 17. References

- Paper: arXiv:2605.09944v1 — *Explicit Stair Geometry Conditioning for Robust Humanoid Locomotion*
- Isaac Lab: https://github.com/isaac-sim/IsaacLab
- Unitree SDK2: https://github.com/unitreerobotics/unitree_sdk2
- unitree_rl_gym (G1 reference env): https://github.com/unitreerobotics/unitree_rl_gym
- MoRE baseline (Table III/IV): Wang et al., 2025

---

*Last updated: 2026-05-17 — §8 follow-up wired (`StairTerrainGenerator` + `(row, col)` GT registry, `PrivilegedTeacher.refresh_from_terrain()`, `mdp.reset_privileged_teacher` event term, `G1StairClimbEnvCfg` scaffolded with the three class-locked presets mixed by proportion); §9 / §11 prose refreshed to match current code state and call out the remaining gaps (BEV bodies, `swing_clearance_bonus`/`step_alignment_bonus`, `foot_contact_flags` obs). Tested with Isaac Lab v2.3.2, Isaac Sim 5.1.0, Python 3.11, PyTorch 2.6.0+cu124, Unitree G1 firmware 1.4.*

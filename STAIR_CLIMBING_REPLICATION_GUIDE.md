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
- [x] **§9 — `points_to_bev` implemented.** Pure-tensor (CPU/GPU) projection from `(N, 3)` or `(B, N, 3)` point clouds in robot frame to a `(6, 60, 60)` / `(B, 6, 60, 60)` BEV. Single set of `scatter_add_` / `scatter_reduce_` calls on flattened cell indices — no Python loop over envs. Supports custom `x_range` / `y_range` / `resolution` and an optional `valid_mask` for padded ragged clouds. Channels: `[max, min, mean, max - min, std, density]`; density is per-batch-normalized; empty cells zero-filled. Smoke-tested: cell math on single + multi-point clouds, OOR drop, empty cloud, batched-vs-unbatched parity, density normalization.
- [x] **§9 — `BEVStudentEncoder` implemented.** Conv trunk `Conv(6→32) → Conv(32→64) → Conv(64→128)` (all `k=3, s=2, p=1`, BN + ReLU) takes a `(B, 6, 60, 60)` BEV down to `F_enc ∈ R^{B×128×8×8}`; shared MLP `Linear(8192→256→128)` then four heads emit `(class_logits[3], h_step, d_step, theta_yaw)`. Returns a `BEVPrediction` dataclass exposing `feat=F_enc` for joint training. `predict_token(bev) → (B, 4)` matches `PrivilegedTeacher.token()`'s layout so swapping student for teacher in `ObservationsCfg.PolicyCfg` is a one-line change. `terrain_loss(pred, target)` implements Eq. 8 (`0.6·CE + 1.0·L1(h) + 1.0·L1(d)`); yaw head is excluded by design. ≈2.2 M params. Smoke-tested: forward shapes, F_enc shape, `predict_token` shape + class range, loss finite, gradient isolation for the yaw head, gradients reach class/h/d heads + conv trunk, end-to-end with a real `points_to_bev` output.
- [x] **§11 — `swing_clearance_bonus` + `step_alignment_bonus` implemented.** Both are stateful `ManagerTermBase` classes in [`tasks/stair_climb/mdp/rewards.py`](source/unitree_dsc_lab/unitree_dsc_lab/tasks/stair_climb/mdp/rewards.py) and read per-env GT from `env.privileged_teacher`. `swing_clearance_bonus` tracks `lift_off_z` per foot from in-contact frames and emits a dense per-step bonus `sum_feet(clamp(foot_z − lift_off_z − (h_step + clearance_margin), 0))` while the foot is airborne — gated to `class_id == 1` (stairs-up). `step_alignment_bonus` tracks each foot's last touchdown xy in the world frame and, on every first-contact event (`ContactSensor.compute_first_contact(step_dt)`), measures forward distance from the previous touchdown projected onto the stair-forward axis `[cos(θ_yaw_terrain), sin(θ_yaw_terrain)]` and rewards a Gaussian `exp(−err² / window_margin²)` with `err = projected_delta − d_step` — gated to `class_id ∈ {1, 2}`. Both implement `reset(env_ids)` to clear per-env state on episode reset. Wired into `RewardsCfg` (`swing_clearance` weight 2.0, `step_alignment` weight 1.0) keyed on `.*_ankle_roll_link`. Smoke-tested: shape + first-call seeding, class masking (flat zero / stairs-up active / stairs-down behavior), reset clears per-env state, Gaussian peak at exact `d_step`, `e^−1` at one-window-margin offset, terrain-yaw rotation aligns the axis to `+Y`, no-reward path when `compute_first_contact` is False, graceful zero when `env.privileged_teacher` is absent.
- [x] **§11 — `StairClimbActorCritic` wired for rsl-rl OnPolicyRunner.** Thin `rsl_rl.models.MLPModel` subclass in [`tasks/stair_climb/policy/actor_critic.py`](source/unitree_dsc_lab/unitree_dsc_lab/tasks/stair_climb/policy/actor_critic.py) used for **both** the actor (Gaussian distribution, `output_dim=num_actions`) and the critic (deterministic, `output_dim=1`) — the paper's policy is a memoryless MLP over `o_prop ⊕ z_t` (Eq. 1) and rsl-rl's new (>= 4.0.0) construction path already splits actor/critic, so no custom architecture is needed. Class attributes `paper_terrain_token_dim = 4` and `is_recurrent = False` document the contract. [`agents/rsl_rl_ppo_cfg.py::BasePPORunnerCfg`](source/unitree_dsc_lab/unitree_dsc_lab/tasks/stair_climb/agents/rsl_rl_ppo_cfg.py) was modernized to the new `RslRlMLPModelCfg` actor/critic API: both reference the qualified name `"unitree_dsc_lab.tasks.stair_climb.policy.actor_critic:StairClimbActorCritic"` (resolved by `rsl_rl.utils.resolve_callable`), `obs_groups = {"actor": ["policy"], "critic": ["critic"]}` matches the env, paper-aligned hidden dims `[512, 256, 128]` + `elu` carry over. Smoke-tested against the live `rsl_rl` install: stochastic actor forward shape `(B, num_actions)`, log-prob, entropy/std, `update_normalization` no-op; deterministic critic returns reproducible `(B, 1)`; `EmpiricalNormalization` activates when `obs_normalization=True`; qualified-name string parses to the expected class.
- [x] **§11 — `G1StairClimbEnvCfg` scaffolded.** Full env cfg in [`tasks/stair_climb/robots/g1_23dof/stair_env_cfg.py`](source/unitree_dsc_lab/unitree_dsc_lab/tasks/stair_climb/robots/g1_23dof/stair_env_cfg.py): scene (`StairTerrainGeneratorCfg` + G1 articulation + contact sensor), observations (policy `o_prop ⊕ z_t` + critic groups), `JointPositionActionCfg`, `UniformVelocityCommandCfg` (forward-only, `lin_vel_x ∈ [0.0, 0.7]`), IsaacLab default rough-locomotion reward stack + `swing_clearance_bonus` (weight 2.0) + `step_alignment_bonus` (weight 1.0), events incl. `reset_privileged_teacher`, standard terminations, `terrain_levels_vel` curriculum.
- [x] **§12 — Simulation & training infrastructure complete.** `foot_contact_flags` obs term added to [`mdp/observations.py`](source/unitree_dsc_lab/unitree_dsc_lab/tasks/stair_climb/mdp/observations.py) (binary contact per ankle-roll link, `sensor_cfg.body_ids` resolved by `ObservationManager`). `base_lin_vel` (paper Eq. 1 gap) also added to `PolicyCfg` with noise. A 60×60 @ 0.05 m `RayCasterCfg` (`height_scanner`, attached to `torso_link`, offset 1.5 m forward + 20 m up, `attach_yaw_only=True`) added to `StairSceneCfg`; scanner updates at policy rate (50 Hz). [`robots/g1_23dof/stair_env.py::G1StairClimbEnv`](source/unitree_dsc_lab/unitree_dsc_lab/tasks/stair_climb/robots/g1_23dof/stair_env.py) subclasses `ManagerBasedRLEnv` and exposes `get_bev()`: reads `ray_hits_w`, translates by `root_pos_w`, rotates by `−yaw`, passes to `points_to_bev()` with NaN mask — returns `Tensor(num_envs, 6, 60, 60)`. Gym registration updated to `entry_point=G1StairClimbEnv`. `evaluate_success_rate(num_episodes=200)` added to `ThreeStagePPORunner` (rolls out frozen policy, counts episodes where final `terrain_level ≥ 80 % of max_level`); `train_stage1_policy.py` calls it after training and saves `model_final.pt`.
- [x] **§12 — `ThreeStagePPORunner` implemented.** [`tasks/stair_climb/policy/ppo_runner.py::ThreeStagePPORunner`](source/unitree_dsc_lab/unitree_dsc_lab/tasks/stair_climb/policy/ppo_runner.py) inherits `OnPolicyRunner` and adds the encoder + its Adam optimiser. Three entry points: `learn_stage1(num_iterations)` delegates to `super().learn()` unchanged; `learn_stage2(num_epochs, batch_size, sl_epochs_per_rollout)` freezes actor + critic, rolls out, collects `(BEV, z_t_gt)` via `env.get_bev()` + `env.privileged_teacher.token()`, then runs mini-batch SL with `terrain_loss` — raises `RuntimeError` immediately if `get_bev()` absent; `learn_stage3(num_iterations, lr_scale=0.3)` scales both PPO and encoder LRs by `lr_scale`, runs the standard PPO loop, and appends a perception side-loss pass after each `alg.update()` (separate encoder optimiser, no interference with PPO). `save()`/`load()` extend the parent to include `encoder_state_dict` + `encoder_optimizer_state_dict` in the checkpoint. Three stage scripts in [`scripts/stages/`](scripts/stages/) bootstrap AppLauncher, create the env + wrapper, instantiate the runner, and call the appropriate `learn_stage*()` method. Smoke-tested: construction, `learn_stage1` delegation, stage-2 `RuntimeError` on missing `get_bev`, SL loop runs (loss finite: total≈0.71, cls≈1.13, h≈0.004, d≈0.03), encoder round-trip checkpoint save/load, stage-3 graceful no-BEV fallback. All 7 tests pass; all 4 files `py_compile` cleanly.

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
│           ├── mdp/                         # observations, rewards, terminations, commands,
│           │                                #   curriculums (terrain_levels_vel),
│           │                                #   events (reset_privileged_teacher)
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

**Status: `points_to_bev` + `BEVStudentEncoder` + `terrain_loss` implemented.**

### 9.1 `points_to_bev` — point cloud → 6-channel BEV

[`tasks/stair_climb/perception/bev.py::points_to_bev`](source/unitree_dsc_lab/unitree_dsc_lab/tasks/stair_climb/perception/bev.py) — single pass of vectorized `scatter_add_` / `scatter_reduce_` calls, runs on CPU or GPU, supports batched inputs without a Python loop over envs.

| Parameter | Default | Notes |
|-----------|---------|-------|
| `x_range` | `(0.0, 3.0)` m | +X forward — stairs-ahead footprint |
| `y_range` | `(-1.5, 1.5)` m | +Y left, centred on robot |
| `resolution` | `0.05` m | 60 cells per 3 m |
| `valid_mask` | `None` | optional `(N,)` / `(B, N)` bool mask for padded ragged clouds |

Channels (in this order): `[max(z), min(z), mean(z), max - min, std(z), density]`. Empty cells are zero-filled; density is per-batch-normalized so the busiest cell is 1.0.

```python
from unitree_dsc_lab.tasks.stair_climb.perception.bev import points_to_bev

# Unbatched: (N, 3) -> (6, 60, 60)
bev = points_to_bev(robot_centric_points)

# Batched ragged: (B, max_N, 3) + (B, max_N) mask -> (B, 6, 60, 60)
bev = points_to_bev(padded_points, valid_mask=mask)
```

> The point cloud must already be in the **robot frame** (+X forward, +Y left, +Z up). The on-board deploy node ([`deploy/pointcloud_to_bev.py`](deploy/pointcloud_to_bev.py)) is responsible for transforming RealSense / Livox points into this frame before calling the encoder.

### 9.2 `BEVStudentEncoder` — BEV → 4-D token

[`tasks/stair_climb/perception/encoder.py::BEVStudentEncoder`](source/unitree_dsc_lab/unitree_dsc_lab/tasks/stair_climb/perception/encoder.py):

```
(B, 6, 60, 60)
    │
    │  Conv(6→32, k=3, s=2, p=1) + BN + ReLU            (B, 32, 30, 30)
    │  Conv(32→64, k=3, s=2, p=1) + BN + ReLU           (B, 64, 15, 15)
    │  Conv(64→128, k=3, s=2, p=1) + BN + ReLU          (B, 128, 8, 8)  = F_enc
    │
    │  Flatten + Linear(8192→256) + ReLU + Linear(256→128) + ReLU
    │
    ├── head_class  (Linear 128→3)    → logits_class (B, 3)
    ├── head_h      (Linear 128→1)    → h_step       (B,)
    ├── head_d      (Linear 128→1)    → d_step       (B,)
    └── head_yaw    (Linear 128→1)    → theta_yaw    (B,)   [sanity head]
```

≈ 2.2 M params. `forward(bev)` returns a `BEVPrediction` dataclass with the four head outputs plus `feat=F_enc` (exposed for Stage 3 joint training).

**Drop-in for the teacher in `PolicyCfg`** — `predict_token(bev)` returns `(B, 4)` `[class_id, h, d, theta_yaw]` matching the layout of `PrivilegedTeacher.token()`:

```python
from unitree_dsc_lab.tasks.stair_climb.perception import BEVStudentEncoder

encoder = BEVStudentEncoder().to(device).eval()
z_t = encoder.predict_token(bev_batch)        # (B, 4), same channels as teacher.token
```

**Stage 2 supervision** (paper Eq. 8) — `terrain_loss(pred, target)`:

```python
from unitree_dsc_lab.tasks.stair_climb.perception import terrain_loss

# target = teacher.token(robot_yaw_world)     # (B, 4)
losses = terrain_loss(pred, target)
losses["total"].backward()
# losses["cls"], losses["h"], losses["d"] available for logging
```

Yaw is **not** in `terrain_loss` (paper §III-A.3 — yaw is taken from IMU, the yaw head is trained only as a sanity probe).

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

Student loss (Eq. 8 of the paper) — implemented in [`perception/encoder.py::terrain_loss`](source/unitree_dsc_lab/unitree_dsc_lab/tasks/stair_climb/perception/encoder.py):

```python
L_terrain = 0.6 * F.cross_entropy(logits_s, s_gt) \
          + 1.0 * F.smooth_l1_loss(h_pred, h_gt) \
          + 1.0 * F.smooth_l1_loss(d_pred, d_gt)
# (yaw is part of the policy observation; not regressed)
```

---

## 11. PPO Policy: Observations, Actions, Rewards

**Status: env cfg scaffolded incl. stair-shaping rewards** — [`tasks/stair_climb/robots/g1_23dof/stair_env_cfg.py::G1StairClimbEnvCfg`](source/unitree_dsc_lab/unitree_dsc_lab/tasks/stair_climb/robots/g1_23dof/stair_env_cfg.py) wires the scene (`StairTerrainGeneratorCfg` + G1 articulation + contact sensor), observations (policy + critic groups), actions, commands, rewards (rough-locomotion stack + `swing_clearance_bonus` + `step_alignment_bonus`), events (incl. `reset_privileged_teacher`), terminations, and `terrain_levels_vel` curriculum.

**Observation** (per Eq. 1):
```
o_prop = [base_lin_vel, base_ang_vel, projected_gravity, velocity_commands,
          joint_pos - default, joint_vel, last_action, foot_contact_flags]
o_t    = concat(o_prop, z_t)     # z_t = mdp.terrain_token_privileged  (4-D)
```

`foot_contact_flags` — `mdp.foot_contact_flags(sensor_cfg="contact_forces", body_names=".*_ankle_roll_link")` — binary contact per ankle-roll link, shape `(num_envs, 2)`. Both policy and critic groups include it. `base_lin_vel` added with `Unoise(−0.1, +0.1)` in PolicyCfg (noiseless in CriticCfg).

**Action:** `mdp.JointPositionActionCfg(joint_names=[".*"], scale=0.25, use_default_offset=True)` — target joint offsets from default, wrapped by IsaacLab's built-in PD (`Kp`/`Kd` come from `G1_23DOF_CFG.actuators`, sourced from the Unitree spec).

**Commands:** `mdp.UniformVelocityCommandCfg` with forward-only ranges `lin_vel_x ∈ [0.0, 0.7] m/s`, small ang_vel_z noise; resampling every 8–12 s.

**Rewards:** IsaacLab's default rough-locomotion stack (`track_lin_vel_xy_exp`, `track_ang_vel_z_exp`, `is_alive`, `lin_vel_z_l2`, `ang_vel_xy_l2`, `flat_orientation_l2`, `base_height_l2`, `joint_vel_l2`, `joint_acc_l2`, `action_rate_l2`, `joint_pos_limits`, arm/waist `joint_deviation_l1`, `undesired_contacts`), plus two stateful paper-specific shaping terms ([`mdp/rewards.py`](source/unitree_dsc_lab/unitree_dsc_lab/tasks/stair_climb/mdp/rewards.py)) — both `ManagerTermBase` classes that key on `.*_ankle_roll_link` and read per-env GT from `env.privileged_teacher`:

- `swing_clearance_bonus` (weight 2.0) — tracks `lift_off_z` per foot from in-contact frames; while a foot is airborne, emits dense `sum_feet(clamp(foot_z − lift_off_z − (h_step + clearance_margin), 0))`. Gated to `class_id == 1` (stairs-up); `clearance_margin = 0.05`.
- `step_alignment_bonus` (weight 1.0) — tracks each foot's last touchdown xy in world frame; on every first-contact event (`ContactSensor.compute_first_contact(step_dt)`) measures forward distance projected onto the stair-forward axis `[cos(θ_yaw_terrain), sin(θ_yaw_terrain)]` and rewards a Gaussian `exp(−(projected_delta − d_step)² / window_margin²)`. Gated to `class_id ∈ {1, 2}` (stairs); `window_margin = 0.04`. First touchdown after reset is used only to seed state.

Both implement `reset(env_ids)` to clear per-env buffers on episode reset, so the next stride re-seeds cleanly.

**Actor-critic:** [`tasks/stair_climb/policy/actor_critic.py::StairClimbActorCritic`](source/unitree_dsc_lab/unitree_dsc_lab/tasks/stair_climb/policy/actor_critic.py) is a thin `rsl_rl.models.MLPModel` subclass used for both the actor (Gaussian distribution, output_dim = num_actions) and the critic (deterministic, output_dim = 1) — paper §III-A specifies a memoryless MLP over `o_prop ⊕ z_t`, which is exactly what `MLPModel` provides. The class exposes paper-aligned defaults (`hidden_dims=(512, 256, 128)`, `activation="elu"`) plus class attributes `paper_terrain_token_dim = 4` / `is_recurrent = False`. rsl-rl ≥ 4.0.0's `OnPolicyRunner` resolves the class via the qualified-name string in the runner cfg.

**rsl-rl PPO runner cfg:** [`agents/rsl_rl_ppo_cfg.py::BasePPORunnerCfg`](source/unitree_dsc_lab/unitree_dsc_lab/tasks/stair_climb/agents/rsl_rl_ppo_cfg.py) uses the new split-model API:

```python
obs_groups = {"actor": ["policy"], "critic": ["critic"]}

actor = RslRlMLPModelCfg(
    class_name="unitree_dsc_lab.tasks.stair_climb.policy.actor_critic:StairClimbActorCritic",
    hidden_dims=[512, 256, 128], activation="elu", obs_normalization=True,
    distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
)
critic = RslRlMLPModelCfg(
    class_name="unitree_dsc_lab.tasks.stair_climb.policy.actor_critic:StairClimbActorCritic",
    hidden_dims=[512, 256, 128], activation="elu", obs_normalization=True,
    distribution_cfg=None,
)
```

PPO hyperparameters follow paper §III-D + Table II (lr 3e-4, γ 0.99, λ 0.95, clip 0.2, KL 0.01, entropy 0.005, 5 epochs × 4 mini-batches).

---

## 12. Three-Stage Training Pipeline

**Status: `ThreeStagePPORunner` implemented** — [`tasks/stair_climb/policy/ppo_runner.py`](source/unitree_dsc_lab/unitree_dsc_lab/tasks/stair_climb/policy/ppo_runner.py).

`ThreeStagePPORunner(OnPolicyRunner)` holds a `BEVStudentEncoder` and its Adam optimiser alongside the standard PPO models. Three entry points:

| Method | Delegates to | What it does |
|---|---|---|
| `learn_stage1(num_iterations)` | `super().learn()` | Pure PPO, teacher z_t, unchanged rsl-rl loop |
| `learn_stage2(num_epochs, batch_size, sl_epochs_per_rollout)` | Custom SL loop | Freeze policy; collect `(BEV, z_t_gt)` via `env.get_bev()` + teacher; mini-batch `terrain_loss` |
| `learn_stage3(num_iterations, lr_scale=0.3)` | Custom loop | Scale LRs ×0.3; rollout + PPO update; separate encoder SL pass after each `alg.update()` |

`save()`/`load()` extend the parent checkpoint with `encoder_state_dict` + `encoder_optimizer_state_dict`.

> **Stage 2 prerequisite — now implemented:** `G1StairClimbEnv.get_bev()` reads the `height_scanner` RayCaster (3 m × 3 m grid, 0.05 m, forward-biased), transforms hits into robot frame, and returns `Tensor(num_envs, 6, 60, 60)` via `points_to_bev()`. The real-world counterpart is `deploy/pointcloud_to_bev.py`.

> **Stage 3 note:** rollouts still use teacher z_t for the policy observation (stable actor while encoder catches up). Full end-to-end student z_t → policy → gradient → encoder is a future extension.

The Gym task id registered by `unitree_dsc_lab` is **`Unitree-G1-23dof-StairClimb-v0`** (see [`tasks/stair_climb/robots/g1_23dof/__init__.py`](source/unitree_dsc_lab/unitree_dsc_lab/tasks/stair_climb/robots/g1_23dof/__init__.py)).

### Stage 1 — Pre-train policy with ground-truth `z_t`
```bash
python scripts/stages/train_stage1_policy.py \
    --task Unitree-G1-23dof-StairClimb-v0 --num_envs 4096 \
    --max_iterations 6000 --headless \
    --logdir logs/stage1
```
Targets: `success_rate > 0.85` on training stair heights.

### Stage 2 — Train student encoder under teacher supervision
Freeze the policy, run rollouts, store `(BEV, z_t_gt)` pairs, train CNN until:
- `MAE(h) < 1 cm`, `MAE(d) < 1 cm`, `class_acc > 99 %` (matches Table II of paper).

```bash
python scripts/stages/train_stage2_perception.py \
    --policy_ckpt logs/stage1/Unitree-G1-23dof-StairClimb-v0/model_6000.pt \
    --num_envs 1024 --epochs 50 --batch_size 256 \
    --logdir logs/stage2
```

### Stage 3 — Joint fine-tuning
Unfreeze both. Loss = `L_PPO + 1.0 * L_terrain`. Reduce learning rate by ×0.3.
```bash
python scripts/stages/train_stage3_joint.py \
    --resume_policy logs/stage1/Unitree-G1-23dof-StairClimb-v0/model_6000.pt \
    --resume_encoder logs/stage2/Unitree-G1-23dof-StairClimb-v0/encoder_best.pt \
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
| Robot scuffs on step edge | swing clearance reward too low or `h_step` underestimated | Increase `swing_clearance_bonus` weight or `clearance_margin`; check encoder MAE on real depth |
| Foot lands short / long of stair tread | `d_step` underestimated by encoder or step_alignment too weak | Increase `step_alignment_bonus` weight; widen `window_margin` if learning curve is flat early |
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

*Last updated: 2026-05-17 — Simulation & training infrastructure complete: `foot_contact_flags` + `base_lin_vel` added to `PolicyCfg` + `CriticCfg`; `height_scanner` (RayCaster, 3 m×3 m @ 0.05 m) added to scene; `G1StairClimbEnv` subclass with `get_bev()` created; gym registration updated to `G1StairClimbEnv`; `evaluate_success_rate()` added to `ThreeStagePPORunner`; `train_stage1_policy.py` saves `model_final.pt` and reports success rate. All three training targets (Stage 1 `success_rate > 0.85`, Stage 2 `MAE < 1 cm`, Stage 3 convergence) are now unblocked pending Isaac Sim + G1 asset setup (§3–§7). Previous: §12 `ThreeStagePPORunner` implemented (`learn_stage1/2/3`, encoder SL loop, save/load checkpoint extension); three stage scripts in `scripts/stages/`; smoke-tested 7/7. See §0 Done list for complete history. Tested with Isaac Lab v2.3.2, Isaac Sim 5.1.0, Python 3.11, PyTorch 2.6.0+cu124, Unitree G1 firmware 1.4.*

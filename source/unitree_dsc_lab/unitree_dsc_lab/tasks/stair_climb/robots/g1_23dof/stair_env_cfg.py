"""Isaac Lab env config for stair-climbing on Unitree G1 (23-DoF).

The terrain is built by :class:`StairTerrainGenerator` from three class-locked
sub-terrains (flat / stairs-up / stairs-down) mixed by proportion, with a
per-row curriculum on step height. Each tile's GT ``(class_id, h_step, d_step,
theta_yaw_terrain)`` is recorded in :class:`StairGTRegistry` keyed by
``(row, col)`` at build time. The reset-mode event
``mdp.reset_privileged_teacher`` attaches ``env.privileged_teacher`` on first
call and refreshes it per env on every reset, so the observation term
``mdp.terrain_token_privileged`` always sees up-to-date ground truth.

If you are running the 29-DoF G1, the wrist joints must be frozen here so the
action dim stays at 23 (paper §III-A).
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_dsc_lab.assets.robots.unitree import G1_23DOF_CFG as ROBOT_CFG
from unitree_dsc_lab.tasks.stair_climb import mdp
from unitree_dsc_lab.tasks.stair_climb.terrains import (
    FlatLeadInCfg,
    StairsDownCfg,
    StairsUpCfg,
    StairTerrainGeneratorCfg,
)


# ---------------------------------------------------------------------------
# Stair terrain config — paper §III-B
# ---------------------------------------------------------------------------

STAIR_TERRAIN_CFG = StairTerrainGeneratorCfg(
    size=(8.0, 4.0),
    border_width=4.0,
    num_rows=10,        # 10 difficulty levels (curriculum along rows)
    num_cols=20,        # 20 tiles per row, mixed across the 3 classes
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "flat": FlatLeadInCfg(proportion=0.2),
        "stairs_up": StairsUpCfg(proportion=0.4),
        "stairs_down": StairsDownCfg(proportion=0.4),
    },
)


@configclass
class StairSceneCfg(InteractiveSceneCfg):
    """Scene = stair terrain + G1 robot + contact sensor."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=STAIR_TERRAIN_CFG,
        max_init_terrain_level=STAIR_TERRAIN_CFG.num_rows - 1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )


# ---------------------------------------------------------------------------
# Commands — forward velocity 0-0.7 m/s (paper §III-C)
# ---------------------------------------------------------------------------


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.7),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-0.2, 0.2),
        ),
    )


# ---------------------------------------------------------------------------
# Actions — joint position offset from default, wrapped by PD
# ---------------------------------------------------------------------------


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
    )


# ---------------------------------------------------------------------------
# Observations — proprio + 4-D terrain token
# ---------------------------------------------------------------------------


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Per-step policy input (paper Eq. 1)."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)

        # 4-D terrain token z_t. Stage 1 = teacher GT; stage 2/3 swaps for the
        # student encoder (BEV → CNN → MLP heads).
        terrain_token = ObsTerm(func=mdp.terrain_token_privileged)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)
        terrain_token = ObsTerm(func=mdp.terrain_token_privileged)

        def __post_init__(self):
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# ---------------------------------------------------------------------------
# Rewards — rough-locomotion stack (paper says use IsaacLab defaults)
# ---------------------------------------------------------------------------


@configclass
class RewardsCfg:
    # task
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    alive = RewTerm(func=mdp.is_alive, weight=0.15)

    # base
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-10.0, params={"target_height": 0.78})

    # joint regularization
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)

    # arm / waist keep-still (G1)
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*"],
            ),
        },
    )
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist.*"])},
    )

    # contact safety — penalize torso/arm hits with the staircase
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )

    # NOTE: paper-specific shaping terms (swing_clearance_bonus,
    # step_alignment_bonus) live in `mdp/rewards.py` as stubs — drop them in
    # here once implemented.


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


@configclass
class EventCfg:
    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.2),
            "dynamic_friction_range": (0.5, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "yaw": (-0.5, 0.5)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (-0.5, 0.5)},
    )

    # Refresh per-env terrain GT (and attach env.privileged_teacher on first
    # call). Must run on every reset so the teacher token matches each env's
    # newly assigned tile.
    reset_privileged_teacher = EventTerm(
        func=mdp.reset_privileged_teacher,
        mode="reset",
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 12.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
    )


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


# ---------------------------------------------------------------------------
# Curriculum — terrain level driven by velocity tracking
# ---------------------------------------------------------------------------


@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


# ---------------------------------------------------------------------------
# Env config
# ---------------------------------------------------------------------------


@configclass
class G1StairClimbEnvCfg(ManagerBasedRLEnvCfg):
    """Training-mode env config (paper §III)."""

    scene: StairSceneCfg = StairSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        self.decimation = 4                 # 200 Hz physics / 50 Hz policy
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.contact_forces.update_period = self.sim.dt

        # tie terrain curriculum to the terrain_levels reward
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = (
                getattr(self.curriculum, "terrain_levels", None) is not None
            )


@configclass
class G1StairClimbPlayEnvCfg(G1StairClimbEnvCfg):
    """Play-mode: fewer envs, no domain rand pushes, fewer rows/cols."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 4
        self.scene.terrain.terrain_generator.num_cols = 8
        self.events.push_robot = None

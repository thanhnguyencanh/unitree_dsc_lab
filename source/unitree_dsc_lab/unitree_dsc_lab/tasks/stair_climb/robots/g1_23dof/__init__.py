import gymnasium as gym

gym.register(
    id="Unitree-G1-23dof-StairClimb-v0",
    entry_point="unitree_dsc_lab.tasks.stair_climb.robots.g1_23dof.stair_env:G1StairClimbEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stair_env_cfg:G1StairClimbEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.stair_env_cfg:G1StairClimbPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "unitree_dsc_lab.tasks.stair_climb.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg"
        ),
    },
)

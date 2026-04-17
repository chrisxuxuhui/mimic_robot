import gymnasium as gym

from . import agents, flat_env_cfg

##
# Register Gym environments.
##

# 用于训练的环境（保持自适应采样）
gym.register(
    id="Tracking-Flat-PM01-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.PM01FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PM01FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-PM01-Wo-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.PM01FlatWoEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PM01FlatPPORunnerCfg",
    },
)

# 可选：演示模式环境（如果需要）
# gym.register(
#     id="Tracking-Flat-PM01-Play-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": flat_env_cfg.PM01FlatEnvCfg,
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PM01FlatPPORunnerCfg",
#     },
# )
"""IGRIS-C motion tracking task registration."""

import gymnasium as gym


gym.register(
    id="Isaac-IGRISC-Motion-Tracking-v0",
    entry_point=f"{__name__}.env_cfg:IGRISCMotionTrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:IGRISCMotionTrackingEnvCfg",
    },
)

"""Register the IGRIS-C HugWBC playback task."""

import gymnasium as gym


gym.register(
    id="Robros-IGRIS-C-Flat-Teleop-Lowerbody-Action",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:IGRISCHugWBCFlatEnvCfg",
    },
)

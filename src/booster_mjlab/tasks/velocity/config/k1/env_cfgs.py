"""Booster K1 velocity tracking environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.velocity import mdp

from booster_mjlab.robots import K1_ACTION_SCALE, get_k1_robot_cfg
from booster_mjlab.robots.booster_k1.k1_constants import K1_FOOT_SOLE_SIZE
from booster_mjlab.tasks.velocity.config.env_cfgs import (
    booster_rough_env_cfg,
    to_flat_terrain,
)


def booster_k1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Booster K1 rough terrain velocity tracking configuration."""
    cfg = booster_rough_env_cfg(
        get_k1_robot_cfg(), K1_ACTION_SCALE, K1_FOOT_SOLE_SIZE, play=play
    )

    assert cfg.commands is not None
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, mdp.UniformVelocityCommandCfg)
    twist_cmd.viz.z_offset = 1.15

    return cfg


def booster_k1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Booster K1 flat terrain velocity tracking configuration."""
    cfg = to_flat_terrain(booster_k1_rough_env_cfg(play=play))

    if play:
        commands = cfg.commands
        assert commands is not None
        twist_cmd = commands["twist"]
        assert isinstance(twist_cmd, mdp.UniformVelocityCommandCfg)

        twist_cmd.ranges.lin_vel_x = (-2.5, 2.5)
        twist_cmd.ranges.lin_vel_y = (-2.0, 2.0)
        twist_cmd.ranges.ang_vel_z = (-3.7, 3.7)

    return cfg

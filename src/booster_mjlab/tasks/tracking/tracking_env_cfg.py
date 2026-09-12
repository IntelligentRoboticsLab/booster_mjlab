"""Booster motion tracking environment configuration."""

from booster_mjlab.terrains.contact import terrain_collisions
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers import EventTermCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.tracking_env_cfg import (
    make_tracking_env_cfg as mjlab_make_tracking_env_cfg,
)
from mjlab.utils.noise import UniformNoiseCfg as Unoise


def make_tracking_env_cfg() -> ManagerBasedRlEnvCfg:
    """Create base tracking task configuration."""
    cfg = mjlab_make_tracking_env_cfg()

    cfg.observations["actor"].terms["projected_gravity"] = ObservationTermCfg(
        func=mdp.projected_gravity,
        noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    cfg.observations["critic"].terms["projected_gravity"] = ObservationTermCfg(
        func=mdp.projected_gravity,
    )

    del cfg.events["base_com"]

    assert cfg.scene.terrain is not None
    cfg.scene.terrain.collisions = terrain_collisions()

    cfg.events["push_robot"].interval_range_s = (1.5, 4.0)
    cfg.events["push_robot"].params["velocity_range"] = {
        "x": (-0.14, 0.14),
        "y": (-0.14, 0.14),
        "z": (-0.1, 0.1),
        "roll": (-0.26, 0.26),
        "pitch": (-0.26, 0.26),
        "yaw": (-0.39, 0.39),
    }

    cfg.events["foot_friction"].params["ranges"] = (0.75, 1.25)
    cfg.events["body_friction"] = EventTermCfg(
        mode="startup",
        func=dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
            "operation": "abs",
            "ranges": (0.3, 0.6),
            "shared_random": True,
        },
    )

    cfg.events["encoder_bias"].params["bias_range"] = (-0.015, 0.015)

    cfg.events["pd_gains"] = EventTermCfg(
        mode="startup",
        func=dr.pd_gains,
        params={
            "asset_cfg": SceneEntityCfg("robot", actuator_names=".*"),
            "operation": "scale",
            "kp_range": (0.8, 1.2),
            "kd_range": (0.8, 1.2),
        },
    )
    # The trunk carries the payload, so its mass and COM move much further than the limbs'.
    cfg.events["trunk_inertia"] = EventTermCfg(
        mode="startup",
        func=dr.pseudo_inertia,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
            "alpha_range": (-0.05, 0.05),
            "t_range": (-0.05, 0.05),
        },
    )
    cfg.events["limb_inertia"] = EventTermCfg(
        mode="startup",
        func=dr.pseudo_inertia,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
            "alpha_range": (-0.05, 0.05),
            "t_range": (-0.025, 0.025),
        },
    )

    return cfg

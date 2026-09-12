"""Booster K1 velocity tracking environment configurations."""

from booster_mjlab.robots.booster_k1.sensors import (
    FootClearanceSensorCfg,
    FootSoleGridPatternCfg,
)

from booster_mjlab.robots import K1_ACTION_SCALE, get_k1_robot_cfg

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
)
from mjlab.tasks.velocity import mdp

from booster_mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def booster_k1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Booster K1 rough terrain velocity tracking configuration."""
    cfg = make_velocity_env_cfg()

    cfg.scene.entities = {"robot": get_k1_robot_cfg()}

    site_names = ("left_foot", "right_foot")
    sole_site_names = ("left_foot_sole", "right_foot_sole")
    geom_names = ("left_foot_collision", "right_foot_collision")
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_foot_link|right_foot_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    nonfoot_ground_cfg = ContactSensorCfg(
        name="non_foot_ground_contact",
        primary=ContactMatch(
            mode="body",
            entity="robot",
            pattern=r".*",
            # Exclude the bodies that are allowed to touch the ground or handled separately.
            exclude=("left_foot_link", "right_foot_link"),
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    foot_height_scan = FootClearanceSensorCfg(
        name="foot_height_scan",
        frame=tuple(
            ObjRef(type="site", name=s, entity="robot") for s in sole_site_names
        ),
        # Sample the inner sole contact area without spilling past the foot.
        pattern=FootSoleGridPatternCfg(),
        ray_alignment="yaw",
        max_distance=1.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
        debug_vis=True,
        viz=FootClearanceSensorCfg.VizCfg(
            show_rays=True,
            hit_color=(1.0, 0.0, 1.0, 0.8),
            hit_sphere_color=(1.0, 0.0, 1.0, 1.0),
        ),
    )
    cfg.scene.sensors = (
        feet_ground_cfg,
        nonfoot_ground_cfg,
        self_collision_cfg,
        foot_height_scan,
    )

    if (
        cfg.scene.terrain is not None
        and cfg.scene.terrain.terrain_generator is not None
    ):
        cfg.scene.terrain.terrain_generator.curriculum = True

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = K1_ACTION_SCALE

    cfg.viewer.body_name = "Trunk"

    assert cfg.commands is not None
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, mdp.UniformVelocityCommandCfg)
    twist_cmd.rel_standing_envs = 0.2
    twist_cmd.viz.z_offset = 1.15

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
    cfg.events["trunk_inertia"].params["asset_cfg"].body_names = ("Trunk",)
    # Every body but the trunk. Name resolution is a fullmatch, so the lookahead excludes exactly "Trunk".
    cfg.events["limb_inertia"].params["asset_cfg"].body_names = (r"(?!Trunk$).*",)

    cfg.rewards["upright"].params["asset_cfg"].body_names = ("Trunk",)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("Trunk",)

    for reward_name in [
        "foot_clearance",
        "foot_slip",
    ]:
        cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

    for metric_cfg in cfg.metrics.values():
        if "asset_cfg" in metric_cfg.params:
            metric_cfg.params["asset_cfg"].site_names = site_names

    assert cfg.curriculum is not None
    assert "command_vel" in cfg.curriculum

    # Apply play mode overrides.
    if play:
        # Effectively infinite episode length.
        cfg.episode_length_s = int(1e9)

        cfg.observations["actor"].enable_corruption = False
        # cfg.events.pop("push_robot", None)
        cfg.terminations.pop("illegal_contact", None)

        if cfg.scene.terrain is not None:
            if cfg.scene.terrain.terrain_generator is not None:
                cfg.scene.terrain.terrain_generator.curriculum = False
                cfg.scene.terrain.terrain_generator.num_cols = 5
                cfg.scene.terrain.terrain_generator.num_rows = 5
                cfg.scene.terrain.terrain_generator.border_width = 10.0

    return cfg


def booster_k1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Booster K1 flat terrain velocity tracking configuration."""
    cfg = booster_k1_rough_env_cfg(play=play)

    cfg.sim.njmax = 300
    cfg.sim.mujoco.ccd_iterations = 60
    cfg.sim.contact_sensor_maxmatch = 64
    cfg.sim.nconmax = 50

    # Switch to flat terrain.
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # Disable terrain curriculum on flat terrain.
    assert cfg.curriculum is not None
    cfg.curriculum.pop("terrain_levels", None)

    if play:
        commands = cfg.commands
        assert commands is not None
        twist_cmd = commands["twist"]
        assert isinstance(twist_cmd, mdp.UniformVelocityCommandCfg)

        twist_cmd.ranges.lin_vel_x = (-2.5, 2.5)
        twist_cmd.ranges.lin_vel_y = (-2.0, 2.0)
        twist_cmd.ranges.ang_vel_z = (-3.7, 3.7)

    return cfg

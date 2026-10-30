"""Flat tracking environment configurations shared by the Booster robots.

The Booster MJCFs share their trunk, foot and collision geom names, so a robot's
config only has to supply its entity, action scale, foot sole size and the bodies
to track.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.velocity.mdp import foot_height

from booster_mjlab.robots.sensors import (
    FootClearanceSensorCfg,
    FootSoleGridPatternCfg,
)
from booster_mjlab.robots.torque_speed import (
    BoosterEntityCfg,
    initialize_torque_speed_limits,
)
from booster_mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


def booster_flat_tracking_env_cfg(
    robot_cfg: BoosterEntityCfg,
    action_scale: dict[str, float],
    foot_sole_size: tuple[float, float],
    body_names: tuple[str, ...],
    ee_body_names: tuple[str, ...],
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create a Booster flat terrain tracking configuration.

    Args:
      robot_cfg: The robot entity, with torque-speed limits.
      action_scale: Joint position action scale per joint-name regex.
      foot_sole_size: Sole area (x, y) sampled by the foot clearance scan.
      body_names: Bodies the motion command tracks.
      ee_body_names: End-effector bodies whose drift from the motion terminates.
      play: Apply the play-mode overrides.
    """
    cfg = make_tracking_env_cfg()

    cfg.scene.entities = {"robot": robot_cfg}

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
            ObjRef(type="site", name=s, entity="robot")
            for s in ("left_foot_sole", "right_foot_sole")
        ),
        # Sample the inner sole contact area without spilling past the foot.
        pattern=FootSoleGridPatternCfg(size=foot_sole_size),
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
    cfg.scene.sensors = (self_collision_cfg, foot_height_scan)
    cfg.observations["critic"].terms["foot_height"] = ObservationTermCfg(
        func=foot_height,
        params={"sensor_name": "foot_height_scan"},
    )

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = action_scale

    cfg.events["initialize_torque_speed_limits"] = EventTermCfg(
        mode="startup", func=initialize_torque_speed_limits
    )

    assert cfg.commands is not None
    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.anchor_body_name = "Trunk"
    motion_cmd.body_names = body_names

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
        "left_foot_collision",
        "right_foot_collision",
    )
    cfg.events["body_friction"].params["asset_cfg"].geom_names = (
        r"(?!(left|right)_foot_collision$).*_collision",
    )
    cfg.events["trunk_inertia"].params["asset_cfg"].body_names = ("Trunk",)
    # Every body but the trunk. Name resolution is a fullmatch, so the lookahead excludes exactly "Trunk".
    cfg.events["limb_inertia"].params["asset_cfg"].body_names = (r"(?!Trunk$).*",)

    # Dedicated foot-position tracking reward. The global motion_body_pos term
    # averages squared error across all tracked bodies, which dilutes the
    # foot signal to the point that a planted policy scores ~0.87 even while
    # the reference is mid-stride. A foot-only term with a tight std restores
    # the gradient on the legs.
    cfg.rewards["motion_foot_pos"] = RewardTermCfg(
        func=mdp.motion_relative_body_position_error_exp,
        weight=2.0,
        params={
            "command_name": "motion",
            "std": 0.1,
            "body_names": ("left_foot_link", "right_foot_link"),
        },
    )

    cfg.terminations["ee_body_pos"].params["body_names"] = ee_body_names

    cfg.viewer.body_name = "Trunk"

    # Apply play mode overrides.
    if play:
        # Effectively infinite episode length.
        cfg.episode_length_s = int(1e9)

        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)

        # Disable RSI randomization.
        motion_cmd.pose_range = {}
        motion_cmd.velocity_range = {}

        motion_cmd.sampling_mode = "start"

        # Disable resets: keep only the (effectively-never-firing) time_out term.
        # cfg.terminations = {"time_out": cfg.terminations["time_out"]}

    return cfg

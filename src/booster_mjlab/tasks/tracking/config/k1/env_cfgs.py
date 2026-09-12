"""Booster K1 flat tracking environment configurations."""

import dataclasses as _dc
from copy import deepcopy

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

from booster_mjlab.robots.booster_k1.sensors import (
    FootClearanceSensorCfg,
    FootSoleGridPatternCfg,
)

from booster_mjlab.robots import (
    get_k1_robot_cfg,
)
from booster_mjlab.robots.booster_k1.k1_constants import (
    K1_ACTUATOR_R14,
    _build_action_scale,
)
from booster_mjlab.robots.booster_k1.torque_speed import (
    BoosterEntityCfg,
    BoosterPositionActuatorCfg,
    initialize_torque_speed_limits,
)
from booster_mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

_TRACKING_ARMS = _dc.replace(K1_ACTUATOR_R14, stiffness=20.0, damping=2.0)
_TRACKING_ARM_ACTION_SCALE = {
    expr: 0.5 * _TRACKING_ARMS.effort_limit / _TRACKING_ARMS.stiffness
    for expr in _TRACKING_ARMS.target_names_expr
}


def booster_k1_flat_tracking_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Booster K1 flat terrain tracking configuration."""
    cfg = make_tracking_env_cfg()

    base_robot_cfg = get_k1_robot_cfg()
    robot_cfg = BoosterEntityCfg(
        **{f.name: getattr(base_robot_cfg, f.name) for f in _dc.fields(base_robot_cfg)}
    )
    robot_cfg.articulation = deepcopy(robot_cfg.articulation)
    assert robot_cfg.articulation is not None
    actuators = []
    for actuator in robot_cfg.articulation.actuators:
        if actuator == K1_ACTUATOR_R14:
            actuator = _TRACKING_ARMS
        actuators.append(
            BoosterPositionActuatorCfg(
                **{f.name: getattr(actuator, f.name) for f in _dc.fields(actuator)}
            )
        )
    robot_cfg.articulation.actuators = tuple(actuators)
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
    cfg.scene.sensors = (self_collision_cfg, foot_height_scan)
    cfg.observations["critic"].terms["foot_height"] = ObservationTermCfg(
        func=foot_height,
        params={"sensor_name": "foot_height_scan"},
    )

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = _build_action_scale(robot_cfg.articulation)
    joint_pos_action.scale.update(_TRACKING_ARM_ACTION_SCALE)

    cfg.events["initialize_torque_speed_limits"] = EventTermCfg(
        mode="startup", func=initialize_torque_speed_limits
    )

    assert cfg.commands is not None
    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.anchor_body_name = "Trunk"
    motion_cmd.body_names = (
        "Trunk",
        "Head_1",
        "Head_2",
        "Left_Arm_1",
        "Left_Arm_2",
        "Left_Arm_3",
        "left_hand_link",
        "left_hand_end_ball",
        "Right_Arm_1",
        "Right_Arm_2",
        "Right_Arm_3",
        "right_hand_link",
        "right_hand_end_ball",
        "Left_Hip_Pitch",
        "Left_Hip_Roll",
        "Left_Hip_Yaw",
        "Left_Shank",
        "Left_Ankle_Cross",
        "left_foot_link",
        "Right_Hip_Pitch",
        "Right_Hip_Roll",
        "Right_Hip_Yaw",
        "Right_Shank",
        "Right_Ankle_Cross",
        "right_foot_link",
    )

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
    # averages squared error across all 25 tracked bodies, which dilutes the
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

    cfg.terminations["ee_body_pos"].params["body_names"] = (
        "left_foot_link",
        "right_foot_link",
        "left_hand_end_ball",
        "right_hand_end_ball",
    )

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

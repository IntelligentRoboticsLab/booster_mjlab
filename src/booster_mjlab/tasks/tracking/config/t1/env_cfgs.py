"""Booster T1 flat tracking environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg

from booster_mjlab.robots import T1_ACTION_SCALE, get_t1_robot_cfg
from booster_mjlab.robots.booster_t1.t1_constants import T1_FOOT_SOLE_SIZE
from booster_mjlab.robots.torque_speed import with_torque_speed_limits
from booster_mjlab.tasks.tracking.config.env_cfgs import booster_flat_tracking_env_cfg

# All T1 bodies, in MJCF order (= the link_body_list of the T1 motion datasets).
T1_TRACKING_BODY_NAMES = (
    "Trunk",
    "H1",
    "H2",
    "AL1",
    "AL2",
    "AL3",
    "left_hand_link",
    "AR1",
    "AR2",
    "AR3",
    "right_hand_link",
    "Waist",
    "Hip_Pitch_Left",
    "Hip_Roll_Left",
    "Hip_Yaw_Left",
    "Shank_Left",
    "Ankle_Cross_Left",
    "left_foot_link",
    "Hip_Pitch_Right",
    "Hip_Roll_Right",
    "Hip_Yaw_Right",
    "Shank_Right",
    "Ankle_Cross_Right",
    "right_foot_link",
)


def booster_t1_flat_tracking_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Booster T1 flat terrain tracking configuration."""
    return booster_flat_tracking_env_cfg(
        with_torque_speed_limits(get_t1_robot_cfg()),
        dict(T1_ACTION_SCALE),
        T1_FOOT_SOLE_SIZE,
        body_names=T1_TRACKING_BODY_NAMES,
        # The T1 has no hand end-effector bodies; the forearm links end at the hands.
        ee_body_names=(
            "left_foot_link",
            "right_foot_link",
            "left_hand_link",
            "right_hand_link",
        ),
        play=play,
    )

"""Booster T1 velocity tracking environment configurations.

The T1 shares the K1's body, site and collision geom names, so these configs
start from the K1 ones and swap in the robot and its size-dependent settings.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from booster_mjlab.robots import T1_ACTION_SCALE, get_t1_robot_cfg
from booster_mjlab.robots.booster_k1.sensors import (
    FootClearanceSensorCfg,
    FootSoleGridPatternCfg,
)
from booster_mjlab.tasks.velocity.config.k1.env_cfgs import (
    booster_k1_flat_env_cfg,
    booster_k1_rough_env_cfg,
)
from booster_mjlab.tasks.velocity.mdp.rewards import (
    track_body_angular_velocity_smoothed,
)
from booster_mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg
from booster_mjlab.tasks.velocity.velocity_env_cfg import (
    use_tracking_velocity_curriculum,
)

# Inner sole contact area. The T1 foot is 0.225 x 0.10 m; this keeps the K1
# grid's margin to the foot edge.
T1_FOOT_SOLE_SIZE = (0.16, 0.075)

# Swing foot target. The T1 reference motions swing ~1.3x higher than the K1's
# (median apex 12.6 vs 9.5 cm), so the K1's 6 cm would reward shuffling.
T1_TARGET_FOOT_HEIGHT = 0.08

# Linear-velocity tracking score (Episode_Reward/track_linear_velocity over its
# 2.25 weight) at which the command curriculum moves on. Runs 9dzo12dm and
# u5golm2n reached ~0.62 on the first stage by iteration ~3k and plateaued at
# ~0.65 by 4k while waiting for the fixed 5k step; later stages held 0.62-0.66.
T1_TRACKING_CURRICULUM_THRESHOLD = 0.62

# Low-pass time constant (s) on the pelvis yaw rate the yaw command is tracked with. On the
# T1 AMASS clips, 0.3 s cuts the in-stride pelvis yaw-rate swing from 0.63 to 0.26 rad/s rms
# at walking speeds (1.29 to 0.49 running), below the trunk's unfiltered 0.51.
T1_YAW_RATE_TIME_CONSTANT = 0.3

# Joints the AMP discriminator scores: the legs, the arms and the waist, so the arm
# swing and waist counter-rotation of the reference motions drive the style reward.
# Unobserved, the waist twisted freely (run 264kmanj); the T1 AMASS clips hold it at
# ~0.19 rad std, counter-rotating against the trunk yaw. The head stays out.
T1_AMP_JOINT_NAMES = (
    r"Waist",
    r".*_Hip_.*",
    r".*_Knee_.*",
    r".*_Ankle_.*",
    r".*_Shoulder_.*",
    r".*_Elbow_.*",
)

# T1 head joints are AAHead_yaw and Head_pitch, so match the head prefix anywhere.
_T1_HEAD = r".*Head_.*"


def make_t1_cfg(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
    """Convert a K1 velocity config to the T1."""
    cfg.scene.entities = {"robot": get_t1_robot_cfg()}

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = T1_ACTION_SCALE

    for sensor in cfg.scene.sensors:
        if isinstance(sensor, FootClearanceSensorCfg):
            sensor.pattern = FootSoleGridPatternCfg(size=T1_FOOT_SOLE_SIZE)

    assert cfg.commands is not None
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    # Keep the command arrows above the head of the taller robot.
    twist_cmd.viz.z_offset = 1.45

    # The trunk sits above the yaw-only waist, so tracking yaw on it lets a waist twist stand in
    # for turning (e.g. on turn-in-place commands). Track on the pelvis so the legs must turn,
    # low-passed so the pelvis's in-stride yaw swing is not penalized.
    track_yaw = cfg.rewards["track_angular_velocity"]
    track_yaw.func = track_body_angular_velocity_smoothed
    track_yaw.params["asset_cfg"] = SceneEntityCfg("robot", body_names=("Waist",))
    track_yaw.params["yaw_time_constant"] = T1_YAW_RATE_TIME_CONSTANT
    # Steer heading commands by the pelvis and report the yaw error on it, not on the root
    # (Trunk) that the waist swings.
    twist_cmd.yaw_body_name = "Waist"
    twist_cmd.yaw_rate_time_constant = T1_YAW_RATE_TIME_CONSTANT

    for reward_name in ("foot_clearance", "foot_swing_height"):
        cfg.rewards[reward_name].params["target_height"] = T1_TARGET_FOOT_HEIGHT
    # Runs 9dzo12dm and u5golm2n shuffled (0.04 m peak foot height, 0.15 s air time) with
    # air_time earning under half of its 0.1 weight.
    cfg.rewards["air_time"].weight = 0.3

    # The posture penalty only keeps the head still: the arms follow the style
    # reward (T1_AMP_JOINT_NAMES) instead of being pulled to the default pose.
    posture = cfg.rewards["upper_body_posture"].params
    posture["asset_cfg"].joint_names = (_T1_HEAD,)
    for regime in ("std_standing", "std_walking", "std_running"):
        posture[regime] = {_T1_HEAD: posture[regime][r"Head_.*"]}

    return use_tracking_velocity_curriculum(cfg, T1_TRACKING_CURRICULUM_THRESHOLD)


def booster_t1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Booster T1 rough terrain velocity tracking configuration."""
    return make_t1_cfg(booster_k1_rough_env_cfg(play=play))


def booster_t1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Booster T1 flat terrain velocity tracking configuration."""
    return make_t1_cfg(booster_k1_flat_env_cfg(play=play))


T1_ACTOR_HISTORY_LEN = 10
"""Actor frames stacked by the history tasks (0.2 s at 50 Hz), as on the K1. With 25 frames, Muon's
update on the ~1950-wide first layer kept the KL above target at the 1e-5 learning-rate floor
(runs 9dzo12dm, u5golm2n, 7p5wdvds, 264kmanj)."""


def with_actor_history(
    cfg: ManagerBasedRlEnvCfg, history_len: int = T1_ACTOR_HISTORY_LEN
) -> ManagerBasedRlEnvCfg:
    """Give the actor the last ``history_len`` frames, flattened by mjlab per term."""
    actor = cfg.observations["actor"]
    actor.history_length = history_len
    actor.flatten_history_dim = True
    return cfg


def booster_t1_flat_history_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Flat T1 velocity task whose actor sees the last ``T1_ACTOR_HISTORY_LEN`` observation frames."""
    return with_actor_history(booster_t1_flat_env_cfg(play=play))

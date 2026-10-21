"""Booster K1 standing with torso and head commands (Disney BDX's "perpetual standing" policy).

The operator commands the torso's height, roll, pitch and yaw and the head's yaw and pitch; the policy
follows a reference pose that keeps both feet planted (k1_stand_library.npz, see stand_library.py)
and balances while doing it. Ranges: crouch 13.5 cm / rise 3 cm, roll +-17 deg, lean 54 deg forward /
48 deg back, twist +-41 deg (single-axis maxima; combinations the legs can't reach are scaled back),
head over its joint range.

Built on the K1 flat velocity environment: same robot, actuators (incl. delays), sensors, domain
randomization, pushes and contact terms; commands, observations, rewards, terminations and the
curriculum are the standing task's own.
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from booster_mjlab.mdp.terminations import stochastic_bad_orientation
from booster_mjlab.tasks.standing import mdp
from booster_mjlab.tasks.standing.commands import StandCommandCfg
from booster_mjlab.tasks.velocity.config.k1.env_cfgs import booster_k1_flat_env_cfg

CMD = "stand"
LEGS = r".*_(Hip|Knee|Ankle)_.*"
HEAD = r"Head_.*"
ARMS = r".*_(Shoulder|Elbow)_.*"
RANGE_RAMP_ITERATIONS = 2000
STEPS_PER_ITERATION = 24


def booster_k1_standing_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = booster_k1_flat_env_cfg(play=play)
    m = {"command_name": CMD}

    # no terrain scan needed for standing
    cfg.scene.sensors = tuple(
        s for s in cfg.scene.sensors if s.name != "foot_height_scan"
    )

    cfg.commands = {CMD: StandCommandCfg(debug_vis=False)}

    # --- observations: the velocity task's actor terms with the stand command ---
    actor = {
        "base_ang_vel": ObservationTermCfg(
            func=vel_mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
            noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        "projected_gravity": ObservationTermCfg(
            func=vel_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
        ),
        "joint_pos": ObservationTermCfg(
            func=vel_mdp.joint_pos_rel,
            params={"biased": True},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=vel_mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5)
        ),
        "actions": ObservationTermCfg(func=vel_mdp.last_action),
        "command": ObservationTermCfg(func=vel_mdp.generated_commands, params=m),
    }
    critic = {
        **actor,
        "joint_pos": ObservationTermCfg(func=vel_mdp.joint_pos_rel),
        "joint_vel": ObservationTermCfg(func=vel_mdp.joint_vel_rel),
        "base_lin_vel": ObservationTermCfg(
            func=vel_mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"}
        ),
        "ref_joint_pos": ObservationTermCfg(func=mdp.ref_joint_pos_rel, params=m),
        "torso_error": ObservationTermCfg(func=mdp.torso_tracking_error, params=m),
        "foot_contact": ObservationTermCfg(
            func=vel_mdp.foot_contact, params={"sensor_name": "feet_ground_contact"}
        ),
    }
    cfg.observations = {
        "actor": ObservationGroupCfg(
            terms=actor, concatenate_terms=True, enable_corruption=not play
        ),
        "critic": ObservationGroupCfg(
            terms=critic, concatenate_terms=True, enable_corruption=False
        ),
    }

    # --- rewards: follow the commanded torso and the reference pose, keep the feet still ---
    cfg.rewards = {
        "torso_height": RewardTermCfg(
            func=mdp.torso_height, weight=1.0, params={**m, "std": 0.02}
        ),
        "torso_orientation": RewardTermCfg(
            func=mdp.torso_orientation, weight=1.5, params={**m, "std": 0.15}
        ),
        "leg_pose": RewardTermCfg(
            func=mdp.joint_pos_tracking,
            weight=1.5,
            params={**m, "joint_names": LEGS, "std": 0.15},
        ),
        "head_pose": RewardTermCfg(
            func=mdp.joint_pos_tracking,
            weight=0.5,
            params={**m, "joint_names": HEAD, "std": 0.1},
        ),
        "arm_pose": RewardTermCfg(
            func=mdp.joint_pos_tracking,
            weight=0.3,
            params={**m, "joint_names": ARMS, "std": 0.2},
        ),
        "both_feet_contact": RewardTermCfg(
            func=mdp.both_feet_contact,
            weight=0.5,
            params={"sensor_name": "feet_ground_contact"},
        ),
        "feet_planted": RewardTermCfg(func=mdp.feet_planted, weight=-1.0, params=m),
        "angular_momentum": RewardTermCfg(
            func=vel_mdp.angular_momentum_penalty,
            weight=-0.005,
            params={"sensor_name": "robot/root_angmom"},
        ),
        "dof_pos_limits": RewardTermCfg(func=vel_mdp.joint_pos_limits, weight=-1.0),
        "action_rate_l2": RewardTermCfg(func=vel_mdp.action_rate_l2, weight=-0.1),
        "self_collisions": RewardTermCfg(
            func=vel_mdp.self_collision_cost,
            weight=-1.0,
            params={"sensor_name": "self_collision"},
        ),
        "termination": RewardTermCfg(func=envs_mdp.is_terminated, weight=-50.0),
    }

    # --- terminations: the commanded 54 deg bow is not a fall ---
    cfg.terminations = {
        "time_out": TerminationTermCfg(func=vel_mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(
            func=stochastic_bad_orientation,
            params={"limit_angle": math.radians(75.0), "probability": 0.02},
        ),
        "too_low": TerminationTermCfg(
            func=envs_mdp.root_height_below_minimum, params={"minimum_height": 0.25}
        ),
        "illegal_contact": TerminationTermCfg(
            func=vel_mdp.illegal_contact,
            params={"sensor_name": "non_foot_ground_contact"},
        ),
    }

    # --- events: start standing (the velocity task's reset throws the robot around) ---
    cfg.events["reset_base"].params["pose_range"] = {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "yaw": (-math.pi, math.pi),
    }
    cfg.events["reset_base"].params["velocity_range"] = {
        "x": (-0.1, 0.1),
        "y": (-0.1, 0.1),
        "roll": (-0.1, 0.1),
        "pitch": (-0.1, 0.1),
        "yaw": (-0.2, 0.2),
    }
    cfg.events["reset_robot_joints"].params["position_range"] = (-0.05, 0.05)

    # --- curriculum: the command ranges grow from 30% to full ---
    cfg.curriculum = {}
    if not play:
        cfg.curriculum["stand_range"] = CurriculumTermCfg(
            func=mdp.stand_range_ramp,
            params={
                **m,
                "start": 0.3,
                "ramp_steps": RANGE_RAMP_ITERATIONS * STEPS_PER_ITERATION,
            },
        )
    cfg.metrics = {}
    if play:
        cfg.terminations.pop("illegal_contact", None)
    cfg.episode_length_s = 20.0 if not play else int(1e9)
    return cfg

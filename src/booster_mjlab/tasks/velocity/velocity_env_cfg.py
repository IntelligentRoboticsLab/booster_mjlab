"""Velocity task configuration."""

from booster_mjlab.mdp.terminations import stochastic_bad_orientation

import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import TerrainHeightSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp

from booster_mjlab.mdp.terrain import randomize_terrain_contact
from booster_mjlab.tasks.velocity.mdp.rewards import (
    standing_pose_deviation_l1,
    track_linear_velocity as booster_mjlab_track_linear_velocity,
    upper_body_posture_penalty,
    variable_upright,
)
from booster_mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg
from booster_mjlab.terrains.contact import terrain_collisions
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import flat, hf_pyramid_slope, perlin_noise, random_rough
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

VELOCITY_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    # Ten rows produce ten curriculum levels from flat to 10 cm of relief.
    num_rows=10,
    num_cols=4,
    sub_terrains={
        "flat": flat(proportion=0.25),
        "heightfield": hf_pyramid_slope(
            proportion=0.25,
            # With an 8 m patch and 2 m center platform, this produces a
            # 10 cm maximum elevation difference at difficulty 1.0.
            slope_range=(0.0, 0.048),
            platform_width=2.0,
            border_width=0.25,
        ),
        "random_noise": random_rough(
            proportion=0.25,
            noise_range=(-0.05, 0.05),
            noise_step=0.01,
            border_width=0.25,
            scale_with_difficulty=True,
        ),
        "perlin_noise": perlin_noise(
            proportion=0.25,
            height_range=(0.0, 0.10),
            # Match the other heightfields' 10 cm grid. The preset's 5 cm
            # grid creates 160x160 cells and can exceed MuJoCo's per-hfield
            # collision candidate limit under a foot. Doubling
            # horizontal_scale preserves the Perlin feature size.
            resolution=0.1,
            horizontal_scale=0.2,
        ),
    },
    add_lights=True,
)


def make_velocity_env_cfg() -> ManagerBasedRlEnvCfg:
    """Create base velocity tracking task configuration."""

    ##
    # Observations
    ##

    actor_terms = {
        "base_ang_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
            noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={"biased": True},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
        "command": ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "twist"},
        ),
    }

    critic_terms = {
        **actor_terms,
        # The critic gets ground-truth joint state without encoder bias or noise.
        "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
        "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
        "base_lin_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_vel"},
        ),
        "foot_height": ObservationTermCfg(
            func=mdp.foot_height,
            params={"sensor_name": "foot_height_scan"},
        ),
        "foot_air_time": ObservationTermCfg(
            func=mdp.foot_air_time,
            params={"sensor_name": "feet_ground_contact"},
        ),
        "foot_contact": ObservationTermCfg(
            func=mdp.foot_contact,
            params={"sensor_name": "feet_ground_contact"},
        ),
        "foot_contact_forces": ObservationTermCfg(
            func=mdp.foot_contact_forces,
            params={"sensor_name": "feet_ground_contact"},
        ),
    }

    observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=True,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    ##
    # Actions
    ##

    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.5,  # Override per robot
            use_default_offset=True,
        )
    }

    ##
    # Commands
    ##

    commands: dict[str, CommandTermCfg] = {
        "twist": UniformVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(1.0, 4.0),
            rel_standing_envs=0.1,
            rel_heading_envs=0.3,
            rel_forward_envs=0.1,
            heading_command=True,
            heading_control_stiffness=0.5,
            debug_vis=True,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.5, 0.5),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-0.5, 0.5),
                heading=(-math.pi, math.pi),
            ),
        )
    }

    ##
    # Events
    ##

    events = {
        "terrain_contact": EventTermCfg(
            func=randomize_terrain_contact,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("terrain"),
                "solref_ranges": {0: (0.006, 0.03), 1: (0.95, 1.05)},
                "solimp_ranges": {0: (0.88, 0.92), 1: (0.94, 0.99), 2: (0.003, 0.01)},
                "shared_random": True,
            },
        ),
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (0.01, 0.05),
                    "yaw": (-3.14, 3.14),
                },
                "velocity_range": {
                    "x": (-1.0, 1.0),
                    "y": (-1.0, 1.0),
                    "z": (0.01, 0.3),
                    "roll": (-0.1, 0.1),
                    "pitch": (-0.1, 0.1),
                    "yaw": (-0.5, 0.5),
                },
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.1, 0.1),
                "velocity_range": (-0.1, 0.1),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "push_robot": EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(1.5, 4.0),
            params={
                "velocity_range": {
                    "x": (-0.28, 0.28),
                    "y": (-0.28, 0.28),
                    "z": (-0.2, 0.2),
                    "roll": (-0.52, 0.52),
                    "pitch": (-0.52, 0.52),
                    "yaw": (-0.78, 0.78),
                },
            },
        ),
        "foot_friction": EventTermCfg(
            mode="startup",
            func=mdp.dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=()),
                "operation": "abs",
                "ranges": (0.75, 1.25),
                "shared_random": True,  # Both feet share the same friction.
            },
        ),
        "encoder_bias": EventTermCfg(
            mode="startup",
            func=mdp.dr.encoder_bias,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "bias_range": (-0.015, 0.015),
            },
        ),
        "pd_gains": EventTermCfg(
            mode="startup",
            func=mdp.dr.pd_gains,
            params={
                "asset_cfg": SceneEntityCfg("robot", actuator_names=".*"),
                "operation": "scale",
                "kp_range": (0.8, 1.2),
                "kd_range": (0.8, 1.2),
            },
        ),
        # The trunk carries the payload, so its mass and COM move much further than the limbs'.
        "trunk_inertia": EventTermCfg(
            mode="startup",
            func=mdp.dr.pseudo_inertia,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=()),
                "alpha_range": (-0.05, 0.05),
                "t_range": (-0.05, 0.05),
            },
        ),
        "limb_inertia": EventTermCfg(
            mode="startup",
            func=mdp.dr.pseudo_inertia,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=()),
                "alpha_range": (-0.05, 0.05),
                "t_range": (-0.025, 0.025),
            },
        ),
    }

    ##
    # Rewards
    ##

    # shared target foot height for multiple reward terms
    target_foot_height = 0.06

    rewards = {
        "track_linear_velocity": RewardTermCfg(
            func=booster_mjlab_track_linear_velocity,
            # 2.25 rather than the angular term's 2.0: the speed-relative tolerance is
            # stricter than the angular term's fixed std at slow commands, and this
            # matches the expected reward at a typical 0.2 m/s error to the previous
            # fixed-std formulation so the two tracking terms stay in balance.
            weight=2.25,
            params={
                "command_name": "twist",
                # Tolerance grows with commanded speed, from std_at_rest at rest to std
                # at |cmd| >= 0.67 m/s. Standing still is >= 1.33 sigma from target.
                "std": math.sqrt(0.25),
                "std_at_rest": math.sqrt(0.01),
                "relative_std": 0.75,
                # Blend in a linear progress term so a standing policy still gets a
                # gradient toward moving at all commanded speeds.
                "progress_weight": 0.5,
            },
        ),
        "track_angular_velocity": RewardTermCfg(
            func=mdp.track_angular_velocity,
            weight=2.0,
            params={"command_name": "twist", "std": math.sqrt(0.5)},
        ),
        "upright": RewardTermCfg(
            func=variable_upright,
            weight=1.0,
            params={
                "command_name": "twist",
                # Tilt tolerance widens with commanded speed: allow a little more lean when walking and more when
                # running. std_standing matches the previous fixed std.
                "std_standing": math.sqrt(0.20),
                "std_walking": math.sqrt(0.25),
                "std_running": math.sqrt(0.35),
                "walking_threshold": 0.05,
                "running_threshold": 1.5,
                "asset_cfg": SceneEntityCfg("robot", body_names=()),
            },
        ),
        "body_ang_vel": RewardTermCfg(
            func=mdp.body_angular_velocity_penalty,
            weight=-0.01,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=())},
        ),
        "angular_momentum": RewardTermCfg(
            func=mdp.angular_momentum_penalty,
            weight=-0.005,
            params={"sensor_name": "robot/root_angmom"},
        ),
        "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
        "air_time": RewardTermCfg(
            func=mdp.feet_air_time,
            weight=0.1,
            params={
                "sensor_name": "feet_ground_contact",
                "threshold_min": 0.05,
                "threshold_max": 0.5,
                "command_name": "twist",
                "command_threshold": 0.2,
            },
        ),
        "foot_clearance": RewardTermCfg(
            func=mdp.feet_clearance,
            weight=-2.0,
            params={
                "target_height": target_foot_height,
                "height_sensor_name": "foot_height_scan",
                "command_name": "twist",
                "command_threshold": 0.05,
                "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
            },
        ),
        "foot_swing_height": RewardTermCfg(
            func=mdp.feet_swing_height,
            weight=-0.25,
            params={
                "sensor_name": "feet_ground_contact",
                "height_sensor_name": "foot_height_scan",
                "target_height": target_foot_height,
                "command_name": "twist",
                "command_threshold": 0.05,
            },
        ),
        "foot_slip": RewardTermCfg(
            func=mdp.feet_slip,
            weight=-0.2,
            params={
                "sensor_name": "feet_ground_contact",
                "command_name": "twist",
                "command_threshold": 0.05,
                "asset_cfg": SceneEntityCfg("robot", site_names=()),
            },
        ),
        "soft_landing": RewardTermCfg(
            func=mdp.soft_landing,
            weight=-0.001,
            params={
                "sensor_name": "feet_ground_contact",
                "command_name": "twist",
                "command_threshold": 0.05,
            },
        ),
        "self_collisions": RewardTermCfg(
            func=mdp.self_collision_cost,
            weight=-1.0,
            params={"sensor_name": "self_collision"},
        ),
        "upper_body_posture": RewardTermCfg(
            func=upper_body_posture_penalty,
            weight=-0.1,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=(r"Head_.*", r".*_Shoulder_.*", r".*_Elbow_.*"),
                ),
                "command_name": "twist",
                "std_standing": {
                    r"Head_.*": 0.05,
                    r".*_Shoulder_.*": 0.05,
                    r".*_Elbow_.*": 0.05,
                },
                "std_walking": {
                    r"Head_.*": 0.05,
                    r".*_Shoulder_.*": 0.15,
                    r".*_Elbow_.*": 0.15,
                },
                "std_running": {
                    r"Head_.*": 0.05,
                    r".*_Shoulder_Pitch": 0.5,
                    r".*_Shoulder_Roll": 0.2,
                    r".*_Elbow_.*": 0.35,
                },
                "walking_threshold": 0.05,
                "running_threshold": 1.0,
            },
        ),
        "standing_pose_l1": RewardTermCfg(
            func=standing_pose_deviation_l1,
            weight=-0.5,
            params={
                "command_name": "twist",
                "command_threshold": 0.05,
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
    }

    metrics: dict[str, MetricsTermCfg] = {}

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(
            func=stochastic_bad_orientation,
            params={"limit_angle": math.radians(63.0), "probability": 0.02},
        ),
        "illegal_contact": TerminationTermCfg(
            func=mdp.illegal_contact,
            params={"sensor_name": "non_foot_ground_contact"},
        ),
    }

    # mjlab restores common_step_counter on resume; keep absolute stage steps.
    curriculum = {
        "terrain_levels": CurriculumTermCfg(
            func=mdp.terrain_levels_vel,
            params={"command_name": "twist"},
        ),
        "soft_landing_weight": CurriculumTermCfg(
            func=mdp.reward_curriculum,
            params={
                "reward_name": "soft_landing",
                "stages": [
                    {"step": 0, "weight": -0.0001},
                    {"step": 1000 * 24, "weight": -0.001},
                    {"step": 7000 * 24, "weight": -0.005},
                ],
            },
        ),
        "command_vel": CurriculumTermCfg(
            func=mdp.commands_vel,
            params={
                "command_name": "twist",
                "velocity_stages": [
                    {
                        "step": 0,
                        "lin_vel_x": (-1.0, 1.2),
                        "lin_vel_y": (-1.0, 1.0),
                        "ang_vel_z": (-1.0, 1.0),
                    },
                    {
                        "step": 5000 * 24,
                        "lin_vel_x": (-1.0, 1.5),
                        "lin_vel_y": (-1.25, 1.25),
                        "ang_vel_z": (-1.25, 1.25),
                    },
                    {
                        "step": 10000 * 24,
                        "lin_vel_x": (-1.25, 1.5),
                        "lin_vel_y": (-1.5, 1.5),
                        "ang_vel_z": (-1.5, 1.5),
                    },
                    {
                        "step": 15000 * 24,
                        "lin_vel_x": (-1.5, 1.75),
                        "lin_vel_y": (-1.75, 1.75),
                        "ang_vel_z": (-1.5, 1.5),
                    },
                ],
            },
        ),
    }

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(
                collisions=terrain_collisions(),
                terrain_type="generator",
                terrain_generator=replace(VELOCITY_ROUGH_TERRAINS_CFG),
                max_init_terrain_level=2,
            ),
            sensors=(
                TerrainHeightSensorCfg(
                    name="foot_height_scan",
                    frame=(),  # Set per-robot.
                    ray_alignment="yaw",
                    max_distance=1.0,
                    exclude_parent_body=True,
                    include_geom_groups=(0,),
                ),
            ),
            num_envs=1,
            extent=2.0,
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum=curriculum,
        metrics=metrics,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="",
            distance=3.0,
            elevation=-5.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            nconmax=80,
            njmax=1500,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
            ),
        ),
        decimation=4,
        episode_length_s=20.0,
    )

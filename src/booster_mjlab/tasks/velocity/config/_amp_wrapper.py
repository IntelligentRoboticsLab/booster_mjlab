from mjlab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
)
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from booster_mjlab.amp.config import DEFAULT_AMP_DATASET
from booster_mjlab.amp.curriculums import anneal_style_reward
from booster_mjlab.mdp.events import reset_from_pose_pool


def with_amp_obs_group(
    cfg,
    fast_sac: bool = False,
    dataset_root: str = DEFAULT_AMP_DATASET,
    speed_factor: float = 1.0,
    dataset_weights: list[float] | None = None,
    augmentations: list[dict[str, object]] | None = None,
    dataset_transform: str | None = None,
    include_base_lin_vel: bool = False,
):
    """Wraps the environment configuration for AMP tasks."""
    if cfg.curriculum is None:
        cfg.curriculum = {}
    cfg.curriculum["amp_style_weight"] = CurriculumTermCfg(
        func=anneal_style_reward,
        params={
            "start_weight": 0.3,
            "end_weight": 0.3,
            "start_step": 0,
            "end_step": 3000 * 24 if not fast_sac else 10000,
        },
    )

    # Discriminator features cover the legs only; the arms and head are excluded
    # so their style from the dataset does not drive the style reward.
    amp_joint_names = (
        r".*_Hip_.*",
        r".*_Knee_.*",
        r".*_Ankle_.*",
    )

    # Term order must match the expert layout in the motion loader:
    # [joint_pos, joint_vel, (base_lin_vel), (base_ang_vel), (projected_gravity)]. The loader selects each optional
    # block by term name, so renaming a term here silently drops it from the expert side.
    amp_terms = {
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=amp_joint_names)},
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=amp_joint_names)},
        ),
    }
    if include_base_lin_vel:
        amp_terms["base_lin_vel"] = ObservationTermCfg(func=mdp.base_lin_vel)

    amp_terms["projected_gravity"] = ObservationTermCfg(func=mdp.projected_gravity)

    cfg.observations["amp"] = ObservationGroupCfg(
        terms=amp_terms,
        concatenate_terms=True,
        enable_corruption=False,
    )

    # Replace default reset events with AMP motion data reset.
    cfg.events.pop("reset_base", None)
    cfg.events.pop("reset_robot_joints", None)
    cfg.events["reset_robot_from_motion"] = EventTermCfg(
        func=reset_from_pose_pool,
        mode="reset",
        params={
            "self_collision_sensor": "self_collision",
            "nonfoot_ground_sensor": "non_foot_ground_contact",
            "num_iterations": 5,
            "dataset_root": dataset_root,
            "speed_factor": speed_factor,
            "dataset_weights": dataset_weights,
            "augmentations": augmentations,
            "dataset_transform": dataset_transform,
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.05),
                "yaw": (-3.14, 3.14),
            },
            "foot_sites": ("left_foot", "right_foot"),
            "min_foot_height": 0.0,
            "z_lift_factor": 0.08,
        },
    )

    return cfg

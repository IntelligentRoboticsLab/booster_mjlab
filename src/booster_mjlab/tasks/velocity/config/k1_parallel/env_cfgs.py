"""Booster K1 velocity configs for the parallel (closed-loop) ankle model."""

from booster_mjlab.robots.booster_k1.k1_parallel_constants import (
    K1_PARALLEL_ACTION_SCALE,
    K1_PARALLEL_ACTUATED_JOINTS,
    K1_PARALLEL_FREE_RESET_JOINTS,
    get_k1_parallel_robot_cfg,
)
from booster_mjlab.tasks.velocity.config.k1.env_cfgs import (
    booster_k1_flat_env_cfg,
    booster_k1_rough_env_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

# Joints the AMP discriminator scores. The passive ankle DOFs are kept (rather
# than the cranks) so the style features stay identical to the serial task.
AMP_JOINT_NAMES = (
    r".*_Hip_.*",
    r".*_Knee_.*",
    r".*_Ankle_Pitch",
    r".*_Ankle_Roll",
)


def actuated_joints_cfg(preserve_order: bool = True) -> SceneEntityCfg:
    """Scene entity selecting the 22 actuated joints in canonical order."""
    return SceneEntityCfg(
        "robot",
        joint_names=K1_PARALLEL_ACTUATED_JOINTS,
        preserve_order=preserve_order,
    )


def _restrict_joint_terms(cfg: ManagerBasedRlEnvCfg) -> None:
    """Point every joint-enumerating term at the actuated joints only."""
    for group in cfg.observations.values():
        for name in ("joint_pos", "joint_vel"):
            term = group.terms.get(name)
            if term is not None:
                term.params = {**term.params, "asset_cfg": actuated_joints_cfg()}

    # Randomizing the closed loop would start the ankles with a violated
    # constraint, so resets only perturb joints outside it.
    free_joints = SceneEntityCfg("robot", joint_names=K1_PARALLEL_FREE_RESET_JOINTS)
    reset_joints = cfg.events.get("reset_robot_joints")
    if reset_joints is not None and "asset_cfg" in reset_joints.params:
        reset_joints.params["asset_cfg"] = free_joints
    cfg.events["encoder_bias"].params["asset_cfg"] = actuated_joints_cfg()

    cfg.rewards["standing_pose_l1"].params["asset_cfg"] = actuated_joints_cfg()


def make_parallel_ankle_cfg(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
    """Convert a serial-ankle K1 velocity config to the parallel-ankle model."""
    cfg.scene.entities = {"robot": get_k1_parallel_robot_cfg()}

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = K1_PARALLEL_ACTION_SCALE

    _restrict_joint_terms(cfg)

    # Four `connect` equalities add 12 constraint rows per environment. Extra
    # solver iterations measurably do not reduce the residual (it is constraint
    # compliance, not convergence), so only the budget goes up.
    cfg.sim.njmax += 100

    return cfg


def with_parallel_amp_joints(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
    """Narrow the AMP observation group to the serial-equivalent ankle DOFs.

    The wrapper's ``.*_Ankle_.*`` pattern also matches the cranks and rods on this
    model, which would widen the discriminator features past the motion dataset.
    """
    amp_group = cfg.observations["amp"]
    for term in amp_group.terms.values():
        # Only joint terms carry an asset_cfg; projected_gravity has no joint selection to narrow.
        if "asset_cfg" in term.params:
            term.params["asset_cfg"] = SceneEntityCfg(
                "robot", joint_names=AMP_JOINT_NAMES
            )
    return cfg


def booster_k1_parallel_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Parallel-ankle K1 rough terrain velocity tracking configuration."""
    cfg = booster_k1_rough_env_cfg(play=play)
    return make_parallel_ankle_cfg(cfg)


def booster_k1_parallel_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Parallel-ankle K1 flat terrain velocity tracking configuration."""
    cfg = booster_k1_flat_env_cfg(play=play)
    return make_parallel_ankle_cfg(cfg)

"""Convert a Booster serial-ankle velocity config to the robot's closed-loop ankle model.

The parallel models keep the serial joint-observation and action width: two crank
drives take the ankle pitch/roll slots, and the passive ankle and rod joints are
left out of observations, rewards and resets.
"""

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg


def actuated_joints_cfg(
    actuated_joints: tuple[str, ...], preserve_order: bool = True
) -> SceneEntityCfg:
    """Scene entity selecting the actuated joints in canonical order."""
    return SceneEntityCfg(
        "robot",
        joint_names=actuated_joints,
        preserve_order=preserve_order,
    )


def _restrict_joint_terms(
    cfg: ManagerBasedRlEnvCfg,
    actuated_joints: tuple[str, ...],
    free_reset_joints: tuple[str, ...],
) -> None:
    """Point every joint-enumerating term at the actuated joints only."""
    for group in cfg.observations.values():
        for name in ("joint_pos", "joint_vel"):
            term = group.terms.get(name)
            if term is not None:
                term.params = {
                    **term.params,
                    "asset_cfg": actuated_joints_cfg(actuated_joints),
                }

    # Randomizing the closed loop would start the ankles with a violated
    # constraint, so resets only perturb joints outside it.
    free_joints = SceneEntityCfg("robot", joint_names=free_reset_joints)
    reset_joints = cfg.events.get("reset_robot_joints")
    if reset_joints is not None and "asset_cfg" in reset_joints.params:
        reset_joints.params["asset_cfg"] = free_joints
    cfg.events["encoder_bias"].params["asset_cfg"] = actuated_joints_cfg(
        actuated_joints
    )

    cfg.rewards["standing_pose_l1"].params["asset_cfg"] = actuated_joints_cfg(
        actuated_joints
    )


def convert_to_parallel_ankle(
    cfg: ManagerBasedRlEnvCfg,
    robot_cfg: EntityCfg,
    action_scale: dict[str, float],
    actuated_joints: tuple[str, ...],
    free_reset_joints: tuple[str, ...],
) -> ManagerBasedRlEnvCfg:
    """Swap a serial-ankle velocity config's robot for its closed-loop model."""
    cfg.scene.entities = {"robot": robot_cfg}

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = action_scale

    _restrict_joint_terms(cfg, actuated_joints, free_reset_joints)

    # Four `connect` equalities add 12 constraint rows per environment. Extra
    # solver iterations measurably do not reduce the residual (it is constraint
    # compliance, not convergence), so only the budget goes up.
    cfg.sim.njmax += 100

    return cfg


def with_parallel_amp_joints(
    cfg: ManagerBasedRlEnvCfg, amp_joint_names: tuple[str, ...]
) -> ManagerBasedRlEnvCfg:
    """Narrow the AMP observation group to ``amp_joint_names``, which should hold the
    serial-equivalent ankle DOFs.

    The wrapper's ``.*_Ankle_.*`` pattern also matches the cranks and rods on this
    model, which would widen the discriminator features past the motion dataset.
    """
    amp_group = cfg.observations["amp"]
    for term in amp_group.terms.values():
        # Only joint terms carry an asset_cfg; projected_gravity has no joint selection to narrow.
        if "asset_cfg" in term.params:
            term.params["asset_cfg"] = SceneEntityCfg(
                "robot", joint_names=amp_joint_names
            )
    return cfg

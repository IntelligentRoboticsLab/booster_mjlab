"""Terrain contact randomization."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mjlab.envs.mdp.dr._core import Ranges, _randomize_model_field
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@requires_model_fields("geom_solref", "geom_solimp")
def randomize_terrain_contact(
    env: "ManagerBasedRlEnv",
    env_ids,
    asset_cfg: SceneEntityCfg,
    solref_ranges: Ranges,
    solimp_ranges: Ranges,
    shared_random: bool = True,
) -> None:
    """Randomize short-turf-like contact compliance on terrain geoms.

    ``solref`` randomizes MuJoCo's contact time constant and damping ratio.
    ``solimp`` randomizes the contact impedance transition while preserving its
    midpoint and power parameters. Values are sampled once per environment at
    startup, so each environment has a consistent turf realization.
    """
    _randomize_model_field(
        env,
        env_ids,
        "geom_solref",
        entity_type="geom",
        ranges=solref_ranges,
        operation="abs",
        asset_cfg=asset_cfg,
        shared_random=shared_random,
        valid_axes=[0, 1],
    )
    _randomize_model_field(
        env,
        env_ids,
        "geom_solimp",
        entity_type="geom",
        ranges=solimp_ranges,
        operation="abs",
        asset_cfg=asset_cfg,
        shared_random=shared_random,
        valid_axes=[0, 1, 2, 3, 4],
    )

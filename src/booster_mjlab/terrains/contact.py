"""Shared terrain contact policy for Booster tasks."""

from mjlab.utils.spec_config import CollisionCfg


def terrain_collisions() -> tuple[CollisionCfg, ...]:
    """Zero friction preserves robot coefficients; solmix=1 weights terrain compliance."""
    return (
        CollisionCfg(
            geom_names_expr=(".*",),
            contype=1,
            conaffinity=1,
            condim=3,
            priority=0,
            friction=(0.0, 0.0, 0.0),
            solmix=1.0,
        ),
    )

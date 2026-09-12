"""Thin wrapper around mjlab's train script.

Adds:
- FastSAC runner config support (RslRlFastSacRunnerCfg union)
- Unpicklable object cleanup before config YAML serialization
"""

import sys
from dataclasses import dataclass, field
from typing import Literal

import tyro

from mjlab import TYRO_FLAGS
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scripts.train import launch_training
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

from booster_mjlab.rl import RslRlFastSacRunnerCfg, RslRlOnPolicyRunnerCfg
from booster_mjlab.scripts.utils import remove_unpicklable_objects


@dataclass(frozen=True)
class TrainConfig:
    env: ManagerBasedRlEnvCfg
    agent: RslRlOnPolicyRunnerCfg | RslRlFastSacRunnerCfg
    registry_name: str | None = None
    video: bool = False
    video_length: int = 200
    video_interval: int = 2000
    enable_nan_guard: bool = False
    log_root: str = "logs/rsl_rl"
    """Root directory under which experiment logs are written."""
    torchrunx_log_dir: str | None = None
    wandb_run_path: str | None = None
    wandb_checkpoint_name: str | None = None
    """Optional checkpoint name within the W&B run to load (e.g. 'model_4000.pt')."""
    gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

    @staticmethod
    def from_task(task_id: str) -> "TrainConfig":
        env_cfg = load_env_cfg(task_id)
        agent_cfg = load_rl_cfg(task_id)
        assert isinstance(agent_cfg, (RslRlOnPolicyRunnerCfg, RslRlFastSacRunnerCfg))
        return TrainConfig(env=env_cfg, agent=agent_cfg)


def _patch_dump_yaml() -> None:
    """Patch mjlab's train module to clean unpicklable objects before YAML serialization."""
    import mjlab.scripts.train as train_mod

    original_dump_yaml = train_mod.dump_yaml

    def safe_dump_yaml(filename, data, sort_keys=False):
        data = remove_unpicklable_objects(data, filename.stem)
        original_dump_yaml(filename, data, sort_keys)

    train_mod.dump_yaml = safe_dump_yaml


def main():
    # Import tasks to populate the registry.
    import mjlab.tasks  # noqa: F401

    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
        config=TYRO_FLAGS,
    )

    args = tyro.cli(
        TrainConfig,
        args=remaining_args,
        default=TrainConfig.from_task(chosen_task),
        prog=sys.argv[0] + f" {chosen_task}",
        config=TYRO_FLAGS,
    )
    del remaining_args

    _patch_dump_yaml()
    launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
    main()

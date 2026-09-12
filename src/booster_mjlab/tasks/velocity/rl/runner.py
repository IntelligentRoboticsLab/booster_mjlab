import os

import wandb
from rsl_rl.utils import WandbLogWriter

from booster_mjlab.amp.runners.amp_on_policy_runner import AmpOnPolicyRunner
from booster_mjlab.rl import RslRlVecEnvWrapper
from booster_mjlab.rl.exporter_utils import (
    attach_metadata_to_onnx,
    get_base_metadata,
)


class VelocityAmpOnPolicyRunner(AmpOnPolicyRunner):
    env: RslRlVecEnvWrapper

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if isinstance(self.logger.writer, WandbLogWriter):
            policy_dir = os.path.dirname(path)
            filename = os.path.basename(policy_dir) + ".onnx"
            self.export_policy_to_onnx(policy_dir, filename)
            run_name = wandb.run.name if wandb.run else "local"
            metadata = get_base_metadata(self.env.unwrapped, run_name)
            onnx_path = os.path.join(policy_dir, filename)
            attach_metadata_to_onnx(onnx_path, metadata)
            if self.cfg.get("upload_model", True):
                wandb.save(onnx_path, base_path=policy_dir)

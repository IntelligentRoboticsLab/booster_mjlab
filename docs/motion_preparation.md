# Preparing Motions for Tracking

booster_mjlab uses WandB to manage motion datasets for the `Mjlab-Tracking-*` tasks.
Clips can come from [GMR](https://github.com/YanjieZe/GMR/) (`.pkl`, the format produced when retargeting
[GVHMR](https://github.com/zju3dv/GVHMR) video reconstructions) or from
[BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking) (`.csv`).

Set your WandB entity first:

```bash
export WANDB_ENTITY=your-organization-name
```

## GMR motions

`upload-motion` converts a `.pkl` clip to the tracking format and uploads it to WandB in one step:

```bash
uv run upload-motion --input-file /path/to/motion.pkl --entity your-org
```

The motion is uploaded as an artifact named `<file-stem>-tracking` to the `motion_upload` project
(override with `--project` and `--artifact-name`). The command prints the artifact's full name, e.g.
`your-org/motion_upload/motion-tracking:v0`, which you pass as `--registry-name` when training.

## BeyondMimic motions

First create a registry collection in your WandB workspace named `motions`, then run:

```bash
MUJOCO_GL=egl uv run python -m booster_mjlab.scripts.csv_to_npz \
    --input-file /path/to/motion.csv \
    --output-name motion_name \
    --input-fps 30 \
    --output-fps 50 \
    --render
```

This uploads the motion to the `motions` registry as `your-org/motions/motion_name`, which you pass as
`--registry-name` when training.

## Next steps

See the [Motion Imitation](../README.md#2-motion-imitation) section of the README for training and evaluation
commands, and use `uv run visualize-motions` to inspect clips before training on them.

<img src="./assets/banner.webp" alt="Project banner">

# booster_mjlab

booster_mjlab is an [mjlab](https://github.com/mujocolab/mjlab) integration for the Booster K1.
It provides the K1 robot model, velocity and motion-tracking tasks, and an [Adversarial Motion Priors](https://arxiv.org/abs/2104.02180) (AMP) training pipeline that learns natural-looking gaits from motion capture data.

Real-robot and simulation demos are on the [project page](https://intelligentroboticslab.github.io/booster_mjlab/),
which also runs the velocity policy live in the browser (MuJoCo WebAssembly + the exported network).

## Getting Started

booster_mjlab requires an NVIDIA GPU for training. macOS is supported for evaluation only.

**Install from source:**

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/IntelligentRoboticsLab/booster-mjlab.git && cd booster-mjlab
uv run list_envs
```

The last command syncs the environment and prints every registered task.

## Training Examples

### 1. Velocity Tracking

Train a Booster K1 to follow velocity commands on flat terrain. The reward has been tuned to work together with AMP reference motions.
The default [motion dataset](https://huggingface.co/datasets/whirlwind-ams/lafan_locomotion_k1) contains a few locomotion clips retargeted from [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset).

```bash
uv run train Mjlab-Velocity-Flat-Amp-DA-Muon-Booster-K1 --env.scene.num-envs 4096
```

Evaluate a policy while training (fetches latest checkpoint from Weights & Biases):

```bash
uv run play Mjlab-Velocity-Flat-Amp-DA-Muon-Booster-K1 --wandb-run-path your-org/mjlab/run-id
```

To train on your own motions, pass a Hugging Face dataset (`namespace/repo`) or a local motion file or directory
with `--agent.dataset-root`.

The `-DA-` tasks train with left/right symmetry data augmentation. Drop `DA` from the task ID to train without it.

The `-Muon-` tasks use the [Muon optimizer](https://kellerjordan.github.io/posts/muon/) (Jordan et al., 2024) for the
actor and critic weight matrices. Drop `Muon` from the task ID for the Adam variant with otherwise identical settings.

### 2. Motion Imitation

Train a Booster K1 to track motion capture clips on flat terrain. Motions are managed as WandB artifacts;
see [Preparing Motions for Tracking](docs/motion_preparation.md) for converting and uploading GMR (`.pkl`)
or BeyondMimic (`.csv`) clips.

Once a motion is uploaded, train and evaluate with:

```bash
uv run train Mjlab-Tracking-Flat-Booster-K1 --registry-name your-org/motion_upload/motion-tracking:v0 --env.scene.num-envs 4096
uv run play Mjlab-Tracking-Flat-Booster-K1 --wandb-run-path your-org/mjlab/run-id
```

For a motion prepared with `csv_to_npz`, pass `--registry-name your-org/motions/motion_name` instead.

### 3. Browse Motion Data

Inspect, play back and edit motion clips in the browser before training on them:

```bash
uv run visualize-motions --help
```

Clips from the [lafan_locomotion_k1](https://huggingface.co/datasets/whirlwind-ams/lafan_locomotion_k1)
dataset, retargeted onto the parallel-ankle K1:

<picture>
    <source
    srcset="./assets/motion_grid_dark.webp"
    media="(prefers-color-scheme: dark)">
    <source
    srcset="./assets/motion_grid_light.webp"
    media="(prefers-color-scheme: light)">
    <img
    src="./assets/motion_grid_light.webp"
    alt="Retargeted lafan_locomotion_k1 clips playing on the K1"
    width="720">
</picture>

## Tasks

Velocity tasks come in `Flat` and `Rough` terrain variants, optionally with `-Amp-` (motion prior) and
`-DA-` (symmetry data augmentation). Tasks suffixed `-Parallel` use the parallel-linkage ankle model of the K1.
Tracking tasks run on flat terrain. List them all with:

```bash
uv run list_envs
```

## Deployment

Example code for deploying trained policies on the real Booster K1 is coming soon.

## Hungry for more?

booster_mjlab is a whIRLwind project. Our mission is to push the boundaries of robotics and AI in robot football. We are always looking for interested students or collaboration partners to join us. Learn more at [whirlwind.team](https://whirlwind.team).

<picture>
    <source
    srcset="./assets/whirlwind_logo_light.png"
    media="(prefers-color-scheme: dark)"
    height="125"
    />
    <img
    src="./assets/whirlwind_logo_dark.png"
    alt="whIRLwind logo"
    height="125"
    />
</picture>

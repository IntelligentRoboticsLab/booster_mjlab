from dataclasses import dataclass, field

from mjlab.rl.config import (
    RslRlBaseRunnerCfg,
    RslRlOnPolicyRunnerCfg as MjlabRslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg as MjlabRslRlPpoAlgorithmCfg,
)
from booster_mjlab.amp.config import AmpDiscriminatorCfg


@dataclass
class RslRlSymmetryCfg:
    """Configuration for the symmetry-augmentation in the training.

    When :meth:`use_data_augmentation` is True, the :meth:`data_augmentation_func` is used to generate
    augmented observations and actions. These are then used to train the model.

    When :meth:`use_mirror_loss` is True, the :meth:`mirror_loss_coeff` is used to weight the
    symmetry-mirror loss. This loss is directly added to the agent's loss function.

    If both :meth:`use_data_augmentation` and :meth:`use_mirror_loss` are False, then no symmetry-based
    training is enabled. However, the :meth:`data_augmentation_func` is called to compute and log
    symmetry metrics. This is useful for performing ablations.

    For more information, please check the work from :cite:`mittal2024symmetry`.
    """

    use_data_augmentation: bool = False
    """Whether to use symmetry-based data augmentation. Default is False."""

    use_mirror_loss: bool = False
    """Whether to use the symmetry-augmentation loss. Default is False."""

    data_augmentation_func: callable = None
    """The symmetry data augmentation function.

    The function signature should be as follows:

    Args:

        env (VecEnv): The environment object. This is used to access the environment's properties.
        obs (tensordict.TensorDict | None): The observation tensor dictionary. If None, the observation is not used.
        action (torch.Tensor | None): The action tensor. If None, the action is not used.

    Returns:
        A tuple containing the augmented observation dictionary and action tensors. The tensors can be None,
        if their respective inputs are None.
    """

    mirror_loss_coeff: float = 0.0
    """The weight for the symmetry-mirror loss. Default is 0.0."""


@dataclass
class RslRlPpoAlgorithmCfg(MjlabRslRlPpoAlgorithmCfg):
    symmetry_cfg: RslRlSymmetryCfg | None = None
    """The symmetry configuration."""


@dataclass
class RslRlMuonPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO with the same hybrid Muon/Adam updates as velocity AMP."""

    class_name: str = "booster_mjlab.rl.muon:MuonPPO"
    muon_weight_decay: float = 0.0
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5


@dataclass
class RslRlOnPolicyRunnerCfg(MjlabRslRlOnPolicyRunnerCfg):
    algorithm: RslRlPpoAlgorithmCfg = field(default_factory=RslRlPpoAlgorithmCfg)


FAST_SAC_CLASS_NAME = "booster_mjlab.rl.fast_sac.fast_sac:FastSAC"


@dataclass
class RslRlFastSacAlgorithmCfg:
    """Configuration for the FastSAC algorithm."""

    class_name: str = FAST_SAC_CLASS_NAME
    """Fully qualified class name resolved by rsl-rl's resolve_callable."""

    # Learning rates
    critic_lr: float = 3e-4
    """Critic learning rate."""
    actor_lr: float = 3e-4
    """Actor learning rate."""
    alpha_lr: float = 3e-4
    """Entropy temperature learning rate."""

    # SAC hyper-parameters
    gamma: float = 0.97
    """Discount factor."""
    tau: float = 0.125
    """Soft target update coefficient."""
    batch_size: int = 8192
    """Global batch size (split across envs)."""
    learning_starts: int = 10
    """Number of env steps before training begins."""
    policy_frequency: int = 4
    """Actor update frequency relative to critic updates."""
    num_updates: int = 8
    """Number of gradient updates per env step (UTD ratio)."""
    target_entropy_ratio: float = 0.0
    """Target entropy as a ratio of -n_act."""
    alpha_init: float = 0.001
    """Initial entropy temperature."""
    use_autotune: bool = True
    """Whether to auto-tune the entropy temperature."""

    # Network architecture
    actor_hidden_dim: int = 512
    """Actor hidden layer base width (halving: 512->256->128)."""
    critic_hidden_dim: int = 768
    """Critic hidden layer base width (halving: 768->384->192)."""
    num_atoms: int = 101
    """Number of atoms for the distributional critic (C51)."""
    v_min: float = -20.0
    """Minimum support value for C51."""
    v_max: float = 20.0
    """Maximum support value for C51."""
    num_q_networks: int = 2
    """Number of Q-networks in the critic ensemble."""
    use_layer_norm: bool = True
    """Whether to use LayerNorm in actor and critic."""
    use_tanh: bool = True
    """Whether to use tanh squashing on actor output."""
    log_std_min: float = -5.0
    """Minimum log standard deviation for the actor."""
    log_std_max: float = 0.0
    """Maximum log standard deviation for the actor."""

    # Replay buffer
    buffer_size: int = 1024
    """Per-env replay buffer capacity."""
    num_steps: int = 1
    """Number of n-step returns."""

    # Observation normalization
    obs_normalization: bool = True
    """Whether to use empirical observation normalization."""

    # Optimization
    max_grad_norm: float = 0.0
    """Maximum gradient norm (0 = no clipping)."""
    weight_decay: float = 0.001
    """AdamW weight decay."""

    # Performance
    compile: bool = True
    """Whether to use torch.compile for update functions."""
    amp: bool = True
    """Whether to use automatic mixed precision."""
    amp_dtype: str = "bf16"
    """AMP dtype: 'bf16' or 'fp16'."""
    amp_discriminator_lr: float = 1e-3
    """Learning rate for the AMP discriminator when AMP priors are enabled."""
    amp_grad_penalty_lambda: float = 10.0
    """Gradient-penalty coefficient for AMP discriminator training."""


@dataclass
class RslRlFastSacRunnerCfg(RslRlBaseRunnerCfg):
    """Runner configuration for FastSAC.

    FastSAC is off-policy, so we use num_steps_per_env=1 and let the
    algorithm handle replay internally.
    """

    class_name: str = "OnPolicyRunner"
    """Runner class name. We reuse the on-policy runner."""
    num_steps_per_env: int = 1
    """Must be 1 for off-policy SAC."""
    algorithm: RslRlFastSacAlgorithmCfg = field(
        default_factory=RslRlFastSacAlgorithmCfg
    )
    """The FastSAC algorithm configuration."""
    obs_groups: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {"actor": ("actor",), "critic": ("critic",)}
    )
    """Observation groups used by actor and critic networks."""
    dataset_root: str | None = None
    """Local motion path or Hugging Face dataset ID for FastSAC+AMP."""
    discriminator: AmpDiscriminatorCfg | None = None
    """AMP discriminator configuration (required only for FastSAC+AMP)."""
    style_reward_weight: float = 0.0
    """Weight of the style reward when Adversarial Motion Priors are used."""
    amp_replay_buffer_size: int = 200_000
    """Size of the AMP policy replay buffer."""
    amp_replay_insert_size: int = 1_000
    """Samples inserted per update once the AMP replay buffer is full."""
    num_amp_obs_steps: int = 10
    """Number of sequential observation frames in each AMP discriminator sample."""
    speed_factor: float = 1.0
    """Playback speed multiplier for AMP motion data."""
    dataset_weights: list[float] | None = None
    """Optional weights for each motion clip in the AMP dataset."""
    dataset_augmentations: list[dict[str, object]] | None = None
    """Optional AMP motion augmentations applied per clip."""
    dataset_transform: str | None = None
    """Optional ``module.path:function`` remapping each clip onto the robot's joint layout."""

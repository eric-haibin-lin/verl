# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Any, Optional

from verl.base_config import BaseConfig


@dataclass
class CheckpointConfig(BaseConfig):
    """Configuration for model checkpointing."""

    _frozen_fields = ["save_contents", "load_contents", "async_save"]

    save_contents: list[str] = field(default_factory=lambda: ["model", "optimizer", "extra"])
    """What to include in saved checkpoints. Options: 'model', 'optimizer', 'extra', 'hf_model'."""

    load_contents: list[str] = field(default_factory=lambda: ["model", "optimizer", "extra"])
    """Contents to load from checkpoint. Defaults to same as save_contents."""

    async_save: bool = False
    """Whether to save checkpoints asynchronously."""


@dataclass
class PolicyLossConfig(BaseConfig):
    """Configuration for policy loss computation."""

    _frozen_fields = ["loss_mode", "clip_cov_ratio", "clip_cov_lb", "clip_cov_ub", "kl_cov_ratio", "ppo_kl_coef"]

    loss_mode: str = "vanilla"
    """Loss function mode. Options: 'vanilla', 'clip-cov', 'kl-cov', 'gpg'."""

    clip_cov_ratio: float = 0.0002
    """Ratio of tokens to be clipped for clip-cov loss."""

    clip_cov_lb: float = 1.0
    """Lower bound for clip-cov loss."""

    clip_cov_ub: float = 5.0
    """Upper bound for clip-cov loss."""

    kl_cov_ratio: float = 0.0002
    """Ratio of tokens to be applied KL penalty for kl-cov loss."""

    ppo_kl_coef: float = 0.1
    """KL divergence penalty coefficient."""


@dataclass
class OptimConfig(BaseConfig):
    """Configuration for optimizer settings."""

    _frozen_fields = [
        "lr",
        "lr_warmup_steps_ratio",
        "total_training_steps",
        "weight_decay",
        "lr_warmup_steps",
        "lr_warmup_init",
        "lr_decay_steps",
        "lr_decay_style",
        "min_lr",
        "weight_decay_incr_style",
        "lr_wsd_decay_style",
        "lr_wsd_decay_steps",
        "use_checkpoint_opt_param_scheduler",
        "optimizer",
        "clip_grad",
        "min_lr_ratio",
        "num_cycles",
        "warmup_style",
    ]

    lr: float = 1e-6
    """Learning rate."""

    lr_warmup_steps_ratio: float = 0.0
    """Warmup steps ratio (used if lr_warmup_steps is negative)."""

    total_training_steps: int = -1
    """Total training steps (must be overridden at runtime)."""

    weight_decay: float = 0.01
    """Weight decay coefficient."""

    lr_warmup_steps: Optional[int] = None
    """Warmup steps; negative value delegates to lr_warmup_steps_ratio."""

    lr_warmup_init: float = 0.0
    """Initial learning rate for warmup."""

    lr_decay_steps: Optional[int] = None
    """Number of steps for learning rate decay."""

    lr_decay_style: str = "constant"
    """Learning rate decay style."""

    min_lr: float = 0.0
    """Minimum learning rate."""

    weight_decay_incr_style: str = "constant"
    """Weight decay increment style."""

    lr_wsd_decay_style: str = "exponential"
    """Learning rate warmup-stable-decay style."""

    lr_wsd_decay_steps: Optional[int] = None
    """Steps for warmup-stable-decay schedule."""

    use_checkpoint_opt_param_scheduler: bool = False
    """Whether to use checkpoint optimizer parameter scheduler."""

    optimizer: str = "adam"
    """Optimizer type."""

    clip_grad: float = 1.0
    """Gradient clipping threshold."""

    min_lr_ratio: float = 0.0
    """Minimum LR ratio for cosine schedule."""

    num_cycles: float = 0.5
    """Number of cosine cycles in LR schedule."""

    warmup_style: str = "constant"
    """LR warmup style: 'constant' or 'cosine'."""


@dataclass
class MegatronConfig(BaseConfig):
    """Configuration for Megatron parallelism."""

    _frozen_fields = [
        "param_offload",
        "grad_offload",
        "optimizer_offload",
        "tensor_model_parallel_size",
        "expert_model_parallel_size",
        "expert_tensor_parallel_size",
        "pipeline_model_parallel_size",
        "virtual_pipeline_model_parallel_size",
        "context_parallel_size",
        "sequence_parallel",
        "use_distributed_optimizer",
        "use_dist_checkpointing",
        "dist_checkpointing_path",
        "seed",
        "override_ddp_config",
        "override_transformer_config",
        "use_mbridge",
    ]

    param_offload: bool = False
    """Whether to offload parameters to CPU."""

    grad_offload: bool = False
    """Whether to offload gradients to CPU."""

    optimizer_offload: bool = False
    """Whether to offload optimizer states to CPU."""

    tensor_model_parallel_size: int = 1
    """Tensor model parallel size."""

    expert_model_parallel_size: int = 1
    """Expert model parallel size for MoE models."""

    expert_tensor_parallel_size: Optional[int] = None
    """Expert tensor parallel size for MoE models."""

    pipeline_model_parallel_size: int = 1
    """Pipeline model parallel size."""

    virtual_pipeline_model_parallel_size: Optional[int] = None
    """Virtual pipeline model parallel size for interleaved scheduling."""

    context_parallel_size: int = 1
    """Context parallel size for long sequences."""

    sequence_parallel: bool = True
    """Whether to enable sequence parallelism."""

    use_distributed_optimizer: bool = True
    """Whether to use distributed optimizer."""

    use_dist_checkpointing: bool = False
    """Whether to use distributed checkpointing."""

    dist_checkpointing_path: Optional[str] = None
    """Path for distributed checkpointing."""

    seed: int = 42
    """Random seed for reproducibility."""

    override_ddp_config: dict[str, Any] = field(default_factory=dict)
    """Override configuration for DDP."""

    override_transformer_config: dict[str, Any] = field(default_factory=dict)
    """Override configuration for transformer."""

    use_mbridge: bool = False
    """Whether to use MBridge for communication."""


@dataclass
class ProfileConfig(BaseConfig):
    """Configuration for profiling."""

    use_profile: bool = False
    """Whether to enable profiling."""

    profile_ranks: Optional[list[int]] = None
    """List of ranks to profile. None means all ranks."""

    step_start: int = -1
    """Starting step for profiling."""

    step_end: int = -1
    """Ending step for profiling."""

    save_path: Optional[str] = None
    """Path to save profiling results."""


@dataclass
class WrapPolicyConfig(BaseConfig):
    """Configuration for FSDP wrap policy."""

    min_num_params: int = 0
    """Minimum number of parameters for a module to be wrapped by FSDP."""


@dataclass
class FSDPConfig(BaseConfig):
    """Configuration for FSDP (Fully Sharded Data Parallel)."""

    wrap_policy: WrapPolicyConfig = field(default_factory=WrapPolicyConfig)
    """Configuration for FSDP wrap policy."""

    param_offload: bool = False
    """Whether to offload parameters to CPU."""

    optimizer_offload: bool = False
    """Whether to offload optimizer states to CPU."""

    offload_policy: bool = False
    """Whether to offload policy model parameters."""

    reshard_after_forward: bool = True
    """Whether to reshard parameters after forward pass."""

    fsdp_size: int = -1
    """FSDP group size. -1 means use all available GPUs."""

    forward_prefetch: bool = False
    """Whether to prefetch parameters for next forward pass."""


@dataclass
class ActorConfig(BaseConfig):
    """Base configuration for actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.
    """

    _frozen_fields = [
        "strategy",
        "ppo_mini_batch_size",
        "ppo_micro_batch_size",
        "ppo_micro_batch_size_per_gpu",
        "use_dynamic_bsz",
        "ppo_max_token_len_per_gpu",
        "clip_ratio",
        "clip_ratio_low",
        "clip_ratio_high",
        "clip_ratio_c",
        "loss_agg_mode",
        "entropy_coeff",
        "use_kl_loss",
        "use_torch_compile",
        "kl_loss_coef",
        "kl_loss_type",
        "ppo_epochs",
        "shuffle",
    ]

    strategy: str = "???"
    """Training strategy. Must be overridden in subclasses."""

    ppo_mini_batch_size: int = 256
    """Mini-batch size for PPO training."""

    ppo_micro_batch_size: Optional[int] = None
    """Micro-batch size for PPO training. If None, uses ppo_micro_batch_size_per_gpu."""

    ppo_micro_batch_size_per_gpu: Optional[int] = None
    """Micro-batch size per GPU for PPO training."""

    use_dynamic_bsz: bool = False
    """Whether to use dynamic batch sizing."""

    ppo_max_token_len_per_gpu: int = 16384
    """Maximum token length per GPU for PPO training."""

    clip_ratio: float = 0.2
    """PPO clipping ratio for policy loss."""

    clip_ratio_low: float = 0.2
    """Lower bound for PPO clipping ratio."""

    clip_ratio_high: float = 0.2
    """Upper bound for PPO clipping ratio."""

    policy_loss: PolicyLossConfig = field(default_factory=PolicyLossConfig)
    """Configuration for policy loss computation."""

    clip_ratio_c: float = 3.0
    """Clipping ratio for critic loss."""

    loss_agg_mode: str = "token-mean"
    """Loss aggregation mode. Options: 'token-mean', 'sample-mean'."""

    entropy_coeff: float = 0
    """Entropy coefficient for regularization."""

    use_kl_loss: bool = False
    """Whether to use KL divergence loss."""

    use_torch_compile: bool = True
    """Whether to use torch.compile for optimization."""

    kl_loss_coef: float = 0.001
    """KL divergence loss coefficient."""

    kl_loss_type: str = "low_var_kl"
    """Type of KL loss to use."""

    ppo_epochs: int = 1
    """Number of PPO epochs per training step."""

    shuffle: bool = False
    """Whether to shuffle data during training."""

    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    """Configuration for checkpointing."""

    optim: OptimConfig = field(default_factory=OptimConfig)
    """Configuration for optimizer."""


@dataclass
class MegatronActorConfig(ActorConfig):
    """Configuration for Megatron actor models.

    The inheritance from ActorConfig provides all base actor configuration fields,
    with additional Megatron-specific settings.
    """

    _frozen_fields = ActorConfig._frozen_fields + ["data_loader_seed", "load_weight"]

    strategy: str = "megatron"
    """Training strategy set to 'megatron' for Megatron parallelism."""

    data_loader_seed: Optional[int] = None
    """Seed for data loader. If None, uses global seed."""

    load_weight: bool = True
    """Whether to load model weights from checkpoint."""

    megatron: MegatronConfig = field(default_factory=MegatronConfig)
    """Configuration for Megatron parallelism settings."""

    profile: ProfileConfig = field(default_factory=ProfileConfig)
    """Configuration for profiling settings."""


@dataclass
class FSDPActorConfig(ActorConfig):
    """Configuration for FSDP actor models.

    The inheritance from ActorConfig provides all base actor configuration fields,
    with additional FSDP-specific settings.
    """

    _frozen_fields = ActorConfig._frozen_fields + [
        "grad_clip",
        "ulysses_sequence_parallel_size",
        "entropy_from_logits_with_chunking",
        "entropy_checkpointing",
    ]

    strategy: str = "fsdp"
    """Training strategy set to 'fsdp' for Fully Sharded Data Parallel."""

    grad_clip: float = 1.0
    """Gradient clipping threshold."""

    ulysses_sequence_parallel_size: int = 1
    """Ulysses sequence parallel size for long sequences."""

    entropy_from_logits_with_chunking: bool = False
    """Whether to compute entropy from logits with chunking for memory efficiency."""

    entropy_checkpointing: bool = False
    """Whether to use gradient checkpointing for entropy computation."""

    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)
    """Configuration for FSDP settings."""

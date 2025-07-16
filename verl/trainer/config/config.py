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
    """Configuration for checkpointing."""

    _frozen_fields = ["save_contents", "load_contents", "async_save"]

    save_contents: list[str] = field(default_factory=lambda: ["model", "optimizer", "extra"])
    load_contents: list[str] = field(default_factory=lambda: ["model", "optimizer", "extra"])
    async_save: bool = False


@dataclass
class PolicyLossConfig(BaseConfig):
    """Configuration for policy loss."""

    _frozen_fields = ["loss_mode", "clip_cov_ratio", "clip_cov_lb", "clip_cov_ub", "kl_cov_ratio", "ppo_kl_coef"]

    loss_mode: str = "vanilla"
    clip_cov_ratio: float = 0.0002
    clip_cov_lb: float = 1.0
    clip_cov_ub: float = 5.0
    kl_cov_ratio: float = 0.0002
    ppo_kl_coef: float = 0.1


@dataclass
class OptimConfig(BaseConfig):
    """Configuration for optimizer."""

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
    lr_warmup_steps_ratio: float = 0.0
    total_training_steps: int = -1
    weight_decay: float = 0.01
    lr_warmup_steps: Optional[int] = None
    lr_warmup_init: float = 0.0
    lr_decay_steps: Optional[int] = None
    lr_decay_style: str = "constant"
    min_lr: float = 0.0
    weight_decay_incr_style: str = "constant"
    lr_wsd_decay_style: str = "exponential"
    lr_wsd_decay_steps: Optional[int] = None
    use_checkpoint_opt_param_scheduler: bool = False
    optimizer: str = "adam"
    clip_grad: float = 1.0
    min_lr_ratio: float = 0.0
    num_cycles: float = 0.5
    warmup_style: str = "constant"


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
    grad_offload: bool = False
    optimizer_offload: bool = False
    tensor_model_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: Optional[int] = None
    pipeline_model_parallel_size: int = 1
    virtual_pipeline_model_parallel_size: Optional[int] = None
    context_parallel_size: int = 1
    sequence_parallel: bool = True
    use_distributed_optimizer: bool = True
    use_dist_checkpointing: bool = False
    dist_checkpointing_path: Optional[str] = None
    seed: int = 42
    override_ddp_config: dict[str, Any] = field(default_factory=dict)
    override_transformer_config: dict[str, Any] = field(default_factory=dict)
    use_mbridge: bool = False


@dataclass
class ProfileConfig(BaseConfig):
    """Configuration for profiling."""

    use_profile: bool = False
    profile_ranks: Optional[list[int]] = None
    step_start: int = -1
    step_end: int = -1
    save_path: Optional[str] = None


@dataclass
class WrapPolicyConfig(BaseConfig):
    """Configuration for FSDP wrap policy."""

    min_num_params: int = 0


@dataclass
class FSDPConfig(BaseConfig):
    """Configuration for FSDP."""

    wrap_policy: WrapPolicyConfig = field(default_factory=WrapPolicyConfig)
    param_offload: bool = False
    optimizer_offload: bool = False
    offload_policy: bool = False
    reshard_after_forward: bool = True
    fsdp_size: int = -1
    forward_prefetch: bool = False


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
    ppo_mini_batch_size: int = 256
    ppo_micro_batch_size: Optional[int] = None
    ppo_micro_batch_size_per_gpu: Optional[int] = None
    use_dynamic_bsz: bool = False
    ppo_max_token_len_per_gpu: int = 16384
    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2
    policy_loss: PolicyLossConfig = field(default_factory=PolicyLossConfig)
    clip_ratio_c: float = 3.0
    loss_agg_mode: str = "token-mean"
    entropy_coeff: float = 0
    use_kl_loss: bool = False
    use_torch_compile: bool = True
    kl_loss_coef: float = 0.001
    kl_loss_type: str = "low_var_kl"
    ppo_epochs: int = 1
    shuffle: bool = False
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)


@dataclass
class MegatronActorConfig(ActorConfig):
    """Configuration for Megatron actor models.

    The inheritance from ActorConfig provides all base actor configuration fields,
    with additional Megatron-specific settings.
    """

    _frozen_fields = ActorConfig._frozen_fields + ["data_loader_seed", "load_weight"]

    strategy: str = "megatron"
    data_loader_seed: Optional[int] = None
    load_weight: bool = True
    megatron: MegatronConfig = field(default_factory=MegatronConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)


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
    grad_clip: float = 1.0
    ulysses_sequence_parallel_size: int = 1
    entropy_from_logits_with_chunking: bool = False
    entropy_checkpointing: bool = False
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)

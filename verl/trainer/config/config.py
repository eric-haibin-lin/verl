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
    """Configuration for model checkpointing.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        save_contents (list[str]): What to include in saved checkpoints.
            Options: 'model', 'optimizer', 'extra', 'hf_model'.
        load_contents (list[str]): Contents to load from checkpoint. Defaults to same as save_contents.
        async_save (bool): Whether to save checkpoints asynchronously.
    """

    _frozen_fields = ["save_contents", "load_contents", "async_save"]

    save_contents: list[str] = field(default_factory=lambda: ["model", "optimizer", "extra"])
    load_contents: list[str] = field(default_factory=lambda: ["model", "optimizer", "extra"])
    async_save: bool = False


@dataclass
class PolicyLossConfig(BaseConfig):
    """Configuration for policy loss computation.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        loss_mode (str): Loss function mode. Options: 'vanilla', 'clip-cov', 'kl-cov', 'gpg'.
        clip_cov_ratio (float): Ratio of tokens to be clipped for clip-cov loss.
        clip_cov_lb (float): Lower bound for clip-cov loss.
        clip_cov_ub (float): Upper bound for clip-cov loss.
        kl_cov_ratio (float): Ratio of tokens to be applied KL penalty for kl-cov loss.
        ppo_kl_coef (float): KL divergence penalty coefficient.
    """

    _frozen_fields = ["loss_mode", "clip_cov_ratio", "clip_cov_lb", "clip_cov_ub", "kl_cov_ratio", "ppo_kl_coef"]

    loss_mode: str = "vanilla"
    clip_cov_ratio: float = 0.0002
    clip_cov_lb: float = 1.0
    clip_cov_ub: float = 5.0
    kl_cov_ratio: float = 0.0002
    ppo_kl_coef: float = 0.1


@dataclass
class MegatronEngineConfig(BaseConfig):
    """Configuration for Megatron parallelism.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        param_offload (bool): Whether to offload parameters to CPU.
        grad_offload (bool): Whether to offload gradients to CPU.
        optimizer_offload (bool): Whether to offload optimizer states to CPU.
        tensor_model_parallel_size (int): Tensor model parallel size.
        expert_model_parallel_size (int): Expert model parallel size for MoE models.
        expert_tensor_parallel_size (Optional[int]): Expert tensor parallel size for MoE models.
        pipeline_model_parallel_size (int): Pipeline model parallel size.
        virtual_pipeline_model_parallel_size (Optional[int]): Virtual pipeline model parallel size
            for interleaved scheduling.
        context_parallel_size (int): Context parallel size for long sequences.
        sequence_parallel (bool): Whether to enable sequence parallelism.
        use_distributed_optimizer (bool): Whether to use distributed optimizer.
        use_dist_checkpointing (bool): Whether to use distributed checkpointing.
        dist_checkpointing_path (Optional[str]): Path for distributed checkpointing.
        seed (int): Random seed for reproducibility.
        override_ddp_config (dict[str, Any]): Override configuration for DDP.
        override_transformer_config (dict[str, Any]): Override configuration for transformer.
        use_mbridge (bool): Whether to use MBridge for communication.
    """

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
    """Configuration for profiling.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        use_profile (bool): Whether to enable profiling.
        profile_ranks (Optional[list[int]]): List of ranks to profile. None means all ranks.
        step_start (int): Starting step for profiling.
        step_end (int): Ending step for profiling.
        save_path (Optional[str]): Path to save profiling results.
    """

    use_profile: bool = False
    profile_ranks: Optional[list[int]] = None
    step_start: int = -1
    step_end: int = -1
    save_path: Optional[str] = None


@dataclass
class FSDPEngineConfig(BaseConfig):
    """Configuration for FSDP (Fully Sharded Data Parallel).

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        wrap_policy (Dict[str, Any]): Configuration for FSDP wrap policy.
        param_offload (bool): Whether to offload parameters to CPU.
        optimizer_offload (bool): Whether to offload optimizer states to CPU.
        offload_policy (bool): Whether to offload policy model parameters.
        reshard_after_forward (bool): Whether to reshard parameters after forward pass.
        fsdp_size (int): FSDP group size. -1 means use all available GPUs.
        forward_prefetch (bool): Whether to prefetch parameters for next forward pass.
    """

    wrap_policy: dict[str, Any] = field(default_factory=dict)

    param_offload: bool = False
    optimizer_offload: bool = False
    offload_policy: bool = False
    reshard_after_forward: bool = True
    fsdp_size: int = -1
    forward_prefetch: bool = False


@dataclass
class ActorConfig(BaseConfig):
    """Configuration for actor model training.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy. Must be overridden in subclasses.
        ppo_mini_batch_size (int): Mini-batch size for PPO training.
        ppo_micro_batch_size (Optional[int]): Micro-batch size for PPO training.
            If None, uses ppo_micro_batch_size_per_gpu.
        ppo_micro_batch_size_per_gpu (Optional[int]): Micro-batch size per GPU for PPO training.
        use_dynamic_bsz (bool): Whether to use dynamic batch sizing.
        ppo_max_token_len_per_gpu (int): Maximum token length per GPU for PPO training.
        clip_ratio (float): PPO clipping ratio for policy loss.
        clip_ratio_low (float): Lower bound for PPO clipping ratio.
        clip_ratio_high (float): Upper bound for PPO clipping ratio.
        policy_loss (PolicyLossConfig): Configuration for policy loss computation.
        clip_ratio_c (float): Clipping ratio for critic loss.
        loss_agg_mode (str): Loss aggregation mode. Options: 'token-mean', 'sample-mean'.
        entropy_coeff (float): Entropy coefficient for regularization.
        use_kl_loss (bool): Whether to use KL divergence loss.
        use_torch_compile (bool): Whether to use torch.compile for optimization.
        kl_loss_coef (float): KL divergence loss coefficient.
        kl_loss_type (str): Type of KL loss to use.
        ppo_epochs (int): Number of PPO epochs per training step.
        shuffle (bool): Whether to shuffle data during training.
        checkpoint (CheckpointConfig): Configuration for checkpointing.
        optim (Dict[str, Any]): Configuration for optimizer.
    """

    _frozen_fields = [
        "strategy",
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

    strategy: str = field(default="")
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
    optim: dict[str, Any] = field(default_factory=dict)


@dataclass
class MegatronActorConfig(ActorConfig):
    """Configuration for Megatron actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy set to 'megatron' for Megatron parallelism.
        data_loader_seed (Optional[int]): Seed for data loader. If None, uses global seed.
        load_weight (bool): Whether to load model weights from checkpoint.
        megatron (MegatronEngineConfig): Configuration for Megatron parallelism settings.
        profile (ProfileConfig): Configuration for profiling settings.
    """

    _frozen_fields = ActorConfig._frozen_fields + ["data_loader_seed", "load_weight"]

    strategy: str = "megatron"
    data_loader_seed: Optional[int] = None
    load_weight: bool = True
    megatron: MegatronEngineConfig = field(default_factory=MegatronEngineConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)


@dataclass
class FSDPActorConfig(ActorConfig):
    """Configuration for FSDP actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy set to 'fsdp' for Fully Sharded Data Parallel.
        grad_clip (float): Gradient clipping threshold.
        ulysses_sequence_parallel_size (int): Ulysses sequence parallel size for long sequences.
        entropy_from_logits_with_chunking (bool): Whether to compute entropy from logits
            with chunking for memory efficiency.
        entropy_checkpointing (bool): Whether to use gradient checkpointing for entropy computation.
        fsdp_config (FSDPEngineConfig): Configuration for FSDP settings.
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
    fsdp_config: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)


# Note(haibin.lin): kw_only=True is required as BaseConfig specifies `extra` with default values
# If all of child class fields have default values, `kw_only=True` is not required.
@dataclass(kw_only=True)
class CriticConfig(BaseConfig):
    """Configuration for critic model training.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Strategy used for critic model training (fsdp, fsdp2, megatron).
        ppo_micro_batch_size_per_gpu (int): Local per-GPU micro batch size.
        rollout_n (int): Number of rollouts per update (mirrors actor rollout_n).
        optim (Dict[str, Any]): Optimizer configuration including lr, weight_decay, etc.
        model (Dict[str, Any]): Model configuration including path, tokenizer_path, etc.
        ppo_mini_batch_size (int): PPO mini-batch size per update.
        ppo_micro_batch_size (Optional[int]): Global micro batch size (deprecated).
        use_dynamic_bsz (bool): Whether to automatically adjust batch size at runtime.
        ppo_max_token_len_per_gpu (int): Max tokens per GPU in one PPO batch.
        forward_max_token_len_per_gpu (int): Max token length per GPU in forward pass.
        ppo_epochs (int): Number of PPO epochs per batch.
        shuffle (bool): Shuffle training data across PPO epochs.
        cliprange_value (float): PPO value function clipping range.
        loss_agg_mode (str): Loss aggregation mode.
        checkpoint (Dict[str, Any]): Checkpoint configuration.
        profiler (Dict[str, Any]): Profiler configuration.
    """

    strategy: str
    ppo_micro_batch_size_per_gpu: Optional[int] = None

    # For legacy reason configs related to batch_size are mutated in each role
    # In the future they will be added to frozen fields instead
    _frozen_fields = [
        "rollout_n",
        "strategy",
        "use_dynamic_bsz",
        "ppo_max_token_len_per_gpu",
        "forward_max_token_len_per_gpu",
        "ppo_epochs",
        "shuffle",
        "cliprange_value",
        "loss_agg_mode",
    ]

    rollout_n: int = 1
    ppo_mini_batch_size: int = 1
    use_dynamic_bsz: bool = False
    ppo_max_token_len_per_gpu: int = 32768
    forward_max_token_len_per_gpu: int = 32768
    ppo_epochs: int = 1
    shuffle: bool = True
    cliprange_value: float = 0.5
    loss_agg_mode: str = "token-mean"
    ppo_micro_batch_size: Optional[int] = None
    optim: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    checkpoint: dict[str, Any] = field(default_factory=dict)
    profiler: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate critic configuration parameters."""
        if not self.use_dynamic_bsz:
            self._check_mutually_exclusive(self.ppo_micro_batch_size, self.ppo_micro_batch_size_per_gpu, "critic")

            if self.ppo_micro_batch_size is not None:
                if self.ppo_mini_batch_size % self.ppo_micro_batch_size != 0:
                    raise ValueError(
                        f"[critic] ppo_mini_batch_size ({self.ppo_mini_batch_size}) must be divisible by "
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size})"
                    )

    def validate(self, n_gpus: int, train_batch_size: int):
        """Validate critic configuration with runtime parameters.

        Args:
            n_gpus: Total number of GPUs available
            train_batch_size: Training batch size from data config
        """
        if not self.use_dynamic_bsz:
            if train_batch_size < self.ppo_mini_batch_size:
                raise ValueError(
                    f"train_batch_size ({train_batch_size}) must be >= "
                    f"critic.ppo_mini_batch_size ({self.ppo_mini_batch_size})"
                )

    @staticmethod
    def _check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        """Validate mutually exclusive micro batch size configuration options.

        Ensures that users don't set both deprecated micro_batch_size and
        the new micro_batch_size_per_gpu parameters simultaneously.

        Args:
            mbs: Deprecated micro batch size parameter value.
            mbs_per_gpu: New micro batch size per GPU parameter value.
            name (str): Configuration section name for error messages.

        Raises:
            ValueError: If both parameters are set or neither is set.
        """
        param = "micro_batch_size"
        param_per_gpu = f"{param}_per_gpu"

        if mbs is None and mbs_per_gpu is None:
            raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

        if mbs is not None and mbs_per_gpu is not None:
            raise ValueError(
                f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
            )


@dataclass
class MegatronCriticConfig(CriticConfig):
    """Configuration for Megatron-based critic model training.

    The inheritance from CriticConfig provides all base critic configuration plus Megatron-specific settings.

    Args:
        nccl_timeout (int): NCCL timeout in seconds for distributed operations.
        megatron (Dict[str, Any]): Megatron-specific parallelism settings.
        load_weight (bool): Whether to load initial weights.
        data_loader_seed (Optional[int]): Seed for data loader.
    """

    _frozen_fields = CriticConfig._frozen_fields + [
        "nccl_timeout",
        "load_weight",
        "data_loader_seed",
    ]

    strategy: str = "megatron"
    nccl_timeout: int = 600
    megatron: dict[str, Any] = field(default_factory=dict)
    load_weight: bool = True
    data_loader_seed: Optional[int] = None

    def validate(self, n_gpus: int, train_batch_size: int):
        """Validate Megatron critic configuration with runtime parameters."""
        super().validate(n_gpus, train_batch_size)


@dataclass
class FSDPCriticConfig(CriticConfig):
    """Configuration for FSDP-based critic model training.

    The inheritance from CriticConfig provides all base critic configuration plus FSDP-specific settings.

    Args:
        forward_micro_batch_size (int): Forward-only batch size during inference (global).
        forward_micro_batch_size_per_gpu (int): Forward-only batch size during inference (per GPU).
        ulysses_sequence_parallel_size (int): Sequence parallelism size for Ulysses-style model parallelism.
        grad_clip (float): Gradient clipping for critic updates.
    """

    _frozen_fields = CriticConfig._frozen_fields + [
        "ulysses_sequence_parallel_size",
        "grad_clip",
    ]

    strategy: str = "fsdp"
    forward_micro_batch_size: int = 1
    forward_micro_batch_size_per_gpu: int = 1
    ulysses_sequence_parallel_size: int = 1
    grad_clip: float = 1.0

    def __post_init__(self):
        """Validate FSDP critic configuration parameters."""
        super().__post_init__()

        if self.strategy in {"fsdp", "fsdp2"}:
            if self.ulysses_sequence_parallel_size > 1:
                if not self.model.get("use_remove_padding", False):
                    raise ValueError(
                        "When using sequence parallelism for critic, you must enable `use_remove_padding`."
                    )

    def validate(self, n_gpus: int, train_batch_size: int):
        """Validate FSDP critic configuration with runtime parameters."""
        super().validate(n_gpus, train_batch_size)

        if not self.use_dynamic_bsz:
            sp_size = self.ulysses_sequence_parallel_size
            if self.ppo_micro_batch_size is not None:
                if self.ppo_micro_batch_size * sp_size < n_gpus:
                    raise ValueError(
                        f"critic.ppo_micro_batch_size ({self.ppo_micro_batch_size}) * "
                        f"ulysses_sequence_parallel_size ({sp_size}) must be >= n_gpus ({n_gpus})"
                    )

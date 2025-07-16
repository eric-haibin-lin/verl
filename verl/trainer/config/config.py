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
from verl.utils.profiler import ProfilerConfig


@dataclass
class CriticConfig(BaseConfig):
    """Configuration for critic model training.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        rollout_n (int): Number of rollouts per update (mirrors actor rollout_n).
        strategy (str): Strategy used for critic model training (fsdp, fsdp2, megatron).
        optim (Dict[str, Any]): Optimizer configuration including lr, weight_decay, etc.
        model (Dict[str, Any]): Model configuration including path, tokenizer_path, etc.
        ppo_mini_batch_size (int): PPO mini-batch size per update.
        ppo_micro_batch_size (Optional[int]): Global micro batch size (deprecated).
        ppo_micro_batch_size_per_gpu (Optional[int]): Local per-GPU micro batch size.
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

    strategy: str = "fsdp"
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
    ppo_micro_batch_size_per_gpu: Optional[int] = None
    optim: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    checkpoint: dict[str, Any] = field(default_factory=dict)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)

    def __post_init__(self):
        """Validate critic configuration parameters."""
        if not self.use_dynamic_bsz:
            self._check_mutually_exclusive(self.ppo_micro_batch_size, self.ppo_micro_batch_size_per_gpu, "critic")

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
        settings = {
            "actor_rollout_ref.actor": "micro_batch_size",
            "critic": "micro_batch_size",
            "reward_model": "micro_batch_size",
            "actor_rollout_ref.ref": "log_prob_micro_batch_size",
            "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
        }

        if name in settings:
            param = settings[name]
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

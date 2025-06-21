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

from dataclasses import dataclass

__all__ = ["OptimConfig"]


@dataclass
class OptimConfig:
    """Optimizer configuration dataclass."""

    lr: float = 1e-6

    lr_warmup_steps: int = -1

    lr_warmup_steps_ratio: float = 0.0

    min_lr_ratio: float = 0.0

    num_cycles: float = 0.5

    warmup_style: str = "constant"

    total_training_steps: int = -1

    weight_decay: float = 0.01

    use_checkpoint_opt_param_scheduler: bool = False

    betas: tuple = (0.9, 0.999)

    warmup_steps_ratio: float = 0.0

    lr_scheduler: str = "cosine"

    clip_grad: float = 1.0

    grad_clip: float = 1.0

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

from dataclasses import dataclass, is_dataclass
from typing import Any, Dict, Type, Union

from omegaconf import DictConfig, OmegaConf

__all__ = ["omega_conf_to_dataclass", "OptimConfig"]


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


def omega_conf_to_dataclass(config: Union[DictConfig, dict], dataclass_type: Type[Any]) -> Any:
    """
    Convert an OmegaConf DictConfig to a dataclass.

    Args:
        config: The OmegaConf DictConfig or dict to convert.
        dataclass_type: The dataclass type to convert to.

    Returns:
        The dataclass instance.
    """
    if not is_dataclass(dataclass_type):
        raise ValueError(f"{dataclass_type} must be a dataclass")
    cfg = OmegaConf.create(config)  # in case it's a dict
    cfg_from_dataclass = OmegaConf.structured(dataclass_type)
    # let cfg override the existing vals in `cfg_from_dataclass`
    cfg_merged = OmegaConf.merge(cfg_from_dataclass, cfg)
    # now convert to `dataclass_type`
    config_object = OmegaConf.to_object(cfg_merged)
    return config_object


def update_dict_with_config(dictionary: Dict, config: DictConfig):
    for key in dictionary:
        if hasattr(config, key):
            dictionary[key] = getattr(config, key)

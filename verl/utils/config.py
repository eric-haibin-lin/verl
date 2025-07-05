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

from dataclasses import is_dataclass
from typing import Any, Dict, Optional, Type, Union

from omegaconf import DictConfig, OmegaConf

__all__ = ["omega_conf_to_dataclass"]


def omega_conf_to_dataclass(
    config: Union[DictConfig, dict],
    dataclass_type: Optional[Type[Any]] = None,
    recursive: bool = False,
) -> Any:
    """
    Convert an OmegaConf DictConfig to a dataclass.

    Args:
        config: The OmegaConf DictConfig or dict to convert.
        dataclass_type: The dataclass type to convert to. When dataclass_type is None,
            the DictConfig must contain _target_ to be instantiated via hydra.instantiate API.
        recursive: If True, recursively process nested configs that contain _target_ fields.

    Returns:
        The dataclass instance.
    """
    if recursive and dataclass_type is None:
        import copy

        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)
        config = copy.deepcopy(config)
        config = _process_config_recursively(config)
        return config

    if dataclass_type is not None and isinstance(config, dataclass_type):
        return config

    if dataclass_type is None:
        assert "_target_" in config, (
            "When dataclass_type is not provided, config must contain _target_."
            "See trainer/config/ppo_trainer.yaml algorithm section for an example."
        )
        from hydra.utils import instantiate

        return instantiate(config, _convert_="partial")

    if not is_dataclass(dataclass_type):
        raise ValueError(f"{dataclass_type} must be a dataclass")
    cfg = OmegaConf.create(config)  # in case it's a dict
    cfg_from_dataclass = OmegaConf.structured(dataclass_type)
    # let cfg override the existing vals in `cfg_from_dataclass`
    cfg_merged = OmegaConf.merge(cfg_from_dataclass, cfg)
    # now convert to `dataclass_type`
    config_object = OmegaConf.to_object(cfg_merged)
    return config_object


def _process_config_recursively(config: Union[DictConfig, dict]) -> Any:
    """
    Recursively process a config, instantiating any nested configs that contain _target_ fields.

    Args:
        config: The config to process recursively.

    Returns:
        The processed config with _target_ fields instantiated.
    """
    if not isinstance(config, (dict, DictConfig)):
        return config

    for key, value in list(config.items()):
        if isinstance(value, (dict, DictConfig)):
            if "_target_" in value:
                from hydra.utils import instantiate

                config[key] = instantiate(value, _convert_="partial")
            else:
                config[key] = _process_config_recursively(value)

    if "_target_" in config:
        from hydra.utils import instantiate

        return instantiate(config, _convert_="partial")

    return config


def update_dict_with_config(dictionary: Dict, config: DictConfig):
    for key in dictionary:
        if hasattr(config, key):
            dictionary[key] = getattr(config, key)

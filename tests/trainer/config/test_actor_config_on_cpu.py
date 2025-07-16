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

import os
import unittest

from verl.trainer.config import ActorConfig, FSDPActorConfig, MegatronActorConfig
from verl.utils.config import omega_conf_to_dataclass


class TestActorConfig(unittest.TestCase):
    """Test the ActorConfig dataclass and its variants."""



    def test_config_inheritance(self):
        """Test that the inheritance hierarchy works correctly."""
        megatron_dict = {
            "_target_": "verl.trainer.config.MegatronActorConfig",
            "strategy": "megatron",
            "ppo_mini_batch_size": 256,
            "clip_ratio": 0.2,
        }
        fsdp_dict = {
            "_target_": "verl.trainer.config.FSDPActorConfig",
            "strategy": "fsdp",
            "ppo_mini_batch_size": 256,
            "clip_ratio": 0.2,
        }

        megatron_config = omega_conf_to_dataclass(megatron_dict)
        fsdp_config = omega_conf_to_dataclass(fsdp_dict)

        self.assertIsInstance(megatron_config, ActorConfig)
        self.assertIsInstance(fsdp_config, ActorConfig)

        self.assertEqual(megatron_config.ppo_mini_batch_size, fsdp_config.ppo_mini_batch_size)
        self.assertEqual(megatron_config.clip_ratio, fsdp_config.clip_ratio)

    def test_actor_config_from_yaml(self):
        """Test creating ActorConfig from YAML file."""
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config/actor")):
            cfg = compose(config_name="actor")

        config = omega_conf_to_dataclass(cfg)

        self.assertIsInstance(config, ActorConfig)
        self.assertEqual(config.strategy, "fsdp")

    def test_fsdp_actor_config_from_yaml(self):
        """Test creating FSDPActorConfig from YAML file."""
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config/actor")):
            cfg = compose(config_name="dp_actor")

        config = omega_conf_to_dataclass(cfg)

        self.assertIsInstance(config, FSDPActorConfig)
        self.assertEqual(config.strategy, "fsdp")

    def test_megatron_actor_config_from_yaml(self):
        """Test creating MegatronActorConfig from YAML file."""
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config/actor")):
            cfg = compose(config_name="megatron_actor")

        config = omega_conf_to_dataclass(cfg)

        self.assertIsInstance(config, MegatronActorConfig)
        self.assertEqual(config.strategy, "megatron")

    def test_config_get_method(self):
        """Test the get method for backward compatibility."""
        config_dict = {
            "_target_": "verl.trainer.config.ActorConfig",
            "strategy": "fsdp",
            "ppo_mini_batch_size": 256,
        }
        config = omega_conf_to_dataclass(config_dict)

        self.assertEqual(config.get("strategy"), "fsdp")
        self.assertEqual(config.get("ppo_mini_batch_size"), 256)

        self.assertIsNone(config.get("non_existing"))
        self.assertEqual(config.get("non_existing", "default"), "default")

    def test_config_dict_like_access(self):
        """Test dictionary-like access to config fields."""
        config_dict = {
            "_target_": "verl.trainer.config.ActorConfig",
            "strategy": "fsdp",
            "ppo_mini_batch_size": 256,
        }
        config = omega_conf_to_dataclass(config_dict)

        self.assertEqual(config["strategy"], "fsdp")
        self.assertEqual(config["ppo_mini_batch_size"], 256)

        field_names = list(config)
        self.assertIn("strategy", field_names)
        self.assertIn("ppo_mini_batch_size", field_names)

        self.assertGreater(len(config), 0)


if __name__ == "__main__":
    unittest.main()

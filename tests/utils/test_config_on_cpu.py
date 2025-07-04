# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import unittest
from dataclasses import dataclass

from omegaconf import OmegaConf

from verl.utils import omega_conf_to_dataclass


@dataclass
class TestDataclass:
    hidden_size: int
    activation: str


@dataclass
class TestTrainConfig:
    batch_size: int
    model: TestDataclass


_cfg_str = """train_config:
  batch_size: 32
  model:
    hidden_size: 768
    activation: relu"""


class TestConfigOnCPU(unittest.TestCase):
    """Test cases for configuration utilities on CPU.

    Test Plan:
    1. Test basic OmegaConf to dataclass conversion for simple nested structures
    2. Test nested OmegaConf to dataclass conversion for complex hierarchical configurations
    3. Verify all configuration values are correctly converted and accessible
    4. Test recursive processing of _target_ fields
    5. Test edge cases and backward compatibility
    """

    def setUp(self):
        self.config = OmegaConf.create(_cfg_str)

    def test_omega_conf_to_dataclass(self):
        sub_cfg = self.config.train_config.model
        cfg = omega_conf_to_dataclass(sub_cfg, TestDataclass)
        self.assertEqual(cfg.hidden_size, 768)
        self.assertEqual(cfg.activation, "relu")
        assert isinstance(cfg, TestDataclass)

    def test_nested_omega_conf_to_dataclass(self):
        cfg = omega_conf_to_dataclass(self.config.train_config, TestTrainConfig)
        self.assertEqual(cfg.batch_size, 32)
        self.assertEqual(cfg.model.hidden_size, 768)
        self.assertEqual(cfg.model.activation, "relu")
        assert isinstance(cfg, TestTrainConfig)
        assert isinstance(cfg.model, TestDataclass)

    def test_recursive_omega_conf_to_dataclass_basic(self):
        """Test basic recursive processing of _target_ fields."""
        config_str = """
        algorithm:
          _target_: verl.utils.profiler.config.ProfilerConfig
          discrete: False
          all_ranks: True
        nested:
          profiler:
            _target_: verl.utils.profiler.config.ProfilerConfig
            discrete: True
            all_ranks: False
        """
        config = OmegaConf.create(config_str)

        result = omega_conf_to_dataclass(config, recursive=True)

        from verl.utils.profiler.config import ProfilerConfig

        self.assertIsInstance(result["algorithm"], ProfilerConfig)
        self.assertEqual(result["algorithm"].discrete, False)
        self.assertEqual(result["algorithm"].all_ranks, True)

        self.assertIsInstance(result["nested"]["profiler"], ProfilerConfig)
        self.assertEqual(result["nested"]["profiler"].discrete, True)
        self.assertEqual(result["nested"]["profiler"].all_ranks, False)

    def test_recursive_omega_conf_to_dataclass_mixed(self):
        """Test recursive processing with mixed _target_ and regular fields."""
        config_str = """
        regular_field: "test_value"
        number_field: 42
        algorithm:
          _target_: verl.utils.profiler.config.ProfilerConfig
          discrete: False
          all_ranks: True
        nested:
          regular_nested: "nested_value"
          profiler:
            _target_: verl.utils.profiler.config.ProfilerConfig
            discrete: True
            all_ranks: False
        """
        config = OmegaConf.create(config_str)

        result = omega_conf_to_dataclass(config, recursive=True)

        self.assertEqual(result["regular_field"], "test_value")
        self.assertEqual(result["number_field"], 42)
        self.assertEqual(result["nested"]["regular_nested"], "nested_value")

        from verl.utils.profiler.config import ProfilerConfig

        self.assertIsInstance(result["algorithm"], ProfilerConfig)
        self.assertIsInstance(result["nested"]["profiler"], ProfilerConfig)

    def test_recursive_omega_conf_to_dataclass_no_targets(self):
        """Test recursive processing with no _target_ fields."""
        config_str = """
        regular_field: "test_value"
        nested:
          another_field: 123
          deep_nested:
            value: "deep"
        """
        config = OmegaConf.create(config_str)

        result = omega_conf_to_dataclass(config, recursive=True)

        self.assertEqual(result["regular_field"], "test_value")
        self.assertEqual(result["nested"]["another_field"], 123)
        self.assertEqual(result["nested"]["deep_nested"]["value"], "deep")

    def test_recursive_backward_compatibility(self):
        """Test that recursive=False maintains backward compatibility."""
        config_str = """
        _target_: verl.utils.profiler.config.ProfilerConfig
        discrete: False
        all_ranks: True
        """
        config = OmegaConf.create(config_str)

        result = omega_conf_to_dataclass(config, recursive=False)

        from verl.utils.profiler.config import ProfilerConfig

        self.assertIsInstance(result, ProfilerConfig)
        self.assertEqual(result.discrete, False)
        self.assertEqual(result.all_ranks, True)

    def test_recursive_deep_copy(self):
        """Test that recursive processing doesn't modify the original config."""
        config_str = """
        algorithm:
          _target_: verl.utils.profiler.config.ProfilerConfig
          discrete: False
          all_ranks: True
        """
        original_config = OmegaConf.create(config_str)

        omega_conf_to_dataclass(original_config, recursive=True)

        self.assertTrue(hasattr(original_config.algorithm, "_target_") or "_target_" in original_config.algorithm)
        self.assertEqual(original_config.algorithm._target_, "verl.utils.profiler.config.ProfilerConfig")

    def test_recursive_edge_cases(self):
        """Test recursive processing with edge cases."""
        config_str = """
        string_value: "test"
        number_value: 42
        list_value: [1, 2, 3]
        none_value: null
        nested:
          empty_dict: {}
          profiler:
            _target_: verl.utils.profiler.config.ProfilerConfig
            discrete: True
            all_ranks: False
        """
        config = OmegaConf.create(config_str)

        result = omega_conf_to_dataclass(config, recursive=True)

        self.assertEqual(result["string_value"], "test")
        self.assertEqual(result["number_value"], 42)
        self.assertEqual(result["list_value"], [1, 2, 3])
        self.assertIsNone(result["none_value"])

        self.assertEqual(result["nested"]["empty_dict"], {})

        from verl.utils.profiler.config import ProfilerConfig

        self.assertIsInstance(result["nested"]["profiler"], ProfilerConfig)


if __name__ == "__main__":
    unittest.main()

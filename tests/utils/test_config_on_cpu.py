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

import torch
import torch.optim as optim
from omegaconf import OmegaConf

from verl.utils import OptimConfig, omega_conf_to_dataclass


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

    def test_optim_config_dataclass(self):
        """Test OptimConfig dataclass conversion and PyTorch optimizer creation."""
        optim_config_dict = {
            "lr": 1e-5,
            "lr_warmup_steps": 100,
            "warmup_style": "cosine",
            "weight_decay": 0.01,
            "betas": (0.9, 0.999),
            "clip_grad": 1.0,
            "lr_scheduler": "cosine",
            "warmup_steps_ratio": 0.1,
            "total_training_steps": 1000,
            "lr_warmup_steps_ratio": 0.1,
            "min_lr_ratio": 0.1,
            "num_cycles": 0.5,
        }

        omega_config = OmegaConf.create(optim_config_dict)
        optim_config = omega_conf_to_dataclass(omega_config, OptimConfig)

        self.assertEqual(optim_config.lr, 1e-5)
        self.assertEqual(optim_config.lr_warmup_steps, 100)
        self.assertEqual(optim_config.warmup_style, "cosine")
        self.assertEqual(optim_config.weight_decay, 0.01)
        assert isinstance(optim_config, OptimConfig)

        model = torch.nn.Linear(10, 1)
        optimizer = optim.AdamW(model.parameters(), lr=optim_config.lr, weight_decay=optim_config.weight_decay, betas=tuple(optim_config.betas))

        assert isinstance(optimizer, torch.optim.AdamW)
        self.assertEqual(optimizer.param_groups[0]["lr"], 1e-5)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.01)


if __name__ == "__main__":
    unittest.main()

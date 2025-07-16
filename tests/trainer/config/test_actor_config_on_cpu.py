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

    def test_actor_config_from_dict(self):
        """Test creating ActorConfig from dictionary with all required fields."""
        config_dict = {
            "_target_": "verl.trainer.config.ActorConfig",
            "strategy": "fsdp",
            "ppo_mini_batch_size": 256,
            "ppo_micro_batch_size": None,
            "ppo_micro_batch_size_per_gpu": None,
            "use_dynamic_bsz": False,
            "ppo_max_token_len_per_gpu": 16384,
            "clip_ratio": 0.2,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.2,
            "policy_loss": {
                "_target_": "verl.trainer.config.config.PolicyLossConfig",
                "loss_mode": "vanilla",
                "clip_cov_ratio": 0.0002,
                "clip_cov_lb": 1.0,
                "clip_cov_ub": 5.0,
                "kl_cov_ratio": 0.0002,
                "ppo_kl_coef": 0.1,
            },
            "clip_ratio_c": 3.0,
            "loss_agg_mode": "token-mean",
            "entropy_coeff": 0,
            "use_kl_loss": False,
            "use_torch_compile": True,
            "kl_loss_coef": 0.001,
            "kl_loss_type": "low_var_kl",
            "ppo_epochs": 1,
            "shuffle": False,
            "checkpoint": {
                "_target_": "verl.trainer.config.config.CheckpointConfig",
                "save_contents": ["model", "optimizer", "extra"],
                "load_contents": ["model", "optimizer", "extra"],
                "async_save": False,
            },
            "optim": {
                "_target_": "verl.trainer.config.config.OptimConfig",
                "lr": 1e-6,
                "lr_warmup_steps_ratio": 0.0,
                "total_training_steps": -1,
                "weight_decay": 0.01,
            },
        }

        config = omega_conf_to_dataclass(config_dict)

        self.assertIsInstance(config, ActorConfig)

        self.assertEqual(config.strategy, "fsdp")
        self.assertEqual(config.ppo_mini_batch_size, 256)
        self.assertEqual(config.clip_ratio, 0.2)
        self.assertEqual(config.ppo_epochs, 1)
        self.assertFalse(config.shuffle)
        self.assertTrue(config.use_torch_compile)
        self.assertEqual(config.kl_loss_type, "low_var_kl")

        self.assertIsNotNone(config.checkpoint)
        self.assertEqual(config.checkpoint.save_contents, ["model", "optimizer", "extra"])

        self.assertIsNotNone(config.optim)
        self.assertEqual(config.optim.lr, 1e-6)
        self.assertEqual(config.optim.weight_decay, 0.01)

        self.assertIsNotNone(config.policy_loss)
        self.assertEqual(config.policy_loss.loss_mode, "vanilla")

    def test_megatron_actor_config_from_dict(self):
        """Test creating MegatronActorConfig from dictionary."""
        config_dict = {
            "_target_": "verl.trainer.config.MegatronActorConfig",
            "strategy": "megatron",
            "ppo_mini_batch_size": 256,
            "clip_ratio": 0.2,
            "data_loader_seed": None,
            "load_weight": True,
            "megatron": {
                "_target_": "verl.trainer.config.config.MegatronConfig",
                "param_offload": False,
                "tensor_model_parallel_size": 1,
                "sequence_parallel": True,
                "seed": 42,
            },
            "profile": {
                "_target_": "verl.trainer.config.config.ProfileConfig",
                "use_profile": False,
                "profile_ranks": None,
            },
            "optim": {
                "_target_": "verl.trainer.config.config.OptimConfig",
                "optimizer": "adam",
                "clip_grad": 1.0,
                "lr_decay_style": "constant",
            },
        }

        config = omega_conf_to_dataclass(config_dict)

        self.assertIsInstance(config, MegatronActorConfig)

        self.assertEqual(config.strategy, "megatron")
        self.assertEqual(config.ppo_mini_batch_size, 256)
        self.assertEqual(config.clip_ratio, 0.2)

        self.assertIsNone(config.data_loader_seed)
        self.assertTrue(config.load_weight)

        self.assertIsNotNone(config.megatron)
        self.assertFalse(config.megatron.param_offload)
        self.assertEqual(config.megatron.tensor_model_parallel_size, 1)
        self.assertTrue(config.megatron.sequence_parallel)
        self.assertEqual(config.megatron.seed, 42)

        self.assertIsNotNone(config.profile)
        self.assertFalse(config.profile.use_profile)
        self.assertIsNone(config.profile.profile_ranks)

    def test_fsdp_actor_config_from_dict(self):
        """Test creating FSDPActorConfig from dictionary."""
        config_dict = {
            "_target_": "verl.trainer.config.FSDPActorConfig",
            "strategy": "fsdp",
            "ppo_mini_batch_size": 256,
            "clip_ratio": 0.2,
            "grad_clip": 1.0,
            "ulysses_sequence_parallel_size": 1,
            "entropy_from_logits_with_chunking": False,
            "entropy_checkpointing": False,
            "fsdp_config": {
                "_target_": "verl.trainer.config.config.FSDPConfig",
                "param_offload": False,
                "optimizer_offload": False,
                "reshard_after_forward": True,
                "fsdp_size": -1,
                "wrap_policy": {
                    "_target_": "verl.trainer.config.config.WrapPolicyConfig",
                    "min_num_params": 0,
                },
            },
            "optim": {
                "_target_": "verl.trainer.config.config.OptimConfig",
                "lr_warmup_steps": -1,
                "min_lr_ratio": 0.0,
                "num_cycles": 0.5,
                "warmup_style": "constant",
            },
        }

        config = omega_conf_to_dataclass(config_dict)

        self.assertIsInstance(config, FSDPActorConfig)

        self.assertEqual(config.strategy, "fsdp")
        self.assertEqual(config.ppo_mini_batch_size, 256)
        self.assertEqual(config.clip_ratio, 0.2)

        self.assertEqual(config.grad_clip, 1.0)
        self.assertEqual(config.ulysses_sequence_parallel_size, 1)
        self.assertFalse(config.entropy_from_logits_with_chunking)
        self.assertFalse(config.entropy_checkpointing)

        self.assertIsNotNone(config.fsdp_config)
        self.assertFalse(config.fsdp_config.param_offload)
        self.assertFalse(config.fsdp_config.optimizer_offload)
        self.assertTrue(config.fsdp_config.reshard_after_forward)
        self.assertEqual(config.fsdp_config.fsdp_size, -1)

        self.assertIsNotNone(config.fsdp_config.wrap_policy)
        self.assertEqual(config.fsdp_config.wrap_policy.min_num_params, 0)

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

    def test_config_from_yaml_via_hydra(self):
        """Test creating configs from YAML files using Hydra composition."""
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
            cfg = compose(config_name="ppo_trainer")

        actor_config = omega_conf_to_dataclass(cfg.actor_rollout_ref)

        self.assertIsInstance(actor_config, FSDPActorConfig)
        self.assertEqual(actor_config.strategy, "fsdp")

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

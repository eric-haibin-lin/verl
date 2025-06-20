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

import numpy as np
import pytest
import torch

import verl.trainer.ppo.core_algos
from verl.trainer.ppo.core_algos import (
    compute_gae_advantage_return,
    compute_grpo_outcome_advantage, 
    compute_policy_loss,
    get_adv_estimator_fn,
    register_adv_est
)


def mock_test_fn():
    pass


class TestRegisterAdvEst(unittest.TestCase):
    def setUp(self):
        """Clear the registry before each test"""
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY.clear()
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY = {
            "gae": lambda x: x * 2,
            "vtrace": lambda x: x + 1,
        }
        self.ADV_ESTIMATOR_REGISTRY = verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY

    def tearDown(self) -> None:
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY.clear()
        return super().tearDown()

    def test_register_new_function(self):
        """Test registering a new function with a string name"""

        @register_adv_est("test_estimator")
        def test_fn():
            pass

        self.assertIn("test_estimator", self.ADV_ESTIMATOR_REGISTRY)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["test_estimator"], test_fn)

    def test_register_with_enum(self):
        """Test registering with an enum value (assuming AdvantageEstimator exists)"""
        from enum import Enum

        class AdvantageEstimator(Enum):
            TEST = "test_enum_estimator"

        @register_adv_est(AdvantageEstimator.TEST)
        def test_fn():
            pass

        self.assertIn("test_enum_estimator", self.ADV_ESTIMATOR_REGISTRY)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["test_enum_estimator"], test_fn)

    def test_duplicate_registration_same_function(self):
        """Test that registering the same function twice doesn't raise an error"""
        register_adv_est("duplicate_test")(mock_test_fn)
        register_adv_est("duplicate_test")(mock_test_fn)

        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["duplicate_test"], mock_test_fn)

    def test_duplicate_registration_different_function(self):
        """Test that registering different functions with same name raises ValueError"""

        @register_adv_est("conflict_test")
        def test_fn1():
            pass

        with self.assertRaises(ValueError):

            @register_adv_est("conflict_test")
            def test_fn2():
                pass

    def test_decorator_preserves_function(self):
        """Test that the decorator returns the original function"""

        def test_fn():
            return "original"

        decorated = register_adv_est("preserve_test")(test_fn)
        self.assertEqual(decorated(), "original")

    def test_multiple_registrations(self):
        """Test registering multiple different functions"""
        init_adv_count = len(self.ADV_ESTIMATOR_REGISTRY)

        @register_adv_est("estimator1")
        def fn1():
            pass

        @register_adv_est("estimator2")
        def fn2():
            pass

        self.assertEqual(len(self.ADV_ESTIMATOR_REGISTRY), 2 + init_adv_count)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["estimator1"], fn1)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["estimator2"], fn2)

    def test_get_adv_estimator_fn_valid_names(self):
        """Test that valid names return the correct function from registry."""
        # Test GAE
        gae_fn = get_adv_estimator_fn("gae")
        assert gae_fn(5) == 10  # 5 * 2 = 10

        # Test Vtrace
        vtrace_fn = get_adv_estimator_fn("vtrace")
        assert vtrace_fn(5) == 6  # 5 + 1 = 6

    def test_get_adv_estimator_fn_invalid_name(self):
        """Test that invalid names raise ValueError."""
        with pytest.raises(ValueError) as excinfo:
            get_adv_estimator_fn("invalid_name")
        assert "Unknown advantage estimator simply: invalid_name" in str(excinfo.value)

    def test_get_adv_estimator_fn_case_sensitive(self):
        """Test that name lookup is case-sensitive."""
        with pytest.raises(ValueError):
            get_adv_estimator_fn("GAE")  # Different case


class TestComputeGAEAdvantageReturn(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.batch_size = 4
        self.response_length = 10
        self.device = torch.device("cpu")
        
    def _create_test_tensors(self, batch_size=None, response_length=None):
        """Helper to create test tensors with realistic values"""
        bs = batch_size or self.batch_size
        rl = response_length or self.response_length
        
        token_level_rewards = torch.randn(bs, rl, device=self.device) * 0.1
        values = torch.randn(bs, rl, device=self.device) * 0.5
        response_mask = torch.ones(bs, rl, device=self.device, dtype=torch.float32)
        
        for i in range(bs):
            min_length = max(2, rl//2)  # Ensure at least 2 tokens for variance calculation
            mask_length = torch.randint(min_length, rl+1, (1,)).item()
            response_mask[i, mask_length:] = 0.0
            
        gamma = torch.tensor(0.99, device=self.device)
        lam = torch.tensor(0.95, device=self.device)
        
        return token_level_rewards, values, response_mask, gamma, lam
    
    def test_basic_functionality(self):
        """Test basic GAE computation with typical inputs"""
        token_level_rewards, values, response_mask, gamma, lam = self._create_test_tensors()
        
        advantages, returns = compute_gae_advantage_return(
            token_level_rewards, values, response_mask, gamma, lam
        )
        
        self.assertEqual(advantages.shape, (self.batch_size, self.response_length))
        self.assertEqual(returns.shape, (self.batch_size, self.response_length))
        self.assertEqual(advantages.dtype, torch.float32)
        self.assertEqual(returns.dtype, torch.float32)
        
        self.assertTrue(torch.all(torch.isfinite(returns)))
        self.assertTrue(torch.all(torch.isfinite(advantages)))
        
    def test_short_response(self):
        """Test with short responses (2 tokens to avoid division by zero in whitening)"""
        token_level_rewards, values, response_mask, gamma, lam = self._create_test_tensors(
            response_length=2
        )
        
        advantages, returns = compute_gae_advantage_return(
            token_level_rewards, values, response_mask, gamma, lam
        )
        
        self.assertEqual(advantages.shape, (self.batch_size, 2))
        self.assertEqual(returns.shape, (self.batch_size, 2))
        
    def test_zero_rewards(self):
        """Test with zero rewards"""
        token_level_rewards, values, response_mask, gamma, lam = self._create_test_tensors()
        token_level_rewards.fill_(0.0)
        
        advantages, returns = compute_gae_advantage_return(
            token_level_rewards, values, response_mask, gamma, lam
        )
        
        self.assertEqual(advantages.shape, (self.batch_size, self.response_length))
        self.assertEqual(returns.shape, (self.batch_size, self.response_length))
        
    def test_different_gamma_lambda(self):
        """Test with different gamma and lambda values"""
        token_level_rewards, values, response_mask, _, _ = self._create_test_tensors()
        
        gamma = torch.tensor(1.0, device=self.device)
        lam = torch.tensor(1.0, device=self.device)
        
        advantages, returns = compute_gae_advantage_return(
            token_level_rewards, values, response_mask, gamma, lam
        )
        
        self.assertEqual(advantages.shape, (self.batch_size, self.response_length))
        self.assertEqual(returns.shape, (self.batch_size, self.response_length))
        
    def test_masked_input_handling(self):
        """Test proper handling of masked inputs"""
        token_level_rewards, values, response_mask, gamma, lam = self._create_test_tensors()
        
        response_mask[:, self.response_length//2:] = 0.0
        
        advantages, returns = compute_gae_advantage_return(
            token_level_rewards, values, response_mask, gamma, lam
        )
        
        self.assertEqual(advantages.shape, (self.batch_size, self.response_length))
        self.assertEqual(returns.shape, (self.batch_size, self.response_length))


class TestComputeGRPOOutcomeAdvantage(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.batch_size = 6
        self.response_length = 8
        self.device = torch.device("cpu")
        
    def _create_test_tensors(self, batch_size=None, response_length=None):
        """Helper to create test tensors for GRPO"""
        bs = batch_size or self.batch_size
        rl = response_length or self.response_length
        
        token_level_rewards = torch.randn(bs, rl, device=self.device) * 0.1
        response_mask = torch.ones(bs, rl, device=self.device, dtype=torch.float32)
        
        for i in range(bs):
            mask_length = torch.randint(rl//2, rl+1, (1,)).item()
            response_mask[i, mask_length:] = 0.0
            
        index = np.array([0, 0, 1, 1, 2, 2])[:bs]
        
        return token_level_rewards, response_mask, index
    
    def test_basic_functionality(self):
        """Test basic GRPO computation"""
        token_level_rewards, response_mask, index = self._create_test_tensors()
        
        advantages, returns = compute_grpo_outcome_advantage(
            token_level_rewards, response_mask, index
        )
        
        self.assertEqual(advantages.shape, (self.batch_size, self.response_length))
        self.assertEqual(returns.shape, (self.batch_size, self.response_length))
        self.assertEqual(advantages.dtype, torch.float32)
        self.assertEqual(returns.dtype, torch.float32)
        
        torch.testing.assert_close(advantages, returns, rtol=1e-5, atol=1e-6)
        
    def test_single_sample_per_index(self):
        """Test with single sample per index"""
        token_level_rewards, response_mask, _ = self._create_test_tensors()
        index = np.arange(self.batch_size)  # Each sample has unique index
        
        advantages, returns = compute_grpo_outcome_advantage(
            token_level_rewards, response_mask, index
        )
        
        self.assertEqual(advantages.shape, (self.batch_size, self.response_length))
        self.assertEqual(returns.shape, (self.batch_size, self.response_length))
        
    def test_norm_adv_by_std_flag(self):
        """Test norm_adv_by_std_in_grpo parameter"""
        token_level_rewards, response_mask, index = self._create_test_tensors()
        
        advantages_norm, returns_norm = compute_grpo_outcome_advantage(
            token_level_rewards, response_mask, index, norm_adv_by_std_in_grpo=True
        )
        
        advantages_no_norm, returns_no_norm = compute_grpo_outcome_advantage(
            token_level_rewards, response_mask, index, norm_adv_by_std_in_grpo=False
        )
        
        self.assertEqual(advantages_norm.shape, (self.batch_size, self.response_length))
        self.assertEqual(advantages_no_norm.shape, (self.batch_size, self.response_length))
        
        self.assertFalse(torch.allclose(advantages_norm, advantages_no_norm, rtol=1e-3))
        
    def test_epsilon_parameter(self):
        """Test epsilon parameter for numerical stability"""
        token_level_rewards, response_mask, index = self._create_test_tensors()
        
        advantages, returns = compute_grpo_outcome_advantage(
            token_level_rewards, response_mask, index, epsilon=1e-8
        )
        
        self.assertEqual(advantages.shape, (self.batch_size, self.response_length))
        self.assertEqual(returns.shape, (self.batch_size, self.response_length))
        
    def test_index_grouping_logic(self):
        """Test proper grouping by index"""
        token_level_rewards, response_mask, _ = self._create_test_tensors(batch_size=4)
        index = np.array([0, 0, 1, 1])  # Two groups of two samples each
        
        advantages, returns = compute_grpo_outcome_advantage(
            token_level_rewards, response_mask, index
        )
        
        self.assertEqual(advantages.shape, (4, self.response_length))
        self.assertEqual(returns.shape, (4, self.response_length))


class TestComputePolicyLoss(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.batch_size = 4
        self.response_length = 10
        self.device = torch.device("cpu")
        
    def _create_test_tensors(self, batch_size=None, response_length=None):
        """Helper to create test tensors for policy loss"""
        bs = batch_size or self.batch_size
        rl = response_length or self.response_length
        
        old_log_prob = torch.randn(bs, rl, device=self.device) * 0.1 - 2.0
        log_prob = old_log_prob + torch.randn(bs, rl, device=self.device) * 0.05
        advantages = torch.randn(bs, rl, device=self.device) * 0.5
        response_mask = torch.ones(bs, rl, device=self.device, dtype=torch.float32)
        
        for i in range(bs):
            mask_length = torch.randint(rl//2, rl+1, (1,)).item()
            response_mask[i, mask_length:] = 0.0
            
        return old_log_prob, log_prob, advantages, response_mask
    
    def test_basic_functionality(self):
        """Test basic policy loss computation"""
        old_log_prob, log_prob, advantages, response_mask = self._create_test_tensors()
        
        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
            old_log_prob, log_prob, advantages, response_mask, cliprange=0.2
        )
        
        self.assertEqual(pg_loss.shape, torch.Size([]))
        self.assertEqual(pg_clipfrac.shape, torch.Size([]))
        self.assertEqual(ppo_kl.shape, torch.Size([]))
        self.assertEqual(pg_clipfrac_lower.shape, torch.Size([]))
        
        self.assertEqual(pg_loss.dtype, torch.float32)
        self.assertEqual(pg_clipfrac.dtype, torch.float32)
        self.assertEqual(ppo_kl.dtype, torch.float32)
        self.assertEqual(pg_clipfrac_lower.dtype, torch.float32)
        
    def test_different_loss_agg_modes(self):
        """Test different loss aggregation modes"""
        old_log_prob, log_prob, advantages, response_mask = self._create_test_tensors()
        
        modes = ["token-mean", "seq-mean-token-sum", "seq-mean-token-mean"]
        
        for mode in modes:
            pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                old_log_prob, log_prob, advantages, response_mask, 
                cliprange=0.2, loss_agg_mode=mode
            )
            
            self.assertEqual(pg_loss.shape, torch.Size([]))
            self.assertEqual(pg_clipfrac.shape, torch.Size([]))
            self.assertEqual(ppo_kl.shape, torch.Size([]))
            self.assertEqual(pg_clipfrac_lower.shape, torch.Size([]))
            
    def test_dual_clip_parameters(self):
        """Test dual clipping with different cliprange parameters"""
        old_log_prob, log_prob, advantages, response_mask = self._create_test_tensors()
        
        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
            old_log_prob, log_prob, advantages, response_mask,
            cliprange_low=0.1, cliprange_high=0.3, clip_ratio_c=2.0
        )
        
        self.assertEqual(pg_loss.shape, torch.Size([]))
        self.assertEqual(pg_clipfrac.shape, torch.Size([]))
        self.assertEqual(ppo_kl.shape, torch.Size([]))
        self.assertEqual(pg_clipfrac_lower.shape, torch.Size([]))
        
    def test_extreme_advantage_values(self):
        """Test with extreme advantage values"""
        old_log_prob, log_prob, _, response_mask = self._create_test_tensors()
        
        advantages = torch.ones_like(old_log_prob) * 10.0
        
        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
            old_log_prob, log_prob, advantages, response_mask, cliprange=0.2
        )
        
        self.assertEqual(pg_loss.shape, torch.Size([]))
        
        advantages = torch.ones_like(old_log_prob) * -10.0
        
        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
            old_log_prob, log_prob, advantages, response_mask, cliprange=0.2
        )
        
        self.assertEqual(pg_loss.shape, torch.Size([]))
        
    def test_cliprange_parameter_validation(self):
        """Test cliprange parameter handling"""
        old_log_prob, log_prob, advantages, response_mask = self._create_test_tensors()
        
        for cliprange in [0.1, 0.2, 0.5]:
            pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                old_log_prob, log_prob, advantages, response_mask, cliprange=cliprange
            )
            
            self.assertEqual(pg_loss.shape, torch.Size([]))
            
    def test_clip_ratio_c_validation(self):
        """Test clip_ratio_c parameter validation"""
        old_log_prob, log_prob, advantages, response_mask = self._create_test_tensors()
        
        with self.assertRaises(AssertionError):
            compute_policy_loss(
                old_log_prob, log_prob, advantages, response_mask,
                cliprange=0.2, clip_ratio_c=0.5  # Should fail
            )


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""
Test script to verify the correctness of the GAE optimization.
This ensures the vectorized implementation produces the same results as the original.
"""

import torch
import numpy as np

def masked_whiten(values, mask, shift_mean=True):
    """Simplified version of masked_whiten for testing"""
    if shift_mean:
        mean = torch.sum(values * mask, dim=-1, keepdim=True) / torch.sum(mask, dim=-1, keepdim=True)
        values = values - mean
    
    var = torch.sum(values**2 * mask, dim=-1, keepdim=True) / torch.sum(mask, dim=-1, keepdim=True)
    std = torch.sqrt(var + 1e-8)
    return values / std


def compute_gae_advantage_return_original(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Original implementation for comparison"""
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = masked_whiten(advantages, response_mask)
    return advantages, returns

def compute_gae_advantage_return_optimized(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Optimized vectorized implementation"""
    with torch.no_grad():
        gen_len = token_level_rewards.shape[-1]
        
        next_values = torch.cat([values[:, 1:], torch.zeros_like(values[:, :1])], dim=1)
        
        deltas = token_level_rewards + gamma * next_values - values
        
        indices = torch.arange(gen_len, device=deltas.device, dtype=torch.long)
        i_indices = indices.unsqueeze(1)
        j_indices = indices.unsqueeze(0)
        
        mask = j_indices >= i_indices
        
        discount_powers = j_indices - i_indices
        discount_matrix = torch.pow(gamma * lam, discount_powers.float()) * mask.float()
        
        advantages = torch.matmul(deltas.unsqueeze(1), discount_matrix.T).squeeze(1)
        
        returns = advantages + values
        advantages = masked_whiten(advantages, response_mask)
    return advantages, returns


def test_gae_correctness():
    """Test that the optimized GAE implementation produces correct results"""
    print("Testing GAE optimization correctness...")
    
    batch_size = 4
    seq_len = 8
    gamma = 0.99
    lam = 0.95
    
    torch.manual_seed(42)
    token_level_rewards = torch.randn(batch_size, seq_len)
    values = torch.randn(batch_size, seq_len)
    response_mask = torch.ones(batch_size, seq_len)
    
    response_mask[0, 6:] = 0
    response_mask[1, 7:] = 0
    response_mask[2, 5:] = 0
    
    advantages_orig, returns_orig = compute_gae_advantage_return_original(
        token_level_rewards, values, response_mask, gamma, lam
    )
    
    advantages_opt, returns_opt = compute_gae_advantage_return_optimized(
        token_level_rewards, values, response_mask, gamma, lam
    )
    
    adv_close = torch.allclose(advantages_orig, advantages_opt, rtol=1e-5, atol=1e-6)
    ret_close = torch.allclose(returns_orig, returns_opt, rtol=1e-5, atol=1e-6)
    
    print(f"Advantages match: {adv_close}")
    print(f"Returns match: {ret_close}")
    
    if not adv_close:
        print("Advantage differences:")
        print(f"Max absolute difference: {torch.max(torch.abs(advantages_orig - advantages_opt))}")
        print(f"Mean absolute difference: {torch.mean(torch.abs(advantages_orig - advantages_opt))}")
    
    if not ret_close:
        print("Returns differences:")
        print(f"Max absolute difference: {torch.max(torch.abs(returns_orig - returns_opt))}")
        print(f"Mean absolute difference: {torch.mean(torch.abs(returns_orig - returns_opt))}")
    
    return adv_close and ret_close


def benchmark_gae_performance():
    """Benchmark the performance improvement of the optimized GAE"""
    print("\nBenchmarking GAE performance...")
    
    import time
    
    batch_size = 32
    seq_len = 512
    gamma = 0.99
    lam = 0.95
    
    torch.manual_seed(42)
    token_level_rewards = torch.randn(batch_size, seq_len)
    values = torch.randn(batch_size, seq_len)
    response_mask = torch.ones(batch_size, seq_len)
    
    for _ in range(5):
        compute_gae_advantage_return_original(token_level_rewards, values, response_mask, gamma, lam)
        compute_gae_advantage_return_optimized(token_level_rewards, values, response_mask, gamma, lam)
    
    num_runs = 100
    start_time = time.time()
    for _ in range(num_runs):
        compute_gae_advantage_return_original(token_level_rewards, values, response_mask, gamma, lam)
    original_time = time.time() - start_time
    
    start_time = time.time()
    for _ in range(num_runs):
        compute_gae_advantage_return_optimized(token_level_rewards, values, response_mask, gamma, lam)
    optimized_time = time.time() - start_time
    
    speedup = original_time / optimized_time
    print(f"Original implementation: {original_time:.4f}s ({num_runs} runs)")
    print(f"Optimized implementation: {optimized_time:.4f}s ({num_runs} runs)")
    print(f"Speedup: {speedup:.2f}x")
    
    return speedup


if __name__ == "__main__":
    correct = test_gae_correctness()
    
    if correct:
        print("✅ GAE optimization is correct!")
        speedup = benchmark_gae_performance()
        print(f"✅ Performance improvement: {speedup:.2f}x speedup")
    else:
        print("❌ GAE optimization produces incorrect results!")
        exit(1)

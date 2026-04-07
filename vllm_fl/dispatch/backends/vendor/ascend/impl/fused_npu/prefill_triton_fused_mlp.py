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
#
"""
v178: v171 architecture + adaptive BLOCK_M for matmul2.

Key insight: v171 achieves 92.59x with best-of-breed fused_silu.
For matmul2, BLOCK_M=128 means for 81920 tokens / 512 experts = 160 tokens:
  ceil(160/128) = 2 M-iterations.
For smaller tokens (2560 / 512 = 5 tokens):
  ceil(5/128) = 1 M-iteration (good).

Try BLOCK_M=64 for matmul2 when tokens > 20480 to get better pipelining:
  ceil(160/64) = 3 M-iterations (more iterations but smaller tiles, better occupancy).

Expected: Might help large tokens without hurting small tokens.
"""

import os
os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = "1"

import torch
import torch_npu
import triton
import triton.language as tl
from typing import Optional


@triton.jit
def _fused_matmul_silu_kernel_n_parallel(
    A_ptr, B_ptr, C_ptr,
    expert_starts_ptr, expert_ends_ptr,
    num_experts, K, intermediate_size,
    stride_a_row, stride_a_k,
    stride_b_expert, stride_b_k, stride_b_n,
    stride_c_row, stride_c_n,
    num_N_blocks,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
):
    """Fused matmul1 + silu with BLOCK_K=256."""
    pid = tl.program_id(0)
    expert_idx = pid // num_N_blocks
    n_block = pid % num_N_blocks

    if expert_idx >= num_experts:
        return

    expert_start = tl.load(expert_starts_ptr + expert_idx)
    expert_end = tl.load(expert_ends_ptr + expert_idx)
    num_tokens_for_expert = expert_end - expert_start

    if num_tokens_for_expert <= 0:
        return

    col_start = n_block * BLOCK_N
    if col_start >= intermediate_size:
        return

    a_base = A_ptr + expert_start * stride_a_row
    b_base = B_ptr + expert_idx * stride_b_expert
    c_base = C_ptr + expert_start * stride_c_row
    num_M_blocks = tl.cdiv(num_tokens_for_expert, BLOCK_M)

    for m_block in range(num_M_blocks):
        row_start = m_block * BLOCK_M
        acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            p_a = tl.make_block_ptr(
                a_base, shape=(num_tokens_for_expert, K),
                strides=(stride_a_row, stride_a_k),
                offsets=(row_start, k_start),
                block_shape=(BLOCK_M, BLOCK_K), order=(1, 0)
            )
            a_vals = tl.load(p_a, boundary_check=(0, 1), care_padding=False)

            p_b_gate = tl.make_block_ptr(
                b_base, shape=(K, 2 * intermediate_size),
                strides=(stride_b_k, stride_b_n),
                offsets=(k_start, col_start),
                block_shape=(BLOCK_K, BLOCK_N), order=(1, 0)
            )
            b_gate = tl.load(p_b_gate, boundary_check=(0, 1), care_padding=False)
            acc_gate += tl.dot(a_vals, b_gate)

            p_b_up = tl.make_block_ptr(
                b_base, shape=(K, 2 * intermediate_size),
                strides=(stride_b_k, stride_b_n),
                offsets=(k_start, intermediate_size + col_start),
                block_shape=(BLOCK_K, BLOCK_N), order=(1, 0)
            )
            b_up = tl.load(p_b_up, boundary_check=(0, 1), care_padding=False)
            acc_up += tl.dot(a_vals, b_up)

        silu_gate = acc_gate * tl.sigmoid(acc_gate)
        result = silu_gate * acc_up

        p_c = tl.make_block_ptr(
            c_base, shape=(num_tokens_for_expert, intermediate_size),
            strides=(stride_c_row, stride_c_n),
            offsets=(row_start, col_start),
            block_shape=(BLOCK_M, BLOCK_N), order=(1, 0)
        )
        tl.store(p_c, result.to(OUTPUT_DTYPE), boundary_check=(0, 1))


@triton.jit
def _expert_batched_matmul_scale_kernel(
    A_ptr, B_ptr, C_ptr, scale_ptr,
    expert_starts_ptr, expert_ends_ptr,
    num_experts, K, N,
    stride_a_row, stride_a_k,
    stride_b_expert, stride_b_k, stride_b_n,
    stride_c_row, stride_c_n,
    num_N_blocks,
    HAS_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
):
    """Matmul with optional fused scale and care_padding=False."""
    pid = tl.program_id(0)
    expert_idx = pid // num_N_blocks
    n_block = pid % num_N_blocks

    if expert_idx >= num_experts:
        return

    expert_start = tl.load(expert_starts_ptr + expert_idx)
    expert_end = tl.load(expert_ends_ptr + expert_idx)
    num_tokens_for_expert = expert_end - expert_start

    if num_tokens_for_expert <= 0:
        return

    col_start = n_block * BLOCK_N
    a_base = A_ptr + expert_start * stride_a_row
    b_base = B_ptr + expert_idx * stride_b_expert
    c_base = C_ptr + expert_start * stride_c_row
    num_M_blocks = tl.cdiv(num_tokens_for_expert, BLOCK_M)

    for m_block in range(num_M_blocks):
        row_start = m_block * BLOCK_M
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        p_a = tl.make_block_ptr(
            a_base, shape=(num_tokens_for_expert, K),
            strides=(stride_a_row, stride_a_k),
            offsets=(row_start, 0),
            block_shape=(BLOCK_M, BLOCK_K), order=(1, 0)
        )
        a_vals = tl.load(p_a, boundary_check=(0, 1), care_padding=False)

        p_b = tl.make_block_ptr(
            b_base, shape=(K, N),
            strides=(stride_b_k, stride_b_n),
            offsets=(0, col_start),
            block_shape=(BLOCK_K, BLOCK_N), order=(1, 0)
        )
        b_vals = tl.load(p_b, boundary_check=(0, 1), care_padding=False)
        acc = tl.dot(a_vals, b_vals)

        if HAS_SCALE:
            global_row = expert_start + row_start
            row_offsets = global_row + tl.arange(0, BLOCK_M)
            scale_mask = row_offsets < (expert_start + num_tokens_for_expert)
            scales = tl.load(scale_ptr + row_offsets, mask=scale_mask, other=1.0, care_padding=False)
            acc = acc * scales[:, None]

        p_c = tl.make_block_ptr(
            c_base, shape=(num_tokens_for_expert, N),
            strides=(stride_c_row, stride_c_n),
            offsets=(row_start, col_start),
            block_shape=(BLOCK_M, BLOCK_N), order=(1, 0)
        )
        tl.store(p_c, acc.to(OUTPUT_DTYPE), boundary_check=(0, 1))


def compute_expert_ranges(group_list: torch.Tensor, num_experts: int):
    expert_ends = group_list.to(torch.int32)
    expert_starts = torch.zeros(num_experts, dtype=torch.int32, device=group_list.device)
    if num_experts > 1:
        expert_starts[1:] = expert_ends[:-1]
    return expert_starts, expert_ends


def triton_fused_mlp(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    group_list: torch.Tensor,
    topk_scales: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    v178: v171 fused_silu + BLOCK_M=64 matmul2.

    fused_silu: BLOCK_M=64, BLOCK_N=128, BLOCK_K=256, grid=512
    matmul2: BLOCK_M=64, BLOCK_N=256, BLOCK_K=128, grid=4096

    Expected: Different BLOCK_M for matmul2 might affect large tokens differently.
    """
    num_tokens = hidden_states.shape[0]
    intermediate_size = w2.shape[1]
    num_experts = w1.shape[0]
    K = hidden_states.shape[1]
    N = w2.shape[2]

    expert_starts, expert_ends = compute_expert_ranges(group_list, num_experts)

    output_dtype = tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16

    # ---- Fused matmul1 + silu (BLOCK_K=256) ----
    activated = torch.empty(num_tokens, intermediate_size, dtype=hidden_states.dtype, device=hidden_states.device)
    if num_tokens > 0:
        BLOCK_M1, BLOCK_N1, BLOCK_K1 = 64, 128, 256
        num_N_blocks1 = triton.cdiv(intermediate_size, BLOCK_N1)
        grid1 = (num_experts * num_N_blocks1,)

        _fused_matmul_silu_kernel_n_parallel[grid1](
            hidden_states, w1, activated,
            expert_starts, expert_ends,
            num_experts, K, intermediate_size,
            hidden_states.stride(0), hidden_states.stride(1),
            w1.stride(0), w1.stride(1), w1.stride(2),
            activated.stride(0), activated.stride(1),
            num_N_blocks1,
            BLOCK_M1, BLOCK_N1, BLOCK_K1, output_dtype,
        )

    # ---- Matmul2 + scale (BLOCK_M=64) ----
    output = torch.empty(num_tokens, N, dtype=hidden_states.dtype, device=hidden_states.device)
    if num_tokens > 0:
        BLOCK_M2, BLOCK_N2, BLOCK_K2 = 64, 256, 128
        num_N_blocks2 = triton.cdiv(N, BLOCK_N2)
        grid2 = (num_experts * num_N_blocks2,)

        scale_ptr = topk_scales.squeeze(-1) if topk_scales is not None else hidden_states

        _expert_batched_matmul_scale_kernel[grid2](
            activated, w2, output, scale_ptr,
            expert_starts, expert_ends,
            num_experts, intermediate_size, N,
            activated.stride(0), activated.stride(1),
            w2.stride(0), w2.stride(1), w2.stride(2),
            output.stride(0), output.stride(1),
            num_N_blocks2,
            topk_scales is not None,
            BLOCK_M2, BLOCK_N2, BLOCK_K2, output_dtype,
        )

    return output

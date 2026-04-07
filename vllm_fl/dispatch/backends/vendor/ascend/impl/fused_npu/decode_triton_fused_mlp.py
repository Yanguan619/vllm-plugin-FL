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
Triton implementation of fused MLP with grouped matmul and SwiGLU activation.

v56 Optimization: Based on v55, try larger BLOCK_N_OUT
- Same inline binary search as v55
- BLOCK_N_OUT=256 (vs v55's 128) - 8 output blocks instead of 16
- BLOCK_K1=256, BLOCK_INTER=128
- Expected: fewer blocks might reduce overhead
"""

import torch
import torch_npu
import triton
import triton.language as tl
from typing import Optional


@triton.jit
def _fused_mlp_kernel_with_search(
    hidden_ptr, w1_ptr, w2_ptr, output_ptr, group_list_ptr,
    num_tokens, hidden_size, inter_size, out_hidden_size, num_experts,
    stride_h_t, stride_h_d,
    stride_w1_e, stride_w1_h, stride_w1_i,
    stride_w2_e, stride_w2_i, stride_w2_h,
    stride_o_t, stride_o_d,
    BLOCK_N_OUT: tl.constexpr,
    BLOCK_K1: tl.constexpr,
    BLOCK_INTER: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
):
    """Fully fused MLP kernel with inline expert lookup."""
    pid_token = tl.program_id(0)
    pid_n = tl.program_id(1)

    if pid_token >= num_tokens:
        return

    # Binary search for expert_idx
    lo = 0
    hi = num_experts
    while lo < hi:
        mid = (lo + hi) // 2
        mid_val = tl.load(group_list_ptr + mid)
        if mid_val <= pid_token:
            lo = mid + 1
        else:
            hi = mid
    expert_idx = lo

    n_out_start = pid_n * BLOCK_N_OUT

    acc_out = tl.zeros((1, BLOCK_N_OUT), dtype=tl.float32)

    for inter_start in range(0, inter_size, BLOCK_INTER):
        acc_gate = tl.zeros((1, BLOCK_INTER), dtype=tl.float32)
        acc_up = tl.zeros((1, BLOCK_INTER), dtype=tl.float32)

        for k1_start in range(0, hidden_size, BLOCK_K1):
            h_ptr = tl.make_block_ptr(
                hidden_ptr,
                shape=(num_tokens, hidden_size),
                strides=(stride_h_t, stride_h_d),
                offsets=(pid_token, k1_start),
                block_shape=(1, BLOCK_K1),
                order=(1, 0)
            )
            h_vals = tl.load(h_ptr, boundary_check=(0, 1))

            w_gate_ptr = tl.make_block_ptr(
                w1_ptr + expert_idx * stride_w1_e,
                shape=(hidden_size, inter_size),
                strides=(stride_w1_h, stride_w1_i),
                offsets=(k1_start, inter_start),
                block_shape=(BLOCK_K1, BLOCK_INTER),
                order=(1, 0)
            )
            w_gate = tl.load(w_gate_ptr, boundary_check=(0, 1))

            w_up_ptr = tl.make_block_ptr(
                w1_ptr + expert_idx * stride_w1_e,
                shape=(hidden_size, inter_size * 2),
                strides=(stride_w1_h, stride_w1_i),
                offsets=(k1_start, inter_size + inter_start),
                block_shape=(BLOCK_K1, BLOCK_INTER),
                order=(1, 0)
            )
            w_up = tl.load(w_up_ptr, boundary_check=(0, 1))

            acc_gate += tl.dot(h_vals, w_gate)
            acc_up += tl.dot(h_vals, w_up)

        silu_gate = acc_gate * tl.sigmoid(acc_gate)
        activated = silu_gate * acc_up

        w2_ptr_block = tl.make_block_ptr(
            w2_ptr + expert_idx * stride_w2_e,
            shape=(inter_size, out_hidden_size),
            strides=(stride_w2_i, stride_w2_h),
            offsets=(inter_start, n_out_start),
            block_shape=(BLOCK_INTER, BLOCK_N_OUT),
            order=(1, 0)
        )
        w2_vals = tl.load(w2_ptr_block, boundary_check=(0, 1))

        acc_out += tl.dot(activated.to(w2_vals.dtype), w2_vals)

    out_ptr = tl.make_block_ptr(
        output_ptr,
        shape=(num_tokens, out_hidden_size),
        strides=(stride_o_t, stride_o_d),
        offsets=(pid_token, n_out_start),
        block_shape=(1, BLOCK_N_OUT),
        order=(1, 0)
    )
    tl.store(out_ptr, acc_out.to(OUTPUT_DTYPE), boundary_check=(0, 1))


@triton.jit
def _scale_kernel(
    input_ptr, scale_ptr, output_ptr,
    num_tokens, hidden_size,
    stride_input_row, stride_output_row,
    BLOCK_SIZE: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
):
    """Element-wise scaling kernel."""
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)
    col_start = col_block_idx * BLOCK_SIZE

    p_in = tl.make_block_ptr(
        input_ptr, shape=(num_tokens, hidden_size),
        strides=(stride_input_row, 1),
        offsets=(row_idx, col_start),
        block_shape=(1, BLOCK_SIZE), order=(1, 0)
    )

    values = tl.load(p_in, boundary_check=(0, 1))
    scale = tl.load(scale_ptr + row_idx)
    result = values.to(tl.float32) * scale.to(tl.float32)

    p_out = tl.make_block_ptr(
        output_ptr, shape=(num_tokens, hidden_size),
        strides=(stride_output_row, 1),
        offsets=(row_idx, col_start),
        block_shape=(1, BLOCK_SIZE), order=(1, 0)
    )
    tl.store(p_out, result.to(OUTPUT_DTYPE), boundary_check=(0, 1))


def triton_scale(input_tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply element-wise scaling."""
    num_tokens, hidden_size = input_tensor.shape
    output = torch.empty_like(input_tensor)
    if num_tokens == 0:
        return output

    BLOCK_SIZE = 256
    num_col_blocks = triton.cdiv(hidden_size, BLOCK_SIZE)
    grid = (num_tokens, num_col_blocks)
    output_dtype = tl.bfloat16 if input_tensor.dtype == torch.bfloat16 else tl.float16

    _scale_kernel[grid](
        input_tensor, scale.squeeze(-1), output,
        num_tokens, hidden_size,
        input_tensor.stride(0), output.stride(0),
        BLOCK_SIZE, output_dtype,
    )
    return output


def triton_fused_mlp(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    group_list: torch.Tensor,
    topk_scales: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Triton implementation of MLP with grouped matmul and SwiGLU activation.

    v56 Optimization: BLOCK_N_OUT=256 with inline binary search
    """
    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    inter_size = w2.shape[1]
    num_experts = w1.shape[0]

    if num_tokens == 0:
        return torch.zeros(num_tokens, hidden_size, dtype=hidden_states.dtype, device=hidden_states.device)

    group_list_int32 = group_list.to(dtype=torch.int32, device=hidden_states.device)

    output = torch.zeros(num_tokens, hidden_size, dtype=hidden_states.dtype, device=hidden_states.device)

    output_dtype = tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16

    BLOCK_N_OUT = 256
    BLOCK_K1 = 256
    BLOCK_INTER = 128

    grid = (num_tokens, triton.cdiv(hidden_size, BLOCK_N_OUT))

    _fused_mlp_kernel_with_search[grid](
        hidden_states, w1, w2, output, group_list_int32,
        num_tokens, hidden_size, inter_size, hidden_size, num_experts,
        hidden_states.stride(0), hidden_states.stride(1),
        w1.stride(0), w1.stride(1), w1.stride(2),
        w2.stride(0), w2.stride(1), w2.stride(2),
        output.stride(0), output.stride(1),
        BLOCK_N_OUT, BLOCK_K1, BLOCK_INTER, output_dtype,
    )

    if topk_scales is not None:
        output = triton_scale(output, topk_scales)

    return output

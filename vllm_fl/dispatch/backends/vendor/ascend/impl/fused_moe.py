# Copyright (c) 2025 BAAI. All rights reserved.
# Adapted from https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/model_executor/layers/fused_moe/layer.py
# Below is the original copyright:
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional, Union

import torch
import torch.nn.functional as F
import torch_npu
from flag_gems.runtime.backend._ascend import fused
from vllm.model_executor.layers.fused_moe import FusedMoE
from .fused_npu import unquant_apply_mlp

def _expert_ids_to_group_list(expert_ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    """
    Convert expert_ids to group_list for npu_grouped_matmul.

    expert_ids: [total_tokens] each value is expert_id
    group_list: [num_experts] cumulative token count per expert
    """
    # Vectorized counting instead of loop
    token_counts = torch.bincount(expert_ids, minlength=num_experts)
    group_list = token_counts.cumsum(dim=0)
    return group_list

class AscendOps:
    ### activation
    @staticmethod
    def silu_and_mul(x):
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        return F.silu(x1) * x2

    @staticmethod
    def gelu_and_mul(x, approximate="none"):
        return torch_npu.npu_gelu_mul(x, approximate=approximate)

    ### moe
    @staticmethod
    def topk_softmax(topk_weights, topk_indices, token_expert_indices, gating_output, renormalize=False,):
        fused.topk_softmax(
            topk_weights,
            topk_indices,
            token_expert_indices,
            gating_output,
            renormalize,
        )
        return topk_weights, topk_indices

    @staticmethod
    def topk_softmax_torch(
        topk_weights: torch.Tensor,
        topk_indices: torch.Tensor,
        token_expert_indices: torch.Tensor,
        gating_output: torch.Tensor,
        renormalize: bool,
    ) -> tuple[torch.Tensor, ...]:
        # Use pure PyTorch implementation to avoid FlagGems Triton kernel
        # issues on Ascend NPU.
        topk = topk_weights.size(1)

        scores = torch.softmax(gating_output.float(), dim=-1)
        topk_weights_out, topk_indices_out = torch.topk(
            scores, k=topk, dim=-1, sorted=False
        )
        if renormalize:
            topk_weights_out = topk_weights_out / topk_weights_out.sum(
                dim=-1, keepdim=True
            )
        topk_weights.copy_(topk_weights_out)
        topk_indices.copy_(topk_indices_out.to(topk_indices.dtype))
        return topk_weights, topk_indices

    @staticmethod
    def fused_experts_impl(
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        inplace: bool = False,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        use_fp8_w8a8: bool = False,
        use_int8_w8a8: bool = False,
        use_int8_w8a16: bool = False,
        use_int4_w4a16: bool = False,
        per_channel_quant: bool = False,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        w1_scale: Optional[torch.Tensor] = None,
        w2_scale: Optional[torch.Tensor] = None,
        w1_zp: Optional[torch.Tensor] = None,
        w2_zp: Optional[torch.Tensor] = None,
        a1_scale: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None,
        block_shape: Optional[list[int]] = None,
        w1_bias: Optional[torch.Tensor] = None,
        w2_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Check constraints.
        if use_int4_w4a16:
            assert hidden_states.size(1) // 2 == w1.size(2), "Hidden size mismatch"
        else:
            assert hidden_states.size(1) == w1.size(2), (
                f"Hidden size mismatch {hidden_states.size(1)} != {w1.size(2)}"
            )

        assert topk_weights.size() == topk_ids.size(), "topk shape mismatch"
        assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
        assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
        assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
        assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]

        num_tokens = hidden_states.size(0)
        E, N, _ = w1.size()
        K = w2.size(1)
        if global_num_experts == -1:
            global_num_experts = E
        top_k_num = topk_ids.size(1)
        # We execute the fused_moe kernel in chunks to circumvent this issue:
        # https://github.com/vllm-project/vllm/issues/5938
        CHUNK_SIZE = envs.VLLM_FUSED_MOE_CHUNK_SIZE
        M = min(num_tokens, CHUNK_SIZE)

        config_dtype = _get_config_dtype_str(
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            ocp_mx_scheme=None,  ## dont support mxfp4
            dtype=hidden_states.dtype,
        )

        # Note: for use_int8_w8a16 or use_int4_w4a16, the activations are
        # quantized prior to calling fused_experts.
        quant_dtype = _get_config_quant_dtype(
            use_fp8_w8a8=use_fp8_w8a8, use_int8_w8a8=use_int8_w8a8, ocp_mx_scheme=None
        )

        get_config_func = functools.partial(
            try_get_optimal_moe_config,
            w1.size(),
            w2.size(),
            top_k_num,
            config_dtype,
            block_shape=block_shape,
        )

        config = get_config_func(M)

        # We can reuse the memory between these because by the time we need
        # cache3, we're done with cache1
        cache13 = torch.empty(
            M * top_k_num * max(N, K),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        intermediate_cache1 = cache13[: M * top_k_num * N].view(M, top_k_num, N)
        intermediate_cache3 = cache13[: M * top_k_num * K].view(M, top_k_num, K)

        # This needs separate memory since it's used concurrently with cache1
        intermediate_cache2 = torch.empty(
            (M * top_k_num, N // 2), device=hidden_states.device, dtype=hidden_states.dtype
        )

        if hidden_states.dtype == torch.bfloat16:
            compute_type = tl.bfloat16
        elif hidden_states.dtype == torch.float16:
            compute_type = tl.float16
        elif hidden_states.dtype == torch.float32:
            compute_type = tl.float32
        else:
            raise ValueError(f"Unsupported compute_type: {hidden_states.dtype}")

        if inplace:
            out_hidden_states = hidden_states
        else:
            out_hidden_states = torch.empty_like(hidden_states)

        for chunk in range((num_tokens // CHUNK_SIZE) + 1):
            begin_chunk_idx, end_chunk_idx = (
                chunk * CHUNK_SIZE,
                min((chunk + 1) * CHUNK_SIZE, num_tokens),
            )
            curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
            tokens_in_chunk, _ = curr_hidden_states.size()

            if tokens_in_chunk == 0:
                break

            if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
                # Adjust the intermediate cache size and config for the last
                # chunk. Note that in most cases we only have one chunk
                # so the cache size and config are already set correctly and
                # do not need to be adjusted.
                intermediate_cache1 = intermediate_cache1[:tokens_in_chunk]
                intermediate_cache2 = intermediate_cache2[
                    : tokens_in_chunk * topk_ids.size(1)
                ]
                intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]
                config = get_config_func(tokens_in_chunk)

            curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
            curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

            qcurr_hidden_states, a1q_scale = moe_kernel_quantize_input(
                A=curr_hidden_states,
                A_scale=a1_scale,
                quant_dtype=quant_dtype,
                per_act_token_quant=per_channel_quant,
                block_shape=block_shape,
            )

            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                curr_topk_ids,
                config["BLOCK_SIZE_M"],
                global_num_experts,
                expert_map,
                ignore_invalid_experts=True,
            )

            # Use unquant_apply_mlp instead of manual npu_grouped_matmul calls
            # Step 1: Reorder hidden_states according to sorted_token_ids
            sorted_hidden = qcurr_hidden_states[sorted_token_ids]

            # Step 2: Convert expert_ids to group_list
            group_list = _expert_ids_to_group_list(expert_ids, global_num_experts)

            # Step 3: Apply MLP using unified function (gate+up -> activation -> down)
            # Note: unquant_apply_mlp handles transposition internally when need_trans=True
            output = unquant_apply_mlp(
                hidden_states=sorted_hidden,
                w1=w1,
                w2=w2,
                group_list=group_list,
                group_list_type=1,
                topk_scales=curr_topk_weights if apply_router_weight_on_input else None,
                need_trans=True,
            )

            # moe_sum: gather results back to original token order
            # Need to reshape output to match expected format for moe_sum
            num_output_tokens = sorted_token_ids.shape[0]
            output_reshaped = output.view(num_output_tokens, -1)
            fl_ops.moe_sum(
                output_reshaped,
                out_hidden_states[begin_chunk_idx:end_chunk_idx],
            )

        return out_hidden_states


    @staticmethod
    def moe_sum(input, output):
        fused.moe_sum(input, output)

    @staticmethod
    def moe_align_block_size(topk_ids, num_experts, block_size, sorted_ids,
                             expert_ids, num_tokens_post_pad,):
        fused.moe_align_block_size_triton(topk_ids, num_experts, block_size, sorted_ids,
                             expert_ids, num_tokens_post_pad,)



def _torch_fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor|None = None,
) -> torch.Tensor:
    """Pure PyTorch implementation of fused MoE experts for NPU.

    This avoids the Triton fused_moe_kernel which has compatibility issues
    on Ascend NPU hardware.
    """
    num_tokens, hidden_dim = hidden_states.size()
    E, N, _ = w1.size()  # w1: [E, N, K_in]
    K = w2.size(1)        # w2: [E, K_out, N//2]
    top_k = topk_ids.size(1)

    if global_num_experts == -1:
        global_num_experts = E

    if inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.zeros_like(hidden_states)

    # Map global expert ids to local expert ids
    if expert_map is not None:
        local_topk_ids = expert_map[topk_ids.long()]
    else:
        local_topk_ids = topk_ids.long()

    # Process each expert
    for expert_idx in range(E):
        # Find which (token, k) pairs are assigned to this expert
        mask = (local_topk_ids == expert_idx)  # [num_tokens, top_k]
        if not mask.any():
            continue

        # Get token indices and their k-slot indices
        token_indices, k_indices = torch.where(mask)

        # Gather the hidden states for these tokens
        expert_input = hidden_states[token_indices]  # [n, hidden_dim]

        # Apply router weight on input if needed
        if apply_router_weight_on_input:
            weights = topk_weights[token_indices, k_indices].unsqueeze(-1)
            expert_input = expert_input * weights.to(expert_input.dtype)

        # First matmul: expert_input @ w1[expert_idx].T
        # w1[expert_idx] shape: [N, hidden_dim], result: [n, N]
        gate_up = torch.mm(expert_input, w1[expert_idx].t())

        # Activation (pure PyTorch to avoid Triton kernel issues on NPU)
        if activation == "silu":
            d = gate_up.shape[-1] // 2
            gate_up = F.silu(gate_up[..., :d]) * gate_up[..., d:]
        elif activation == "gelu":
            gate_up = torch_npu.npu_gelu_mul(gate_up)
        elif activation == "silu_no_mul":
            gate_up = F.silu(gate_up)
        elif activation == "gelu_no_mul":
            gate_up = torch_npu.npu_gelu(gate_up)
        else:
            raise ValueError(f"Unsupported FusedMoe activation: {activation}.")

        # Second matmul: activated @ w2[expert_idx].T
        # w2[expert_idx] shape: [K_out, N//2], result: [n, K_out]
        expert_output = torch.mm(gate_up, w2[expert_idx].t())

        # Apply router weight on output if not applied on input
        if not apply_router_weight_on_input:
            weights = topk_weights[token_indices, k_indices].unsqueeze(-1)
            expert_output = expert_output * weights.to(expert_output.dtype)

        # Accumulate results
        out_hidden_states.index_add_(0, token_indices, expert_output)

    return out_hidden_states


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Check constraints.
    if use_int4_w4a16:
        assert hidden_states.size(1) // 2 == w1.size(2), "Hidden size mismatch"
    else:
        assert hidden_states.size(1) == w1.size(2), (
            f"Hidden size mismatch {hidden_states.size(1)} != {w1.size(2)}"
        )

    assert topk_weights.size() == topk_ids.size(), "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
    assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]

    # Use pure-torch implementation on NPU to avoid Triton kernel
    # compatibility issues with the Ascend backend.
    return _torch_fused_experts_impl(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=inplace,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
    )



class AscendFusedMoE(FusedMoE):
    def forward_oot(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        og_hidden_states = hidden_states.shape[-1]
        if self.hidden_size != og_hidden_states:
            hidden_states = F.pad(
                hidden_states,
                (0, self.hidden_size - og_hidden_states),
                mode="constant",
                value=0.0,
            )

        def reduce_output(states: torch.Tensor) -> torch.Tensor:
            if (
                not self.is_sequence_parallel
                and not self.use_dp_chunking
                and self.reduce_results
                and (self.tp_size > 1 or self.ep_size > 1)
            ):
                states = self.maybe_all_reduce_tensor_model_parallel(states)
            return states

        if self.shared_experts is None:
            fused_output = torch.ops.vllm.moe_forward(
                hidden_states, router_logits, self.layer_name
            )
            if self.zero_expert_num is not None and self.zero_expert_num > 0:
                assert isinstance(fused_output, tuple)
                fused_output, zero_expert_result = fused_output
                return (reduce_output(fused_output) + zero_expert_result)[
                    ..., :og_hidden_states
                ]
            else:
                return reduce_output(fused_output)[..., :og_hidden_states]
        else:
            shared_output, fused_output = torch.ops.vllm.moe_forward_shared(
                hidden_states, router_logits, self.layer_name
            )
            return (
                reduce_output(shared_output)[..., :og_hidden_states],
                reduce_output(fused_output)[..., :og_hidden_states],
            )

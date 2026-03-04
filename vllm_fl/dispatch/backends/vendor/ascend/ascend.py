# Copyright (c) 2026 BAAI. All rights reserved.

"""
Ascend backend implementation.

This backend provides operator implementations for Huawei Ascend NPUs.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import torch
from vllm.attention.backends.utils import PAD_SLOT_ID

from vllm_fl.dispatch.backends.base import Backend


class AscendBackend(Backend):
    """
    Ascend backend for operator implementations.

    This backend uses Ascend CANN libraries to provide high-performance
    operator implementations for Huawei Ascend NPUs.
    """

    _available: Optional[bool] = None

    def __init__(self) -> None:
        super().__init__()

    @property
    def name(self) -> str:
        return "ascend"

    @property
    def vendor(self) -> Optional[str]:
        return "ascend"

    def is_available(self) -> bool:
        """Check if Ascend hardware and libraries are available."""
        if AscendBackend._available is None:
            # Check if NPU device is available
            if torch.npu.is_available() and torch.npu.device_count() > 0:
                AscendBackend._available = True
            else:
                AscendBackend._available = False
        return AscendBackend._available

    # ==================== Operator Implementations ====================

    def quick_gelu(self, x: torch.tensor) -> torch.Tensor:
        from .impl.activation import quick_gelu_ascend

        return quick_gelu_ascend(x)

    def silu_and_mul(self, obj, x: torch.Tensor) -> torch.Tensor:
        """
        SiLU activation followed by element-wise multiplication.

        Args:
            obj: The calling obj (for interface consistency)
            x: Input tensor of shape [..., 2*d]

        Returns:
            Output tensor of shape [..., d]
        """
        from .impl.activation import silu_and_mul_ascend

        return silu_and_mul_ascend(obj, x)

    def rms_norm(
        self,
        obj,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        RMS normalization.

        Args:
            obj: The calling obj (e.g., RMSNorm layer)
            x: Input tensor
            residual: Optional residual tensor

        Returns:
            Normalized tensor, or tuple of (normalized, residual) if residual is provided
        """
        from .impl.normalization import rms_norm_ascend

        return rms_norm_ascend(obj, x, residual)

    def gemma_rms_norm_ascend(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor],
        weight: torch.Tensor,
        epsilon: float,
    ):
        from .impl.normalization import gemma_rms_norm_ascend

        x, _ = gemma_rms_norm_ascend(x, residual, 1.0 + weight, epsilon)
        return x

    def rotary_embedding(
        self,
        obj,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_interleaved: bool = False,
        inplace: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embedding.

        Args:
            obj: The calling obj (for interface consistency)
            query: Query tensor
            key: Key tensor
            cos: Cosine cache
            sin: Sine cache
            position_ids: Position indices
            rotary_interleaved: Whether to use interleaved rotary
            inplace: Whether to modify tensors in-place

        Returns:
            Tuple of (embedded_query, embedded_key)
        """
        from .impl.rotary import rotary_embedding_ascend

        return rotary_embedding_ascend(
            obj,
            query,
            key,
            cos,
            sin,
            position_ids,
            rotary_interleaved=rotary_interleaved,
            inplace=inplace,
        )

    def attention_backend(self, use_mla: bool = False) -> str:
        """
        Get the attention backend class path for Ascend NPU.

        This method returns the native Ascend attention backend that uses
        torch_npu operators (npu_fused_infer_attention_score, etc.)
        instead of flag_gems operators.

        Uses vllm_fl's native Ascend implementation which directly calls
        torch_npu operators without depending on vllm-ascend package.

        Args:
            use_mla: Whether to use Multi-head Latent Attention (MLA)

        Returns:
            Fully qualified class path string
        """
        if use_mla:
            return "vllm_fl.dispatch.backends.vendor.ascend.impl.attention.AscendMLABackend"
        return "vllm_fl.dispatch.backends.vendor.ascend.impl.attention.AscendAttentionBackend"

    def chunk_gated_delta_rule_fwd(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.LongTensor | None = None,
    ):
        from .impl.fla.chunk import chunk_gated_delta_rule_fwd

        return chunk_gated_delta_rule_fwd(
            q,
            k,
            v,
            g,
            beta,
            scale,
            initial_state,
            output_final_state,
            cu_seqlens,
        )

    def fused_recurrent_gated_delta_rule_fwd(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        inplace_final_state: bool = True,
        cu_seqlens: torch.LongTensor | None = None,
        ssm_state_indices: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from .impl.fla.fused_recurrent import fused_recurrent_gated_delta_rule_fwd

        return fused_recurrent_gated_delta_rule_fwd(
            q,
            k,
            v,
            g,
            beta,
            scale,
            initial_state,
            inplace_final_state,
            cu_seqlens,
            ssm_state_indices,
            num_accepted_tokens,
            use_qk_l2norm_in_kernel,
        )

    def causal_conv1d_fn(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        activation: Optional[str] = "silu",
        conv_states: Optional[torch.Tensor] = None,
        has_initial_state: Optional[torch.Tensor] = None,
        cache_indices: Optional[torch.Tensor] = None,
        query_start_loc: Optional[torch.Tensor] = None,
        metadata: Optional[Any] = None,
        pad_slot_id: int = PAD_SLOT_ID,
    ):
        from .impl.causal_conv1d import causal_conv1d_fn

        return causal_conv1d_fn(
            x,
            weight,
            bias,
            activation,
            conv_states,
            has_initial_state,
            cache_indices,
            query_start_loc,
            metadata,
            pad_slot_id,
        )

    def causal_conv1d_update(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        activation: bool | str | None = None,
        conv_state_indices: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
        max_query_len: int = -1,
        pad_slot_id: int = PAD_SLOT_ID,
        block_idx_last_scheduled_token: torch.Tensor | None = None,
        initial_state_idx: torch.Tensor | None = None,
        validate_data=False,
    ):
        from .impl.causal_conv1d import causal_conv1d_update_npu

        return causal_conv1d_update_npu(
            x,
            conv_state,
            weight,
            bias,
            activation,
            conv_state_indices,
            num_accepted_tokens,
            query_start_loc,
            max_query_len,
            pad_slot_id,
            block_idx_last_scheduled_token,
            initial_state_idx,
            validate_data,
        )

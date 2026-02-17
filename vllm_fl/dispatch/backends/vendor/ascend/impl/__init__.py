# Copyright (c) 2026 BAAI. All rights reserved.

"""
Ascend operator implementations.
https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/torchnpuCustomsapi/docs/context/torch_npu_list.md
"""

from .activation import quick_gelu_ascend, silu_and_mul_ascend
from .attention import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendAttentionState,
    AscendMetadata,
    AscendMLABackend,
    is_torch_npu_available,
)
from .attention_mask import AttentionMaskBuilder, get_attention_mask_builder
from .fused_moe.fused_moe import AscendFusedMoE
from .normalization import gemma_rms_norm_ascend, rms_norm_ascend
from .rotary import rotary_embedding_ascend

__all__ = [
    "quick_gelu_ascend",
    "silu_and_mul_ascend",
    "rms_norm_ascend",
    "gemma_rms_norm_ascend",
    "rotary_embedding_ascend",
    "AscendAttentionBackend",
    "AscendAttentionBackendImpl",
    "AscendAttentionMetadataBuilder",
    "AscendMetadata",
    "AscendAttentionState",
    "AscendMLABackend",
    "AscendFusedMoE",
    "is_torch_npu_available",
    "AttentionMaskBuilder",
    "get_attention_mask_builder",
]

# Copyright (c) 2026 BAAI. All rights reserved.
from vllm_fl.dispatch.backends.vendor.ascend.impl.fla.chunk import (
    chunk_gated_delta_rule as ascend_chunk_gated_delta_rule,
)

__all__ = ["ascend_chunk_gated_delta_rule"]

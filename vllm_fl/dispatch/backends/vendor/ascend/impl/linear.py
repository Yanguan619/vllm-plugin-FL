# Copyright (c) 2026 BAAI. All rights reserved.

"""
Ascend activation operator implementations.
"""

from __future__ import annotations

import torch


def linear_ascend(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
) -> torch.Tensor:
    import torch_npu

    return torch_npu.npu_linear(x, weight, bias=bias)

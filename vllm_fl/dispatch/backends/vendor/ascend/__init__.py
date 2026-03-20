# Copyright (c) 2026 BAAI. All rights reserved.

"""
Ascend (Huawei) backend for vllm-plugin-FL dispatch.
"""
import vllm

from .ascend import AscendBackend
from .patches.platform.patch_mamba_config import verify_and_update_config

vllm.model_executor.models.config.HybridAttentionMambaModelConfig.verify_and_update_config = verify_and_update_config

__all__ = ["AscendBackend"]

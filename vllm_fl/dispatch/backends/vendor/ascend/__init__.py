# Copyright (c) 2026 BAAI. All rights reserved.

"""
Ascend (Huawei) backend for vllm-plugin-FL dispatch.
"""

from .ascend import AscendBackend
from .patch import patch_empty_cache, patch_mamba_config, patch_sampler

patch_mamba_config()
patch_empty_cache()
patch_sampler()

__all__ = ["AscendBackend"]

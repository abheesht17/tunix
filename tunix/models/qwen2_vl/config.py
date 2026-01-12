# Copyright 2025 Google LLC
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

"""Configuration classes for Qwen2-VL vision-language models."""

import dataclasses
from typing import Tuple

from tunix.models.qwen2.model import ShardingConfig


@dataclasses.dataclass(slots=True)
class VisionConfig:
  """Configuration for Qwen2-VL vision encoder."""

  depth: int  # Number of transformer layers
  hidden_size: int  # Vision encoder embedding dimension
  out_hidden_size: int  # Output projection size (matches text model)
  num_heads: int  # Number of attention heads
  in_chans: int  # Input channels (3 for RGB)
  patch_size: int  # Patch size for image tokenization
  spatial_patch_size: int  # Spatial patch size
  temporal_patch_size: int  # Temporal patch size (for video)
  spatial_merge_size: int  # Spatial merge factor
  intermediate_size: int  # MLP intermediate dimension
  hidden_act: str  # Activation function ("silu" or "gelu")
  norm_eps: float  # Layer norm epsilon

  # Window attention parameters
  window_size: int  # Attention window size
  fullatt_block_indexes: Tuple[int, ...]  # Layers with full attention

  # Video-specific
  tokens_per_second: int  # Tokens per second for video

  @classmethod
  def qwen2p5_vl_7b(cls):
    """Config for Qwen2.5-VL-7B."""
    return cls(
        depth=32,
        hidden_size=1280,
        out_hidden_size=3584,
        num_heads=16,
        in_chans=3,
        patch_size=14,
        spatial_patch_size=14,
        temporal_patch_size=2,
        spatial_merge_size=2,
        intermediate_size=3420,
        hidden_act="silu",
        norm_eps=1e-6,
        window_size=112,
        fullatt_block_indexes=(7, 15, 23, 31),
        tokens_per_second=2,
    )

  @classmethod
  def qwen2p5_vl_3b(cls):
    """Config for Qwen2.5-VL-3B."""
    # TODO: Fill in actual 3B config values from HuggingFace
    return cls(
        depth=32,
        hidden_size=1280,
        out_hidden_size=2048,  # Approximate
        num_heads=16,
        in_chans=3,
        patch_size=14,
        spatial_patch_size=14,
        temporal_patch_size=2,
        spatial_merge_size=2,
        intermediate_size=3420,
        hidden_act="silu",
        norm_eps=1e-6,
        window_size=112,
        fullatt_block_indexes=(7, 15, 23, 31),
        tokens_per_second=2,
    )

  @property
  def head_dim(self) -> int:
    """Compute head dimension."""
    return self.hidden_size // self.num_heads


@dataclasses.dataclass(slots=True)
class Qwen2VLConfig:
  """Combined configuration for Qwen2-VL vision-language model."""

  vision_config: VisionConfig
  text_config: 'ModelConfig'  # From qwen2.model
  shd_config: ShardingConfig

  # Vision-language specific
  min_pixels: int = 256 * 28 * 28  # Minimum pixels per image
  max_pixels: int = 1280 * 28 * 28  # Maximum pixels per image
  freeze_vision_tower: bool = False  # Whether to freeze vision encoder

  @classmethod
  def qwen2p5_vl_7b(cls, shd_config: ShardingConfig):
    """Create config for Qwen2.5-VL-7B."""
    from tunix.models.qwen2.model import ModelConfig

    return cls(
        vision_config=VisionConfig.qwen2p5_vl_7b(),
        text_config=ModelConfig.qwen2p5_7b(),
        shd_config=shd_config,
    )

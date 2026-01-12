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

"""Vision-language connector for Qwen2-VL models."""

from flax import nnx
import jax
from jax import numpy as jnp
import jaxtyping

from tunix.models.qwen2_vl.config import VisionConfig


class PatchMerger(nnx.Module):
  """Merge and project vision features to language model dimension."""

  def __init__(
      self,
      vision_hidden_size: int,
      text_hidden_size: int,
      *,
      rngs: nnx.Rngs,
  ):
    """
    Args:
      vision_hidden_size: Vision encoder output dimension
      text_hidden_size: Text model hidden dimension
    """
    self.vision_hidden_size = vision_hidden_size
    self.text_hidden_size = text_hidden_size

    # Layer normalization
    self.ln_q = nnx.LayerNorm(
        num_features=vision_hidden_size,
        epsilon=1e-6,
        rngs=rngs,
    )

    # Projection layers with GELU activation
    self.mlp = [
        nnx.Linear(
            in_features=vision_hidden_size,
            out_features=text_hidden_size,
            use_bias=True,
            rngs=rngs,
        ),
        nnx.gelu,
        nnx.Linear(
            in_features=text_hidden_size,
            out_features=text_hidden_size,
            use_bias=True,
            rngs=rngs,
        ),
    ]

  @jax.named_scope('patch_merger')
  def __call__(self, x: jaxtyping.Array) -> jaxtyping.Array:
    """
    Args:
      x: Vision features [B, num_patches, vision_hidden_size]

    Returns:
      Projected features [B, num_patches, text_hidden_size]
    """
    # Layer norm
    x = self.ln_q(x)

    # Apply MLP
    for layer in self.mlp:
      if callable(layer):
        x = layer(x)

    return x


class VisionLanguageConnector(nnx.Module):
  """Connector to bridge vision encoder and language model."""

  def __init__(
      self,
      vision_config: VisionConfig,
      text_hidden_size: int,
      *,
      rngs: nnx.Rngs,
  ):
    """
    Args:
      vision_config: Vision encoder configuration
      text_hidden_size: Text model hidden dimension
      rngs: Random number generators
    """
    self.vision_config = vision_config

    # Projection from vision to text dimension
    self.merger = PatchMerger(
        vision_hidden_size=vision_config.hidden_size,
        text_hidden_size=text_hidden_size,
        rngs=rngs,
    )

  @jax.named_scope('vision_language_connector')
  def __call__(self, vision_features: jaxtyping.Array) -> jaxtyping.Array:
    """Project vision features to text embedding space.

    Args:
      vision_features: Vision encoder outputs [B, num_patches, vision_hidden_size]

    Returns:
      Projected vision embeddings [B, num_patches, text_hidden_size]
    """
    return self.merger(vision_features)

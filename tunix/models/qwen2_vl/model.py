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

"""Qwen2.5-VL vision-language model."""

from typing import Optional, Tuple

from flax import nnx
import jax
from jax import numpy as jnp
import jaxtyping

from tunix.generate.mappings import BackendMappingMixin
from tunix.models.qwen2.model import Qwen2, Cache
from tunix.models.qwen2_vl.config import Qwen2VLConfig, VisionConfig
from tunix.models.qwen2_vl.connector import VisionLanguageConnector
from tunix.models.qwen2_vl.vision_encoder import Qwen2VisionTransformerPretrainedModel


# Special token for image placeholder
IMAGE_TOKEN_ID = 151655  # <|image_pad|> token ID in Qwen2-VL tokenizer


class Qwen2VLForConditionalGeneration(BackendMappingMixin, nnx.Module):
  """Qwen2.5-VL model for conditional generation with vision and language."""

  def __init__(
      self,
      config: Qwen2VLConfig,
      *,
      rngs: nnx.Rngs,
  ):
    """
    Args:
      config: Qwen2VL configuration
      rngs: Random number generators
    """
    self.config = config

    # Vision encoder
    self.visual = Qwen2VisionTransformerPretrainedModel(
        config.vision_config,
        rngs=rngs,
    )

    # Vision-language connector
    self.connector = VisionLanguageConnector(
        vision_config=config.vision_config,
        text_hidden_size=config.text_config.embed_dim,
        rngs=rngs,
    )

    # Language model
    self.language_model = Qwen2(
        config=config.text_config,
        rngs=rngs,
        shd_config=config.shd_config,
    )

    # Whether to freeze vision tower
    self.freeze_vision_tower = config.freeze_vision_tower

  def encode_images(
      self,
      pixel_values: jaxtyping.Array,
      grid_thw: Optional[jaxtyping.Array] = None,
  ) -> jaxtyping.Array:
    """Encode images to vision features.

    Args:
      pixel_values: [B, T, H, W, C] pixel values
      grid_thw: [B, 3] grid dimensions (temporal, height, width)

    Returns:
      vision_embeddings: [B, num_patches, text_hidden_size]
    """
    # Encode with vision transformer
    vision_features = self.visual(pixel_values, grid_thw)

    # Project to text embedding space
    vision_embeddings = self.connector(vision_features)

    # Optionally stop gradient if vision tower is frozen
    if self.freeze_vision_tower:
      vision_embeddings = jax.lax.stop_gradient(vision_embeddings)

    return vision_embeddings

  def merge_vision_text_embeddings(
      self,
      input_ids: jaxtyping.Array,
      vision_embeddings: Optional[jaxtyping.Array],
  ) -> jaxtyping.Array:
    """Merge vision and text embeddings at image token positions.

    Args:
      input_ids: [B, L] token IDs (may contain IMAGE_TOKEN_ID)
      vision_embeddings: [B, num_patches, D] vision features or None

    Returns:
      merged_embeddings: [B, L', D] where L' includes vision tokens
    """
    # Get text embeddings
    text_embeddings = self.language_model.embedder.encode(input_ids)

    # If no vision embeddings, return text embeddings as is
    if vision_embeddings is None:
      return text_embeddings

    # Find image token positions
    # For each image token, we'll replace it with the vision embeddings
    # This is a simplified version - full implementation would handle
    # multiple images and proper token replacement

    batch_size, seq_len, embed_dim = text_embeddings.shape

    # Create mask for image tokens
    image_mask = (input_ids == IMAGE_TOKEN_ID)  # [B, L]

    # For now, simple approach: if there are image tokens, replace the first
    # occurrence with vision embeddings
    # TODO: Implement proper multi-image handling and token replacement

    # Check if there are any image tokens
    has_images = jnp.any(image_mask)

    def replace_with_vision():
      """Replace image tokens with vision embeddings."""
      # Find first image token position for each sample in batch
      # This is simplified - proper implementation would handle all images
      first_image_pos = jnp.argmax(image_mask.astype(jnp.int32), axis=1)  # [B]

      # For simplicity, we'll concatenate vision embeddings at the image position
      # In reality, we need to properly splice them in
      # This is a placeholder for the actual implementation
      return text_embeddings

    # Conditional replacement
    merged = jax.lax.cond(
        has_images,
        replace_with_vision,
        lambda: text_embeddings,
    )

    return merged

  def __call__(
      self,
      input_ids: jaxtyping.Array,  # [B, L]
      positions: jaxtyping.Array,  # [B, L]
      cache: Cache | None,
      attention_mask: jaxtyping.Array,  # [B, L, L']
      pixel_values: Optional[jaxtyping.Array] = None,  # [B, T, H, W, C]
      image_grid_thw: Optional[jaxtyping.Array] = None,  # [B, 3]
      output_hidden_states: bool = False,
  ) -> Tuple[jaxtyping.Array, Cache | None]:
    """Forward pass for Qwen2.5-VL.

    Args:
      input_ids: Input token IDs
      positions: Position indices
      cache: KV cache
      attention_mask: Attention mask
      pixel_values: Optional image pixel values
      image_grid_thw: Optional image grid dimensions
      output_hidden_states: Whether to output hidden states

    Returns:
      logits: [B, L, vocab_size]
      new_cache: Updated KV cache
    """
    # Encode images if provided
    vision_embeddings = None
    if pixel_values is not None:
      vision_embeddings = self.encode_images(pixel_values, image_grid_thw)

    # For now, we'll use a simplified approach where we pass input_ids directly
    # TODO: Implement proper vision-text embedding merging
    # merged_embeddings = self.merge_vision_text_embeddings(input_ids, vision_embeddings)

    # Forward through language model
    # Note: The proper implementation would pass merged_embeddings as inputs_embeds
    # For now, we'll just use the text model as-is
    logits, new_cache = self.language_model(
        input_tokens=input_ids,
        positions=positions,
        cache=cache,
        attention_mask=attention_mask,
        output_hidden_states=output_hidden_states,
    )

    return logits, new_cache

  def get_model_input(self):
    """Returns a dummy model input for the vision-language model."""
    dummy_batch_size = 2
    dummy_seq_len = 1

    # Get base text input
    text_input = self.language_model.get_model_input()

    # Add vision inputs
    text_input['pixel_values'] = None
    text_input['image_grid_thw'] = None

    return text_input

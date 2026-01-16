# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gemma3 implementation of the vision encoders.

Based on google-deepmind/gemma multimodal/vision.py
Converted from Flax linen to Flax nnx to match Tunix codebase.
"""

from __future__ import annotations

import dataclasses
import functools

import einops
import flax
from flax import nnx
import jax
from jax import numpy as jnp
import jaxtyping
from tunix.models.gemma3 import image as image_lib
from tunix.models.gemma3 import vision_utils

# Vision token constants
BEGIN_IMAGE_TOKEN = 255999
END_IMAGE_TOKEN = 262144
NEW_LINE_TOKEN = 108
TOKEN_PLACEHOLDER = -2
NUM_PLACEHOLDER_TOKENS_PER_IMAGE = 256
NUM_TOKENS_PER_MEDIA = NUM_PLACEHOLDER_TOKENS_PER_IMAGE + 4


def check_mask(
    input_data: jaxtyping.Float[jaxtyping.Array, "L"]
) -> tuple[jaxtyping.Bool[jaxtyping.Array, ""], jaxtyping.Int[jaxtyping.Array, "L"]]:
  """Checks that the mask contains the correct number of blocks.

  Args:
    input_data: The input data to check.

  Returns:
    A boolean indicating whether the mask contains the correct number of blocks
    and their starting positions (filled with 0 for jit compatibility).
  """
  is_mask = input_data == TOKEN_PLACEHOLDER
  start_idx = (
      jnp.where(
          jnp.logical_and(is_mask[:-1] != is_mask[1:], ~is_mask[:-1]),
          size=is_mask.shape[0],
          fill_value=-1,
      )[0]
      + 1
  )
  end_idx = (
      jnp.where(
          jnp.logical_and(is_mask[:-1] != is_mask[1:], is_mask[:-1]),
          size=is_mask.shape[0],
          fill_value=NUM_PLACEHOLDER_TOKENS_PER_IMAGE - 1,
      )[0]
      + 1
  )
  all_blocks = end_idx - start_idx
  is_valid = jnp.all(all_blocks == NUM_PLACEHOLDER_TOKENS_PER_IMAGE)
  is_valid = jnp.logical_and(~is_mask[0], is_valid)
  is_valid = jnp.logical_and(~is_mask[-1], is_valid)
  return is_valid, start_idx


def check_special_vision_token(
    input_data: jaxtyping.Float[jaxtyping.Array, "B L"],
    *,
    start_positions: jaxtyping.Int[jaxtyping.Array, "B L"],
    special_token: int,
    position_offset: int,
) -> jaxtyping.Bool[jaxtyping.Array, ""]:
  """Checks that the input data contains the correct special vision tokens.

  Args:
    input_data: The input data to check.
    start_positions: The starting positions of the blocks (filled with 0 for jit
      compatibility).
    special_token: The mask token.
    position_offset: The position offset.

  Returns:
    A boolean indicating whether the input data contains the correct special
    vision tokens.
  """
  dummy_data = jnp.copy(input_data)
  # Fix the zero filled array
  dummy_data = dummy_data.at[jnp.arange(len(dummy_data)), position_offset].set(
      special_token
  )
  return jnp.all(
      dummy_data[jnp.arange(len(input_data)), start_positions + position_offset]
      == special_token
  )


@flax.struct.dataclass
class VisionInitEmbeddings:
  """Container for vision encoder output."""

  patches: jaxtyping.Float[jaxtyping.Array, "B N P D"] | None
  token_buffer: jaxtyping.Int[jaxtyping.Array, "B NEW_BUFFER"]
  num_input_tokens: jaxtyping.Int[jaxtyping.Array, "B"]


def initialize_vision_tokens(
    patches: jaxtyping.Float[jaxtyping.Array, "B N P D"] | None,
    token_buffer: jaxtyping.Int[jaxtyping.Array, "B BUFFER"],
    num_input_tokens: jaxtyping.Int[jaxtyping.Array, "B"],
) -> VisionInitEmbeddings:
  """Initializes vision embeddings.

  Vision data initialization wrapper for sampling.

  Example (text only inference):
    Input:
    {
        "patches": None,
        "token_buffer": [[255999, 108, 262144, 108, ...]],
        "num_input_tokens": [100]
    }
    Output = Input

  Example (images):
    Input:
    {
        "patches": [
            [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
            [[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]]],
        ],
        "token_buffer": [[255999, 108, 262144, 108, ...]],
        "num_input_tokens": [100]
    }
    Output = {
        "patches": [
            [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
            [[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]]],
        ],
        "token_buffer": [
            [255999, DoubleNewLine, BeginImageToken, -2, -2, ..., -2,
            EndImageToken, DoubleNewLine,
            108, 262144, 108, ...]
        ],
        "num_input_tokens": [360]
    }
  where -2 is the image token placeholder.

  NOTE: all images are inserted at the beginning of the token buffer and not
  interleaved with text tokens.

  Args:
    patches: patched images of shape BxNxPxD.
    token_buffer: The token buffer to initialize.
    num_input_tokens: The number of input tokens.

  Returns:
    The vision embeddings, the token buffer, and the number of input tokens.
  """
  if patches is not None:
    # First create an array filled with the vision token placeholder [-2,...,-2]
    base_mm_tokens = jnp.full(
        shape=(token_buffer.shape[0], NUM_TOKENS_PER_MEDIA),
        fill_value=TOKEN_PLACEHOLDER,
        dtype=jnp.int32,
    )
    # Then edit the ends with double new line, begin image and end image tokens
    base_mm_tokens = base_mm_tokens.at[:, 0].set(NEW_LINE_TOKEN)
    base_mm_tokens = base_mm_tokens.at[:, 1].set(BEGIN_IMAGE_TOKEN)
    base_mm_tokens = base_mm_tokens.at[:, -2].set(END_IMAGE_TOKEN)
    base_mm_tokens = base_mm_tokens.at[:, -1].set(NEW_LINE_TOKEN)
    # Then we repeat this tensor for each image in the batch
    mm_tokens = jnp.repeat(base_mm_tokens, patches.shape[1], axis=0)
    # Insert images after the first token, which is BOS as expected in gemma v3
    token_buffer = jnp.concatenate(
        [token_buffer[:, :1], mm_tokens, token_buffer[:, 1:]], axis=1
    )
    num_input_tokens += NUM_TOKENS_PER_MEDIA * patches.shape[1]
  return VisionInitEmbeddings(patches, token_buffer, num_input_tokens)


class VisionExit(nnx.Module):
  """The vision exit layer.

  Possibly downsample the soft tokens to a required output length.
  """

  def __init__(self, output_length: int = 256):
    """Initialize vision exit layer.

    Args:
      output_length: The embed will be spatially avg-pooled to this output
        length.
    """
    self.output_length = output_length

  def __call__(
      self, x: jaxtyping.Float[jaxtyping.Array, "B INPUT_LENGTH D"]
  ) -> jaxtyping.Float[jaxtyping.Array, "B OUTPUT_LENGTH D"]:
    """Apply spatial pooling to downsample tokens.

    Args:
      x: Input tokens.

    Returns:
      Downsampled tokens.
    """
    cur_length = x.shape[1]
    if cur_length == self.output_length:
      return x
    cur_width = int(cur_length**0.5)
    assert cur_width**2 == cur_length
    output_width = int(self.output_length**0.5)
    assert (
        output_width**2 == self.output_length
    ), f"Cannot pool {x.shape=} to {self.output_length}!"
    x = einops.rearrange(x, "b (h w) d -> b h w d", h=cur_width, w=cur_width)
    assert not cur_width % output_width, f"{cur_width=} {output_width=}"
    window = cur_width // output_width
    window_shape = (window, window)
    # Use JAX pooling instead of Flax linen's avg_pool
    x = jax.lax.reduce_window(
        x,
        init_value=0.0,
        computation=jax.lax.add,
        window_dimensions=(1, window, window, 1),
        window_strides=(1, window, window, 1),
        padding="VALID",
    )
    x = x / (window * window)  # Average pooling
    return einops.rearrange(x, "b h w d -> b (h w) d")


class SigLiPFromPatches(nnx.Module):
  """SigLIP vision encoder forward pass from PatchifiedMedia."""

  def __init__(
      self,
      *,
      siglip_encoder: vision_utils.ViTModel | None = None,
      num_mm_tokens_per_image_prepool: int = 4096,
      num_mm_tokens_per_image: int = 256,
      image_height: int = 896,
      image_width: int = 896,
      image_channels: int = 3,
      apply_stop_gradient: bool = True,
      rngs: nnx.Rngs | None = None,
  ):
    """Initialize SigLIP encoder.

    Args:
      siglip_encoder: ViT encoder model.
      num_mm_tokens_per_image_prepool: Number of tokens before pooling.
      num_mm_tokens_per_image: Number of tokens after pooling.
      image_height: Expected image height.
      image_width: Expected image width.
      image_channels: Number of image channels.
      apply_stop_gradient: Whether to stop gradients on vision encoder outputs.
      rngs: Random number generators.
    """
    self.num_mm_tokens_per_image_prepool = num_mm_tokens_per_image_prepool
    self.num_mm_tokens_per_image = num_mm_tokens_per_image
    self.image_height = image_height
    self.image_width = image_width
    self.image_channels = image_channels
    self.apply_stop_gradient = apply_stop_gradient

    # Initialize vision encoder
    if siglip_encoder is None:
      siglip_encoder = vision_utils.ViTModel(rngs=rngs)
    self.siglip_encoder = siglip_encoder

    # Initialize vision exit layer
    self.siglip_exit = VisionExit(output_length=num_mm_tokens_per_image)

  def __call__(
      self,
      *,
      patches: jaxtyping.Float[jaxtyping.Array, "B N P D"],
      is_training: bool,
      rngs: nnx.Rngs | None = None,
  ) -> jaxtyping.Float[jaxtyping.Array, "B N siglip_embed_dim"]:
    """Encode image patches to soft tokens.

    Args:
      patches: Image patches.
      is_training: Whether in training mode.
      rngs: Random number generators (needed for ViT position embedding init).

    Returns:
      Soft tokens from vision encoder.
    """
    batch_size, num_frames, num_patches, num_channels = patches.shape
    num_patches_one_side = self.image_height // self.siglip_encoder.patch_size[0]

    assert num_channels == 3 * self.siglip_encoder.patch_size[0] ** 2
    assert num_patches == num_patches_one_side**2

    # Reshape patches to images
    flattened_images = einops.rearrange(
        patches,
        "b n (h w) c -> (b n) h w c",
        h=num_patches_one_side,
        w=num_patches_one_side,
        c=num_channels,
    )
    flattened_images = einops.rearrange(
        flattened_images,
        "b h w (p q c) -> b (h p) (w q) c",
        h=num_patches_one_side,
        w=num_patches_one_side,
        p=self.siglip_encoder.patch_size[0],
        q=self.siglip_encoder.patch_size[0],
        c=3,
    )

    # Encode through ViT
    soft_tokens = self.siglip_encoder(
        flattened_images, train=is_training, rngs=rngs
    )

    # Spatial pooling if needed
    if self.num_mm_tokens_per_image_prepool != self.num_mm_tokens_per_image:
      soft_tokens = self.siglip_exit(soft_tokens)
      assert soft_tokens.shape[-2] == self.siglip_exit.output_length

    # Reshape back to batch
    soft_tokens = einops.rearrange(
        soft_tokens, "(b n) ... -> b n ...", b=batch_size, n=num_frames
    )

    if self.apply_stop_gradient:
      soft_tokens = jax.lax.stop_gradient(soft_tokens)

    return soft_tokens

  def patchify_images(
      self, images: jaxtyping.Float[jaxtyping.Array, "*B H W C"]
  ) -> jaxtyping.Float[jaxtyping.Array, "*B P D"]:
    """Patchify images.

    Args:
      images: The images to patchify.

    Returns:
      The patches of the images of shape (*batch, num_patches, patch_size *
      patch_size * channels)
    """
    *batch_dims, _, _, _ = images.shape
    images = einops.rearrange(images, "... h w c -> (...) h w c")

    preprocess_fn = functools.partial(
        image_lib.pre_process_image,
        image_height=self.image_height,
        image_width=self.image_width,
    )
    images = jax.vmap(preprocess_fn)(images)

    patches = image_lib.patchify_images(
        images,
        patch_size=self.siglip_encoder.patch_size[0],
    )
    patches = patches.reshape((*batch_dims,) + patches.shape[1:])
    return patches

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

"""Utility functions for vision encoders.

Based on google-deepmind/gemma multimodal/vision_utils.py
Converted from Flax linen to Flax nnx to match Tunix codebase.
"""

from __future__ import annotations
from collections.abc import Sequence

from flax import nnx
import jax
from jax import numpy as jnp
import jaxtyping
import numpy as np


def _posemb_sincos_2d(
    h: int,
    w: int,
    *,
    width: int,
    temperature: float = 10_000.0,
    dtype: jnp.dtype = jnp.float32,
) -> jaxtyping.Float[jaxtyping.Array, "1 M D"]:
  """Follows the MoCo v3 logic."""
  y, x = jnp.mgrid[:h, :w]

  assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
  omega = jnp.arange(width // 4) / (width // 4 - 1)
  omega = 1.0 / (temperature**omega)
  y = jnp.einsum("m,d->md", y.flatten(), omega)
  x = jnp.einsum("m,d->md", x.flatten(), omega)
  pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
  return jnp.asarray(pe, dtype)[None, :, :]


class MlpBlock(nnx.Module):
  """Transformer MLP / feed-forward block."""

  def __init__(
      self,
      input_dim: int,
      *,
      mlp_dim: int | None = None,
      dropout: float = 0.0,
      dtype_mm: jnp.dtype = jnp.float32,
      rngs: nnx.Rngs,
  ):
    """Initialize MLP block.

    Args:
      input_dim: Input dimension.
      mlp_dim: Hidden dimension in the MLP (defaults to 4x input dim).
      dropout: Dropout rate.
      dtype_mm: Data type for matrix multiplication.
      rngs: Random number generators.
    """
    self.mlp_dim = mlp_dim or 4 * input_dim
    self.dropout_rate = dropout
    self.dtype_mm = dtype_mm

    # Initialize layers
    self.dense1 = nnx.Linear(
        in_features=input_dim,
        out_features=self.mlp_dim,
        use_bias=True,
        dtype=dtype_mm,
        kernel_init=nnx.initializers.xavier_uniform(),
        bias_init=nnx.initializers.normal(stddev=1e-6),
        rngs=rngs,
    )
    self.dense2 = nnx.Linear(
        in_features=self.mlp_dim,
        out_features=input_dim,
        use_bias=True,
        dtype=dtype_mm,
        kernel_init=nnx.initializers.xavier_uniform(),
        bias_init=nnx.initializers.normal(stddev=1e-6),
        rngs=rngs,
    )
    self.dropout = nnx.Dropout(rate=dropout, rngs=rngs)

  def __call__(
      self, x: jax.Array, deterministic: bool = True
  ) -> jax.Array:
    """Apply MLP block."""
    x = self.dense1(x)
    x = nnx.gelu(x)
    x = self.dropout(x, deterministic=deterministic)
    x = self.dense2(x)
    return x


class Encoder1DBlock(nnx.Module):
  """Single transformer encoder block (MHSA + MLP)."""

  def __init__(
      self,
      embed_dim: int,
      *,
      num_heads: int = 12,
      mlp_dim: int | None = None,
      dropout: float = 0.0,
      dtype_mm: jnp.dtype = jnp.float32,
      rngs: nnx.Rngs,
  ):
    """Initialize encoder block.

    Args:
      embed_dim: Embedding dimension.
      num_heads: Number of attention heads.
      mlp_dim: Hidden dimension in the MLP (defaults to 4x embed_dim).
      dropout: Dropout rate.
      dtype_mm: Data type for matrix multiplication.
      rngs: Random number generators.
    """
    self.num_heads = num_heads
    self.dropout_rate = dropout
    self.dtype_mm = dtype_mm

    # Layer norm before attention
    self.ln1 = nnx.LayerNorm(num_features=embed_dim, rngs=rngs)

    # Multi-head attention
    self.attn = nnx.MultiHeadAttention(
        num_heads=num_heads,
        in_features=embed_dim,
        qkv_features=embed_dim,
        out_features=embed_dim,
        dtype=dtype_mm,
        kernel_init=nnx.initializers.xavier_uniform(),
        rngs=rngs,
    )
    self.dropout1 = nnx.Dropout(rate=dropout, rngs=rngs)

    # Layer norm before MLP
    self.ln2 = nnx.LayerNorm(num_features=embed_dim, rngs=rngs)

    # MLP
    self.mlp = MlpBlock(
        input_dim=embed_dim,
        mlp_dim=mlp_dim,
        dropout=dropout,
        dtype_mm=dtype_mm,
        rngs=rngs,
    )
    self.dropout2 = nnx.Dropout(rate=dropout, rngs=rngs)

  def __call__(
      self, x: jax.Array, deterministic: bool = True
  ) -> jax.Array:
    """Apply encoder block."""
    # Attention block
    y = self.ln1(x)
    y = self.attn(y)
    y = self.dropout1(y, deterministic=deterministic)
    x = x + y

    # MLP block
    y = self.ln2(x)
    y = self.mlp(y, deterministic=deterministic)
    y = self.dropout2(y, deterministic=deterministic)
    x = x + y

    return x


class Encoder(nnx.Module):
  """Transformer Model Encoder for sequence to sequence translation."""

  def __init__(
      self,
      embed_dim: int,
      *,
      depth: int,
      num_heads: int = 12,
      mlp_dim: int | None = None,
      dropout: float = 0.0,
      dtype_mm: jnp.dtype = jnp.float32,
      rngs: nnx.Rngs,
  ):
    """Initialize encoder.

    Args:
      embed_dim: Embedding dimension.
      depth: Number of encoder layers.
      num_heads: Number of attention heads.
      mlp_dim: Hidden dimension in the MLP (defaults to 4x embed_dim).
      dropout: Dropout rate.
      dtype_mm: Data type for matrix multiplication.
      rngs: Random number generators.
    """
    self.depth = depth

    # Create encoder blocks
    self.blocks = [
        Encoder1DBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            dtype_mm=dtype_mm,
            rngs=rngs,
        )
        for _ in range(depth)
    ]

    # Final layer norm
    self.encoder_norm = nnx.LayerNorm(num_features=embed_dim, rngs=rngs)

  def __call__(
      self, x: jax.Array, deterministic: bool = True
  ) -> jax.Array:
    """Apply encoder."""
    # Apply all encoder blocks
    for block in self.blocks:
      x = block(x, deterministic=deterministic)

    # Final normalization
    x = self.encoder_norm(x)
    return x


class ViTModel(nnx.Module):
  """ViT model.

  Attributes:
    patch_size: The size to patchify images.
    width: The model dimension of the vision encoder.
    depth: The number of the layers.
    mlp_dim: The hidden dimension in the ffw layers.
    num_heads: The number of the heads.
    posemb: The position embedding type.
    dropout: The dropout rate.
    dtype_mm: The dtype to convert the input to.
  """

  def __init__(
      self,
      *,
      patch_size: Sequence[int] = (14, 14),
      width: int = 1152,
      depth: int = 27,
      mlp_dim: int | None = 4304,
      num_heads: int = 16,
      posemb: str = "learn",  # Can also be "sincos2d"
      dropout: float = 0.0,
      dtype_mm: jnp.dtype = jnp.float32,
      rngs: nnx.Rngs,
  ):
    """Initialize ViT model.

    Args:
      patch_size: Size of image patches (height, width).
      width: Model dimension.
      depth: Number of transformer layers.
      mlp_dim: Hidden dimension in MLP blocks.
      num_heads: Number of attention heads.
      posemb: Position embedding type ("learn" or "sincos2d").
      dropout: Dropout rate.
      dtype_mm: Data type for matrix multiplication.
      rngs: Random number generators.
    """
    self.patch_size = patch_size
    self.width = width
    self.depth = depth
    self.mlp_dim = mlp_dim
    self.num_heads = num_heads
    self.posemb_type = posemb
    self.dropout_rate = dropout
    self.dtype_mm = dtype_mm

    # Patch embedding layer (convolution)
    self.embedding = nnx.Conv(
        in_features=3,
        out_features=width,
        kernel_size=patch_size,
        strides=patch_size,
        padding="VALID",
        dtype=dtype_mm,
        rngs=rngs,
    )

    # Position embeddings will be initialized on first call
    self.pos_embedding = None
    self._pos_embedding_initialized = False

    # Dropout
    self.dropout = nnx.Dropout(rate=dropout, rngs=rngs)

    # Transformer encoder
    self.transformer = Encoder(
        embed_dim=width,
        depth=depth,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        dropout=dropout,
        dtype_mm=dtype_mm,
        rngs=rngs,
    )

  def _get_posemb(
      self,
      seqshape: tuple[int, int],
      width: int,
      dtype: jnp.dtype,
      rngs: nnx.Rngs | None = None,
  ) -> jaxtyping.Float[jaxtyping.Array, "1 M D"]:
    """Returns the position embedding."""
    if self.posemb_type == "learn":
      if not self._pos_embedding_initialized:
        # Initialize learned position embeddings
        shape = (1, np.prod(seqshape), width)
        self.pos_embedding = nnx.Param(
            nnx.initializers.normal(stddev=1 / np.sqrt(width))(
                rngs.params(), shape, dtype
            )
        )
        self._pos_embedding_initialized = True
      return self.pos_embedding.value
    elif self.posemb_type == "sincos2d":
      return _posemb_sincos_2d(*seqshape, width=width, dtype=dtype)
    else:
      raise ValueError(f"Unknown posemb type: {self.posemb_type}")

  def __call__(
      self,
      image: jaxtyping.Float[jaxtyping.Array, "B H W C"],
      *,
      train: bool = False,
      rngs: nnx.Rngs | None = None,
  ) -> jaxtyping.Float[jaxtyping.Array, "B L D"]:
    """Apply ViT model.

    Args:
      image: Input images.
      train: Whether in training mode.
      rngs: Random number generators (needed for position embedding init).

    Returns:
      Encoded image tokens.
    """
    image = jnp.asarray(image, self.dtype_mm)

    # Patch extraction
    x = self.embedding(image)

    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])

    # Add positional embeddings
    posemb = self._get_posemb(
        seqshape=(h, w),
        width=c,
        dtype=x.dtype,
        rngs=rngs,
    )
    x = x + posemb

    # Dropout
    x = self.dropout(x, deterministic=not train)

    # Transformer encoder
    x = self.transformer(x, deterministic=not train)

    return x

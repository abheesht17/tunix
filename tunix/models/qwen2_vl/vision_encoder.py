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

"""Vision encoder for Qwen2-VL models."""

import math
from typing import Optional, Tuple

from flax import nnx
import jax
from jax import numpy as jnp
import jaxtyping

from tunix.models.qwen2_vl.config import VisionConfig


class PatchEmbed(nnx.Module):
  """Convert images/videos into patch embeddings using 3D convolution."""

  def __init__(
      self,
      config: VisionConfig,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.patch_size = config.patch_size
    self.temporal_patch_size = config.temporal_patch_size
    self.in_chans = config.in_chans
    self.embed_dim = config.hidden_size

    # 3D convolution for patch embedding
    # kernel: [temporal_patch_size, patch_size, patch_size]
    # For images, temporal dimension is 1
    self.proj = nnx.Conv(
        in_features=self.in_chans,
        out_features=self.embed_dim,
        kernel_size=(self.temporal_patch_size, self.patch_size, self.patch_size),
        strides=(self.temporal_patch_size, self.patch_size, self.patch_size),
        padding='VALID',
        use_bias=False,
        rngs=rngs,
    )

  @jax.named_scope('patch_embed')
  def __call__(self, pixel_values: jaxtyping.Array) -> jaxtyping.Array:
    """
    Args:
      pixel_values: [B, T, H, W, C] for video or [B, 1, H, W, C] for images
        where T=temporal frames, H=height, W=width, C=channels

    Returns:
      embeddings: [B, T', H', W', D] where D=embed_dim and
        T'=T//temporal_patch_size, H'=H//patch_size, W'=W//patch_size
    """
    # Apply 3D convolution to create patches
    embeddings = self.proj(pixel_values)
    return embeddings


class VisionRotaryEmbedding(nnx.Module):
  """Rotary position embeddings for vision with separate temporal/spatial dims."""

  def __init__(self, dim: int, theta: float = 10000.0):
    """
    Args:
      dim: Dimension per head
      theta: Base for frequency computation
    """
    self.dim = dim
    self.theta = theta

  def __call__(
      self,
      seqlen: int,
  ) -> Tuple[jaxtyping.Array, jaxtyping.Array]:
    """Generate sin/cos for rotary embeddings.

    Args:
      seqlen: Sequence length

    Returns:
      sin, cos: Arrays of shape [seqlen, dim // 2]
    """
    # Compute frequencies
    fraction = jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim
    freqs = 1.0 / (self.theta ** fraction)

    # Generate position indices
    positions = jnp.arange(seqlen, dtype=jnp.float32)

    # Compute sinusoids
    sinusoid_inp = jnp.outer(positions, freqs)
    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)

    return sin, cos


def apply_rotary_embedding_vision(
    q: jaxtyping.Array,
    k: jaxtyping.Array,
    sin: jaxtyping.Array,
    cos: jaxtyping.Array,
) -> Tuple[jaxtyping.Array, jaxtyping.Array]:
  """Apply rotary position embeddings to query and key.

  Args:
    q: Query tensor [B, num_heads, seq_len, head_dim]
    k: Key tensor [B, num_heads, seq_len, head_dim]
    sin: Sin embeddings [seq_len, head_dim // 2]
    cos: Cos embeddings [seq_len, head_dim // 2]

  Returns:
    q_rot, k_rot: Rotated query and key tensors
  """
  # Expand sin/cos to match query/key shape
  # [seq_len, head_dim // 2] -> [1, 1, seq_len, head_dim // 2]
  sin = sin[None, None, :, :]
  cos = cos[None, None, :, :]

  # Split into two halves
  q1, q2 = jnp.split(q, 2, axis=-1)
  k1, k2 = jnp.split(k, 2, axis=-1)

  # Apply rotation
  q_rot = jnp.concatenate([q1 * cos - q2 * sin, q2 * cos + q1 * sin], axis=-1)
  k_rot = jnp.concatenate([k1 * cos - k2 * sin, k2 * cos + k1 * sin], axis=-1)

  return q_rot, k_rot


class VisionAttention(nnx.Module):
  """Multi-head attention for vision encoder."""

  def __init__(
      self,
      config: VisionConfig,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.hidden_size = config.hidden_size
    self.num_heads = config.num_heads
    self.head_dim = config.head_dim

    self.scale = self.head_dim**-0.5

    # QKV projection
    self.qkv = nnx.Linear(
        in_features=self.hidden_size,
        out_features=3 * self.hidden_size,
        use_bias=True,
        rngs=rngs,
    )

    # Output projection
    self.proj = nnx.Linear(
        in_features=self.hidden_size,
        out_features=self.hidden_size,
        use_bias=True,
        rngs=rngs,
    )

  @jax.named_scope('vision_attention')
  def __call__(
      self,
      x: jaxtyping.Array,
      attention_mask: Optional[jaxtyping.Array] = None,
      cu_seqlens: Optional[jaxtyping.Array] = None,
      rotary_pos_emb: Optional[Tuple[jaxtyping.Array, jaxtyping.Array]] = None,
  ) -> jaxtyping.Array:
    """
    Args:
      x: Input tensor [B, seq_len, hidden_size]
      attention_mask: Optional attention mask
      cu_seqlens: Cumulative sequence lengths for packed sequences
      rotary_pos_emb: Optional (sin, cos) tuple for rotary embeddings

    Returns:
      Output tensor [B, seq_len, hidden_size]
    """
    batch_size, seq_len, _ = x.shape

    # Compute QKV
    qkv = self.qkv(x)  # [B, seq_len, 3 * hidden_size]
    qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
    qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))  # [3, B, num_heads, seq_len, head_dim]
    q, k, v = qkv[0], qkv[1], qkv[2]

    # Apply rotary embeddings if provided
    if rotary_pos_emb is not None:
      sin, cos = rotary_pos_emb
      q, k = apply_rotary_embedding_vision(q, k, sin, cos)

    # Compute attention scores
    attn_weights = jnp.einsum('bhqd,bhkd->bhqk', q, k) * self.scale

    # Apply attention mask if provided
    if attention_mask is not None:
      attn_weights = jnp.where(attention_mask, attn_weights, -1e9)

    # Softmax and apply to values
    attn_weights = jax.nn.softmax(attn_weights, axis=-1)
    attn_output = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, v)

    # Reshape and project
    attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))  # [B, seq_len, num_heads, head_dim]
    attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
    output = self.proj(attn_output)

    return output


class VisionMLP(nnx.Module):
  """MLP for vision encoder with configurable activation."""

  def __init__(
      self,
      config: VisionConfig,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config

    self.fc1 = nnx.Linear(
        in_features=config.hidden_size,
        out_features=config.intermediate_size,
        use_bias=True,
        rngs=rngs,
    )
    self.fc2 = nnx.Linear(
        in_features=config.intermediate_size,
        out_features=config.hidden_size,
        use_bias=True,
        rngs=rngs,
    )

    # Activation function
    if config.hidden_act == "silu":
      self.act = nnx.silu
    elif config.hidden_act == "gelu":
      self.act = nnx.gelu
    else:
      raise ValueError(f"Unsupported activation: {config.hidden_act}")

  @jax.named_scope('vision_mlp')
  def __call__(self, x: jaxtyping.Array) -> jaxtyping.Array:
    """
    Args:
      x: Input tensor [B, seq_len, hidden_size]

    Returns:
      Output tensor [B, seq_len, hidden_size]
    """
    x = self.fc1(x)
    x = self.act(x)
    x = self.fc2(x)
    return x


class Qwen2VLVisionBlock(nnx.Module):
  """Single transformer block for vision encoder."""

  def __init__(
      self,
      config: VisionConfig,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config

    self.norm1 = nnx.LayerNorm(
        num_features=config.hidden_size,
        epsilon=config.norm_eps,
        rngs=rngs,
    )
    self.attn = VisionAttention(config, rngs=rngs)

    self.norm2 = nnx.LayerNorm(
        num_features=config.hidden_size,
        epsilon=config.norm_eps,
        rngs=rngs,
    )
    self.mlp = VisionMLP(config, rngs=rngs)

  @jax.named_scope('vision_block')
  def __call__(
      self,
      x: jaxtyping.Array,
      attention_mask: Optional[jaxtyping.Array] = None,
      cu_seqlens: Optional[jaxtyping.Array] = None,
      rotary_pos_emb: Optional[Tuple[jaxtyping.Array, jaxtyping.Array]] = None,
  ) -> jaxtyping.Array:
    """
    Args:
      x: Input tensor [B, seq_len, hidden_size]
      attention_mask: Optional attention mask
      cu_seqlens: Cumulative sequence lengths
      rotary_pos_emb: Optional rotary embeddings

    Returns:
      Output tensor [B, seq_len, hidden_size]
    """
    # Attention with residual
    residual = x
    x = self.norm1(x)
    x = self.attn(x, attention_mask, cu_seqlens, rotary_pos_emb)
    x = residual + x

    # MLP with residual
    residual = x
    x = self.norm2(x)
    x = self.mlp(x)
    x = residual + x

    return x


class Qwen2VisionTransformerPretrainedModel(nnx.Module):
  """Vision transformer encoder for Qwen2-VL."""

  def __init__(
      self,
      config: VisionConfig,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config

    # Patch embedding
    self.patch_embed = PatchEmbed(config, rngs=rngs)

    # Rotary position embeddings
    self.rotary_pos_emb = VisionRotaryEmbedding(
        dim=config.head_dim,
        theta=10000.0,
    )

    # Transformer blocks
    self.blocks = [
        Qwen2VLVisionBlock(config, rngs=rngs) for _ in range(config.depth)
    ]

    # Spatial merge (2x2 patches -> 1 token)
    self.merger = nnx.Linear(
        in_features=config.hidden_size * (config.spatial_merge_size ** 2),
        out_features=config.hidden_size,
        use_bias=True,
        rngs=rngs,
    )

  def spatial_merge(self, x: jaxtyping.Array) -> jaxtyping.Array:
    """Merge spatial patches (2x2 -> 1 token).

    Args:
      x: Input [B, T, H, W, D] where H, W are divisible by spatial_merge_size

    Returns:
      Merged output [B, T, H', W', D] where H'=H//2, W'=W//2
    """
    B, T, H, W, D = x.shape
    merge_size = self.config.spatial_merge_size

    # Reshape to group patches
    # [B, T, H, W, D] -> [B, T, H//2, 2, W//2, 2, D]
    x = x.reshape(B, T, H // merge_size, merge_size, W // merge_size, merge_size, D)

    # Permute to bring merge dims together
    # -> [B, T, H//2, W//2, 2, 2, D]
    x = jnp.transpose(x, (0, 1, 2, 4, 3, 5, 6))

    # Flatten merge dimensions
    # -> [B, T, H//2, W//2, 4*D]
    x = x.reshape(B, T, H // merge_size, W // merge_size, merge_size * merge_size * D)

    # Project to original dimension
    x = self.merger(x)

    return x

  @jax.named_scope('vision_encoder')
  def __call__(
      self,
      pixel_values: jaxtyping.Array,
      grid_thw: Optional[jaxtyping.Array] = None,
  ) -> jaxtyping.Array:
    """
    Args:
      pixel_values: [B, T, H, W, C] for video or [B, 1, H, W, C] for images
      grid_thw: Grid dimensions [B, 3] specifying (temporal, height, width) patches

    Returns:
      hidden_states: [B, num_tokens, hidden_size]
    """
    # Patch embedding
    x = self.patch_embed(pixel_values)  # [B, T', H', W', D]

    B, T, H, W, D = x.shape

    # Flatten spatial dimensions for attention
    # [B, T', H', W', D] -> [B, T'*H'*W', D]
    seq_len = T * H * W
    x = x.reshape(B, seq_len, D)

    # Generate rotary embeddings
    sin, cos = self.rotary_pos_emb(seq_len)
    rotary_pos_emb = (sin, cos)

    # Apply transformer blocks
    for i, block in enumerate(self.blocks):
      # Use full attention for specified layers, window attention otherwise
      # For simplicity, we'll use full attention for all blocks
      # TODO: Implement window attention for non-fullatt blocks
      x = block(x, rotary_pos_emb=rotary_pos_emb)

    # Reshape back to spatial format
    x = x.reshape(B, T, H, W, D)

    # Spatial merge (2x2 patches -> 1 token)
    x = self.spatial_merge(x)  # [B, T, H//2, W//2, D]

    # Flatten to sequence
    _, T, H, W, D = x.shape
    x = x.reshape(B, T * H * W, D)

    return x

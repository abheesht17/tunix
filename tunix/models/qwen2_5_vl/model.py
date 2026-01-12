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

"""Qwen2.5 VL model - Vision-Language multimodal model."""

import dataclasses
import enum
from typing import Tuple

import flax
from flax import nnx
import jax
from jax import numpy as jnp
from jax.interpreters import pxla
import jax.sharding as shd
import jaxtyping
from tunix.utils import compat
from tunix.utils import env_utils

env_utils.setup_sharding_environment()

K_MASK = -2.3819763e38

LayerCache = dict[str, jaxtyping.Array]
Cache = dict[str, LayerCache]


class RematConfig(enum.Enum):
  NONE = enum.auto()  # No remat, all activations will be stored in HBM.
  BLOCK = enum.auto()  # Remat the entire attn block.


@dataclasses.dataclass(slots=True, frozen=True)
class ShardingConfig:
  """Sharding configuration for Qwen2.5 VL model."""

  emb_vd: Tuple[str | None, ...]
  emb_dv: Tuple[str | None, ...]
  q_weight_dnh: Tuple[str | None, ...]
  kv_weight_dnh: Tuple[str | None, ...]
  o_weight_nhd: Tuple[str | None, ...]
  ffw_weight_df: Tuple[str | None, ...]
  ffw_weight_fd: Tuple[str | None, ...]
  rms_norm_weight: Tuple[str | None, ...]
  act_btd: Tuple[str | None, ...]
  act_btf: Tuple[str | None, ...]
  act_btnh: Tuple[str | None, ...]
  qkv_bias: Tuple[str | None, ...]
  # Vision-specific sharding
  vision_patch_embed: Tuple[str | None, ...]
  vision_act: Tuple[str | None, ...]

  @staticmethod
  def get_default_sharding(is_sampling: bool = False):
    fsdp = 'fsdp' if not is_sampling else None

    return ShardingConfig(
        emb_vd=('tp', fsdp),
        emb_dv=(fsdp, 'tp'),
        q_weight_dnh=(fsdp, 'tp', None),
        kv_weight_dnh=(fsdp, 'tp', None),
        o_weight_nhd=('tp', None, fsdp),
        ffw_weight_df=(fsdp, 'tp'),
        ffw_weight_fd=('tp', fsdp),
        rms_norm_weight=('tp',),
        act_btd=('fsdp', None, None if is_sampling else 'tp'),
        act_btf=('fsdp', None, 'tp'),
        act_btnh=('fsdp', None, 'tp', None),
        qkv_bias=('tp',),
        vision_patch_embed=('tp', fsdp),
        vision_act=('fsdp', None, 'tp'),
    )


@dataclasses.dataclass(slots=True)
class VisionConfig:
  """Configuration for the vision encoder."""

  depth: int  # Number of vision transformer layers
  hidden_size: int  # Vision hidden dimension
  hidden_act: str  # Activation function
  intermediate_size: int  # Vision MLP intermediate size
  num_heads: int  # Vision attention heads
  in_channels: int  # Input image channels
  patch_size: int  # Spatial patch size
  spatial_merge_size: int  # Spatial merge factor
  temporal_patch_size: int  # Temporal patch size for video
  window_size: int  # Attention window size (currently unused in full attention)
  out_hidden_size: int  # Output dimension after patch merger
  norm_eps: float


@dataclasses.dataclass(slots=True)
class ModelConfig:
  """Configuration for the Qwen2.5 VL model."""

  # Text model config
  num_layers: int
  vocab_size: int
  embed_dim: int
  hidden_dim: int
  num_heads: int
  head_dim: int
  num_kv_heads: int
  rope_theta: int
  norm_eps: float
  use_tied_embedding: bool = False
  # Vision config
  vision_config: VisionConfig | None = None
  # Special token IDs
  image_token_id: int = 151655
  video_token_id: int = 151656
  vision_start_token_id: int = 151652
  vision_end_token_id: int = 151653
  # Multimodal RoPE config
  mrope_section: Tuple[int, int, int] = (16, 24, 24)  # temporal, height, width splits
  # Sharding config
  shd_config: ShardingConfig = ShardingConfig.get_default_sharding()
  remat_config: RematConfig = RematConfig.NONE

  @classmethod
  def qwen2p5_vl_3b(cls):
    """Qwen2.5-VL-3B configuration."""
    vision_config = VisionConfig(
        depth=32,
        hidden_size=3584,
        hidden_act='silu',
        intermediate_size=3420,
        num_heads=16,
        in_channels=3,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        window_size=112,
        out_hidden_size=3584,
        norm_eps=1e-6,
    )
    return cls(
        num_layers=80,
        vocab_size=152064,
        embed_dim=8192,
        hidden_dim=29568,
        num_heads=64,
        head_dim=128,
        num_kv_heads=8,
        norm_eps=1e-05,
        rope_theta=1_000_000,
        use_tied_embedding=False,
        vision_config=vision_config,
        mrope_section=(16, 24, 24),
    )


def shard(x: jnp.ndarray, s: Tuple[str, ...]):
  mesh = pxla.thread_resources.env.physical_mesh
  if mesh.empty or jax.devices()[0].platform == 'cpu':
    return x
  return jax.lax.with_sharding_constraint(
      x, shd.NamedSharding(mesh, shd.PartitionSpec(*s))
  )


class Einsum(nnx.Module):
  """Einsum is a convenience module for parameterized tensor multiplication."""

  def __init__(
      self,
      einsum_str: str,
      shape: flax.typing.Shape,
      *,
      rngs: nnx.Rngs,
      sharding: Tuple[str | None, ...],
  ):
    self.einsum_str = einsum_str
    self.shape = shape
    self.w = nnx.Param(
        nnx.initializers.normal()(rngs.params(), shape), sharding=sharding
    )

  @jax.named_scope('einsum')
  def __call__(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    return jnp.einsum(self.einsum_str, x, self.w.value)


class Embedder(nnx.Module):
  """Embedder module for text tokens."""

  def __init__(
      self,
      vocab_size: int,
      embed_dim: int,
      *,
      rngs: nnx.Rngs,
      shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ):
    self.input_embedding = nnx.Param(
        nnx.initializers.normal()(rngs.params(), (vocab_size, embed_dim)),
        sharding=shd_config.emb_vd,
    )
    self.shd_config = shd_config

  @jax.named_scope('embedder_encode')
  def encode(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    x = self.input_embedding[(x,)]
    x = shard(x, self.shd_config.act_btd)
    return x

  @jax.named_scope('embedder_decode')
  def decode(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    return jnp.dot(x, self.input_embedding.value.T)


class VisionPatchEmbed(nnx.Module):
  """3D patch embedding for vision inputs (images/videos)."""

  def __init__(
      self,
      vision_config: VisionConfig,
      *,
      rngs: nnx.Rngs,
      shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ):
    self.patch_size = vision_config.patch_size
    self.temporal_patch_size = vision_config.temporal_patch_size
    self.in_channels = vision_config.in_channels
    self.embed_dim = vision_config.hidden_size

    # 3D convolution for patch embedding
    kernel_shape = (
        vision_config.temporal_patch_size,
        vision_config.patch_size,
        vision_config.patch_size,
        vision_config.in_channels,
        vision_config.hidden_size,
    )
    self.proj = nnx.Param(
        nnx.initializers.normal(0.02)(rngs.params(), kernel_shape),
        sharding=shd_config.vision_patch_embed,
    )

  @jax.named_scope('vision_patch_embed')
  def __call__(self, x: jaxtyping.Array) -> jaxtyping.Array:
    """
    Args:
      x: [B, T, H, W, C] - batch, temporal, height, width, channels

    Returns:
      [B, num_patches, embed_dim]
    """
    # Apply 3D convolution with stride = kernel size
    # This is equivalent to non-overlapping patch extraction
    patches = jax.lax.conv_general_dilated(
        x,
        self.proj.value,
        window_strides=(
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        ),
        padding='VALID',
        dimension_numbers=('NTHWC', 'THWIO', 'NTHWC'),
    )
    # Reshape: [B, T', H', W', D] -> [B, T'*H'*W', D]
    batch_size = patches.shape[0]
    num_patches = patches.shape[1] * patches.shape[2] * patches.shape[3]
    patches = patches.reshape(batch_size, num_patches, self.embed_dim)
    return patches


class VisionPatchMerger(nnx.Module):
  """Merges spatial patches to reduce sequence length."""

  def __init__(
      self,
      vision_config: VisionConfig,
      *,
      rngs: nnx.Rngs,
      shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ):
    context_dim = vision_config.out_hidden_size
    spatial_merge_size = vision_config.spatial_merge_size
    self.hidden_size = context_dim * (spatial_merge_size**2)
    self.spatial_merge_size = spatial_merge_size

    self.ln_q = RMSNorm(
        context_dim,
        norm_eps=vision_config.norm_eps,
        rngs=rngs,
        shd_config=shd_config,
    )
    # Two-layer MLP with GELU
    self.mlp_1 = nnx.Linear(
        self.hidden_size,
        self.hidden_size,
        use_bias=True,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.normal(0.02), shd_config.ffw_weight_df
        ),
    )
    self.mlp_2 = nnx.Linear(
        self.hidden_size,
        vision_config.hidden_size,
        use_bias=True,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.normal(0.02), shd_config.ffw_weight_df
        ),
    )

  @jax.named_scope('vision_patch_merger')
  def __call__(
      self, x: jaxtyping.Array, grid_thw: jaxtyping.Array
  ) -> jaxtyping.Array:
    """
    Merges spatial patches by spatial_merge_size.

    Args:
      x: [B, num_patches, D] vision features
      grid_thw: [B, 3] - temporal, height, width grid dimensions

    Returns:
      [B, num_merged_patches, D] merged features
    """
    x = self.ln_q(x)
    # Reshape to merge spatial patches
    # [B, T*H*W, D] -> [B, T, H, W, D]
    batch_size = x.shape[0]
    t, h, w = grid_thw[0, 0], grid_thw[0, 1], grid_thw[0, 2]
    x = x.reshape(batch_size, t, h, w, -1)

    # Merge spatial patches by grouping
    merge_size = self.spatial_merge_size
    # Reshape: [B, T, H, W, D] -> [B, T, H//merge, merge, W//merge, merge, D]
    h_merged, w_merged = h // merge_size, w // merge_size
    x = x.reshape(
        batch_size,
        t,
        h_merged,
        merge_size,
        w_merged,
        merge_size,
        -1,
    )
    # Permute and reshape: [B, T, H', W', merge*merge, D] -> [B, T*H'*W', merge*merge*D]
    x = jnp.transpose(x, (0, 1, 2, 4, 3, 5, 6))
    x = x.reshape(batch_size, t * h_merged * w_merged, -1)

    # Apply MLP
    x = nnx.gelu(self.mlp_1(x))
    x = self.mlp_2(x)
    return x


class RMSNorm(nnx.Module):
  """RMSNorm layer."""

  def __init__(
      self,
      dim: int,
      *,
      norm_eps: float = 1e-06,
      rngs: nnx.Rngs,
      shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ):
    self.w = nnx.Param(
        nnx.initializers.ones_init()(rngs.params(), dim),
        sharding=shd_config.rms_norm_weight,
    )
    self.norm_eps = norm_eps

  @jax.named_scope('rms_norm')
  def __call__(self, x: jaxtyping.Array) -> jaxtyping.Array:
    dtype = x.dtype
    rms = jnp.sqrt(
        jnp.mean(jnp.astype(x, jnp.float32) ** 2, axis=-1, keepdims=True)
        + self.norm_eps
    )
    return self.w * jnp.astype(x / rms, dtype)


def _generate_pos_embeddings(
    positions: jax.Array,
    features: int,
    rope_theta: int,
) -> tuple[jax.Array, jax.Array]:
  """Generate Sin/Cos for Rotary Embeddings.

  Args:
      positions: [batch, time]
      features: head_dim.
      rope_theta: the rope_theta parameter.

  Returns:
      sin: a float32 array with shape [batch, length, features // 2]
      cos: a float32 array with shape [batch, length, features // 2]
  """
  fraction = jnp.arange(0, features, 2, dtype=jnp.float32) / features
  timescale = rope_theta**fraction
  rotational_frequency = 1.0 / timescale
  sinusoid_inp = jnp.einsum(
      'BT,k->BTk',
      positions,
      rotational_frequency,
      precision=jax.lax.Precision.HIGHEST,
  )
  return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)


def apply_rotary_embedding(
    x: jax.Array, sin: jax.Array, cos: jax.Array
) -> jax.Array:
  """Standard RoPE for text tokens."""
  assert x.ndim == 4 and sin.ndim == 3 and cos.ndim == 3
  x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
  # [B, T, head_dim] -> [B, h, T, head_dim]
  sin, cos = sin[:, :, None, :], cos[:, :, None, :]
  return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def apply_multimodal_rotary_embedding(
    q: jax.Array,
    k: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    mrope_section: Tuple[int, int, int],
) -> tuple[jax.Array, jax.Array]:
  """
  Apply 3D multimodal RoPE for vision-language inputs.

  Args:
    q, k: [B, T, H, D] query and key tensors
    cos, sin: [B, T, 3, D//2] - 3 sets of cos/sin for temporal, height, width
    mrope_section: (temporal_dim, height_dim, width_dim) - how to split head_dim

  Returns:
    q_embed, k_embed: rotated query and key
  """
  # Split head dimension according to mrope_section
  mrope_section_doubled = tuple(s * 2 for s in mrope_section)
  total_dim = sum(mrope_section_doubled)

  # cos/sin shape: [B, T, 3, D//2]
  # Split each into sections and apply cyclically
  q_parts = jnp.split(q, 3, axis=-1)
  k_parts = jnp.split(k, 3, axis=-1)

  q_rotated = []
  k_rotated = []

  for i, (q_part, k_part) in enumerate(zip(q_parts, k_parts)):
    # Use cos/sin for dimension i % 3
    cos_i = cos[:, :, i % 3, :]  # [B, T, D//2]
    sin_i = sin[:, :, i % 3, :]
    # Expand for heads: [B, T, D//2] -> [B, H, T, D//2]
    cos_i = cos_i[:, None, :, :]
    sin_i = sin_i[:, None, :, :]

    # Rotate this section
    q1, q2 = q_part[..., : q_part.shape[-1] // 2], q_part[..., q_part.shape[-1] // 2 :]
    k1, k2 = k_part[..., : k_part.shape[-1] // 2], k_part[..., k_part.shape[-1] // 2 :]

    q_rot = jnp.concatenate([q1 * cos_i - q2 * sin_i, q2 * cos_i + q1 * sin_i], axis=-1)
    k_rot = jnp.concatenate([k1 * cos_i - k2 * sin_i, k2 * cos_i + k1 * sin_i], axis=-1)

    q_rotated.append(q_rot)
    k_rotated.append(k_rot)

  return jnp.concatenate(q_rotated, axis=-1), jnp.concatenate(k_rotated, axis=-1)


class VisionAttention(nnx.Module):
  """Vision transformer attention with 2D RoPE."""

  def __init__(
      self,
      vision_config: VisionConfig,
      *,
      rngs: nnx.Rngs,
      shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ):
    self.num_heads = vision_config.num_heads
    self.head_dim = vision_config.hidden_size // vision_config.num_heads
    self.scale = self.head_dim**-0.5

    # QKV projection (combined)
    self.qkv = nnx.Linear(
        vision_config.hidden_size,
        3 * vision_config.hidden_size,
        use_bias=True,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.normal(0.02), shd_config.q_weight_dnh
        ),
    )
    self.o_proj = nnx.Linear(
        vision_config.hidden_size,
        vision_config.hidden_size,
        use_bias=False,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.normal(0.02), shd_config.o_weight_nhd
        ),
    )

  @jax.named_scope('vision_attention')
  def __call__(
      self,
      x: jaxtyping.Array,
      position_embeddings: tuple[jax.Array, jax.Array] | None = None,
  ) -> jaxtyping.Array:
    """
    Args:
      x: [B, N, D] input features
      position_embeddings: (cos, sin) for 2D RoPE

    Returns:
      [B, N, D] output features
    """
    batch_size, seq_len, _ = x.shape

    # QKV projection: [B, N, D] -> [B, N, 3*D]
    qkv = self.qkv(x)
    # Reshape: [B, N, 3*D] -> [B, N, 3, num_heads, head_dim]
    qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
    # Split into Q, K, V: each [B, N, num_heads, head_dim]
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

    # Apply 2D RoPE if position embeddings provided
    if position_embeddings is not None:
      cos, sin = position_embeddings
      # Reshape for RoPE: [B, N, H, D] -> [B, H, N, D]
      q = jnp.transpose(q, (0, 2, 1, 3))
      k = jnp.transpose(k, (0, 2, 1, 3))

      # Apply RoPE (simplified 2D version for vision)
      q1, q2 = q[..., : q.shape[-1] // 2], q[..., q.shape[-1] // 2 :]
      k1, k2 = k[..., : k.shape[-1] // 2], k[..., k.shape[-1] // 2 :]
      cos = cos[:, None, :, :]  # [B, 1, N, D//2]
      sin = sin[:, None, :, :]
      q = jnp.concatenate([q1 * cos - q2 * sin, q2 * cos + q1 * sin], axis=-1)
      k = jnp.concatenate([k1 * cos - k2 * sin, k2 * cos + k1 * sin], axis=-1)

      # Reshape back: [B, H, N, D] -> [B, N, H, D]
      q = jnp.transpose(q, (0, 2, 1, 3))
      k = jnp.transpose(k, (0, 2, 1, 3))

    # Attention: [B, N, H, D] @ [B, N, H, D]^T -> [B, H, N, N]
    q = jnp.transpose(q, (0, 2, 1, 3))  # [B, H, N, D]
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = jnp.transpose(v, (0, 2, 1, 3))

    attn = jnp.einsum('BHND,BHMD->BHNM', q, k) * self.scale
    attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(x.dtype)

    # Apply attention to values
    out = jnp.einsum('BHNM,BHMD->BHND', attn, v)
    out = jnp.transpose(out, (0, 2, 1, 3))  # [B, N, H, D]
    out = out.reshape(batch_size, seq_len, -1)

    # Output projection
    out = self.o_proj(out)
    return out


class VisionMLP(nnx.Module):
  """Vision MLP block."""

  def __init__(
      self,
      vision_config: VisionConfig,
      *,
      rngs: nnx.Rngs,
      shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ):
    self.fc1 = nnx.Linear(
        vision_config.hidden_size,
        vision_config.intermediate_size,
        use_bias=True,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.normal(0.02), shd_config.ffw_weight_df
        ),
    )
    self.fc2 = nnx.Linear(
        vision_config.intermediate_size,
        vision_config.hidden_size,
        use_bias=True,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.normal(0.02), shd_config.ffw_weight_fd
        ),
    )

  @jax.named_scope('vision_mlp')
  def __call__(self, x: jaxtyping.Array) -> jaxtyping.Array:
    # GELU activation (used in vision transformer)
    return self.fc2(nnx.gelu(self.fc1(x)))


class VisionBlock(nnx.Module):
  """Vision transformer block."""

  def __init__(
      self,
      vision_config: VisionConfig,
      *,
      rngs: nnx.Rngs,
      shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ):
    self.norm1 = RMSNorm(
        vision_config.hidden_size,
        norm_eps=vision_config.norm_eps,
        rngs=rngs,
        shd_config=shd_config,
    )
    self.norm2 = RMSNorm(
        vision_config.hidden_size,
        norm_eps=vision_config.norm_eps,
        rngs=rngs,
        shd_config=shd_config,
    )
    self.attn = VisionAttention(
        vision_config,
        rngs=rngs,
        shd_config=shd_config,
    )
    self.mlp = VisionMLP(
        vision_config,
        rngs=rngs,
        shd_config=shd_config,
    )

  @jax.named_scope('vision_block')
  def __call__(
      self,
      x: jaxtyping.Array,
      position_embeddings: tuple[jax.Array, jax.Array] | None = None,
  ) -> jaxtyping.Array:
    # Pre-norm architecture with residual connections
    x = x + self.attn(self.norm1(x), position_embeddings)
    x = x + self.mlp(self.norm2(x))
    return x


class VisionTransformer(nnx.Module):
  """Vision encoder with transformer blocks."""

  def __init__(
      self,
      vision_config: VisionConfig,
      *,
      rngs: nnx.Rngs,
      shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ):
    self.patch_embed = VisionPatchEmbed(
        vision_config,
        rngs=rngs,
        shd_config=shd_config,
    )
    self.blocks = compat.ModuleList([
        VisionBlock(vision_config, rngs=rngs, shd_config=shd_config)
        for _ in range(vision_config.depth)
    ])
    self.merger = VisionPatchMerger(
        vision_config,
        rngs=rngs,
        shd_config=shd_config,
    )

  @jax.named_scope('vision_transformer')
  def __call__(
      self,
      pixel_values: jaxtyping.Array,
      grid_thw: jaxtyping.Array,
  ) -> jaxtyping.Array:
    """
    Args:
      pixel_values: [B, T, H, W, C] image/video tensor
      grid_thw: [B, 3] temporal, height, width grid

    Returns:
      [B, num_merged_patches, embed_dim] vision features
    """
    # Patch embedding
    x = self.patch_embed(pixel_values)

    # Generate 2D position embeddings for vision (simplified)
    # In full implementation, this would be 3D (T, H, W)
    batch_size, num_patches, _ = x.shape
    positions = jnp.arange(num_patches)[None, :].repeat(batch_size, axis=0)
    # Simplified 2D RoPE for vision
    cos, sin = _generate_pos_embeddings(
        positions.astype(jnp.float32), 64, 10000
    )  # Use fixed params for vision

    # Apply transformer blocks
    for block in self.blocks:
      x = block(x, (cos, sin))

    # Merge patches to reduce sequence length
    x = self.merger(x, grid_thw)
    return x


class TextAttention(nnx.Module):
  """Text decoder attention with multimodal RoPE support."""

  def __init__(
      self,
      config: ModelConfig,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.shd_config = config.shd_config
    self.q_proj = Einsum(
        einsum_str='BTD,DNH->BTNH',
        shape=(config.embed_dim, config.num_heads, config.head_dim),
        rngs=rngs,
        sharding=self.shd_config.q_weight_dnh,
    )
    self.k_proj = Einsum(
        einsum_str='BSD,DKH->BSKH',
        shape=(config.embed_dim, config.num_kv_heads, config.head_dim),
        rngs=rngs,
        sharding=self.shd_config.kv_weight_dnh,
    )
    self.v_proj = Einsum(
        einsum_str='BSD,DKH->BSKH',
        shape=(config.embed_dim, config.num_kv_heads, config.head_dim),
        rngs=rngs,
        sharding=self.shd_config.kv_weight_dnh,
    )
    self.o_proj = Einsum(
        einsum_str='BTNH,NHD->BTD',
        shape=(config.num_heads, config.head_dim, config.embed_dim),
        rngs=rngs,
        sharding=self.shd_config.o_weight_nhd,
    )
    self.n_rep = config.num_heads // config.num_kv_heads
    self.scale = self.head_dim**-0.5
    self.q_bias = nnx.Param(
        nnx.initializers.zeros_init()(
            rngs.params(), config.num_heads * config.head_dim
        ),
        sharding=self.shd_config.qkv_bias,
    )
    self.k_bias = nnx.Param(
        nnx.initializers.zeros_init()(
            rngs.params(), config.num_kv_heads * config.head_dim
        ),
        sharding=self.shd_config.qkv_bias,
    )
    self.v_bias = nnx.Param(
        nnx.initializers.zeros_init()(
            rngs.params(), config.num_kv_heads * config.head_dim
        ),
        sharding=self.shd_config.qkv_bias,
    )

  def block(
      self,
      x: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array | None,
      sin: jaxtyping.Array,
      cos: jaxtyping.Array,
      use_multimodal_rope: bool = False,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    """Text attention block with optional multimodal RoPE."""
    seq_len = x.shape[1]

    query_proj = self.q_proj(x)
    b, t, n, h = query_proj.shape
    query_proj = jnp.reshape(query_proj, (b, t, n * h)) + self.q_bias
    query_proj = jnp.reshape(query_proj, (b, t, n, h))
    key_proj = self.k_proj(x)
    _, s, k, h = key_proj.shape
    key_proj = jnp.reshape(key_proj, (b, s, k * h)) + self.k_bias
    key_proj = jnp.reshape(key_proj, (b, s, k, h))
    value_proj = self.v_proj(x)
    value_proj = jnp.reshape(value_proj, (b, s, k * h)) + self.v_bias
    value_proj = jnp.reshape(value_proj, (b, s, k, h))

    query_proj = shard(query_proj, self.shd_config.act_btnh)
    key_proj = shard(key_proj, self.shd_config.act_btnh)
    value_proj = shard(value_proj, self.shd_config.act_btnh)

    # Apply RoPE (standard or multimodal)
    if use_multimodal_rope and sin.ndim == 4:
      # Multimodal RoPE: sin/cos are [B, T, 3, D//2]
      query_proj, key_proj = apply_multimodal_rotary_embedding(
          query_proj, key_proj, cos, sin, self.config.mrope_section
      )
    else:
      # Standard RoPE
      query_proj = apply_rotary_embedding(query_proj, sin, cos)
      key_proj = apply_rotary_embedding(key_proj, sin, cos)

    if cache is not None:
      end_index = cache['end_index'][0]
      slice_indices = (0, end_index % cache['v'].shape[1], 0, 0)
      value_proj = jax.lax.dynamic_update_slice(
          cache['v'],
          value_proj,
          slice_indices,
      )
      key_proj = jax.lax.dynamic_update_slice(
          cache['k'], key_proj, slice_indices
      )

    b, t, qh, d = query_proj.shape
    _, s, kh, _ = key_proj.shape

    # GQA
    query_proj = query_proj.reshape((b, t, kh, qh // kh, d))
    attn = jnp.einsum('BTHGD,BSHD->BHGTS', query_proj, key_proj) * self.scale
    attn = attn.reshape((b, qh, t, s))

    if attn_mask is not None:
      attn = jnp.where((jnp.expand_dims(attn_mask, -3)), attn, K_MASK)

    attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(
        key_proj.dtype
    )

    attn = attn.reshape((b, kh, qh // kh, t, s))
    qkv = jnp.einsum('BHGTS,BSHD->BTHGD', attn, value_proj)
    qkv = qkv.reshape((b, t, qh, d))

    outputs = self.o_proj(qkv)
    outputs = shard(outputs, self.shd_config.act_btd)

    if cache is not None:
      new_cache = {
          'v': value_proj,
          'k': key_proj,
          'end_index': cache['end_index'] + seq_len,
      }
    else:
      new_cache = None

    return new_cache, outputs

  @jax.named_scope('text_attention')
  def __call__(
      self,
      x: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array | None,
      sin: jaxtyping.Array,
      cos: jaxtyping.Array,
      use_multimodal_rope: bool = False,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    if self.config.remat_config == RematConfig.BLOCK:
      return nnx.remat(self.block.__func__)(
          self, x, cache, attn_mask, sin, cos, use_multimodal_rope
      )
    else:
      return self.block(x, cache, attn_mask, sin, cos, use_multimodal_rope)

  @property
  def head_dim(self):
    return self.o_proj.shape[1]

  @property
  def num_heads(self):
    return self.q_proj.shape[0]

  @property
  def num_kv_heads(self):
    return self.k_proj.shape[1]


class MLP(nnx.Module):
  """MLP module for text decoder."""

  def __init__(
      self,
      config: ModelConfig,
      *,
      rngs: nnx.Rngs,
  ):
    self.shd_config = config.shd_config
    kernel_init_fn = nnx.initializers.zeros_init()
    self.gate_proj = nnx.Linear(
        in_features=config.embed_dim,
        out_features=config.hidden_dim,
        use_bias=False,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            kernel_init_fn, self.shd_config.ffw_weight_df
        ),
    )
    self.up_proj = nnx.Linear(
        in_features=config.embed_dim,
        out_features=config.hidden_dim,
        use_bias=False,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            kernel_init_fn, self.shd_config.ffw_weight_df
        ),
    )
    self.down_proj = nnx.Linear(
        in_features=config.hidden_dim,
        out_features=config.embed_dim,
        use_bias=False,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            kernel_init_fn, self.shd_config.ffw_weight_fd
        ),
    )

  @jax.named_scope('feed_forward')
  def __call__(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    activations = nnx.silu(self.gate_proj(x)) * self.up_proj(x)
    activations = shard(activations, self.shd_config.act_btf)
    outputs = self.down_proj(activations)
    return outputs


class TextDecoderLayer(nnx.Module):
  """Text decoder layer."""

  def __init__(
      self,
      config: ModelConfig,
      *,
      rngs: nnx.Rngs,
  ):
    self.input_layernorm = RMSNorm(
        config.embed_dim,
        norm_eps=config.norm_eps,
        rngs=rngs,
        shd_config=config.shd_config,
    )
    self.attn = TextAttention(
        config=config,
        rngs=rngs,
    )
    self.post_attention_layernorm = RMSNorm(
        config.embed_dim,
        norm_eps=config.norm_eps,
        rngs=rngs,
        shd_config=config.shd_config,
    )
    self.mlp = MLP(
        config=config,
        rngs=rngs,
    )

  def __call__(
      self,
      x: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array,
      sin: jaxtyping.Array,
      cos: jaxtyping.Array,
      use_multimodal_rope: bool = False,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    inputs_normalized = self.input_layernorm(x)
    cache, attn_output = self.attn(
        inputs_normalized,
        cache,
        attn_mask,
        sin,
        cos,
        use_multimodal_rope,
    )
    attn_output += x
    residual = attn_output
    attn_output = self.post_attention_layernorm(attn_output)
    outputs = self.mlp(attn_output)
    outputs = residual + outputs
    return cache, outputs


class Qwen2_5_VL(nnx.Module):
  """Qwen2.5 VL multimodal model."""

  def __init__(
      self,
      config: ModelConfig,
      *,
      rngs: nnx.Rngs,
      shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ):
    self.config = config
    # Text embedder
    self.embedder = Embedder(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        rngs=rngs,
        shd_config=shd_config,
    )
    # Vision encoder (if config has vision)
    if config.vision_config is not None:
      self.vision_encoder = VisionTransformer(
          config.vision_config,
          rngs=rngs,
          shd_config=shd_config,
      )
    # Text decoder layers
    self.layers = compat.ModuleList([
        TextDecoderLayer(config=config, rngs=rngs)
        for _ in range(config.num_layers)
    ])
    self.final_norm = RMSNorm(
        config.embed_dim,
        rngs=rngs,
        norm_eps=config.norm_eps,
        shd_config=shd_config,
    )
    if not self.config.use_tied_embedding:
      self.lm_head = Einsum(
          einsum_str='BTD,DV->BTV',
          shape=(config.embed_dim, config.vocab_size),
          rngs=rngs,
          sharding=shd_config.emb_dv,
      )

  def __call__(
      self,
      input_tokens: jaxtyping.Array,  # [B, L]
      positions: jaxtyping.Array,  # [B, L]
      cache: Cache | None,  # (sequence length L')
      attention_mask: jaxtyping.Array,  # [B, L, L']
      pixel_values: jaxtyping.Array | None = None,  # [B, T, H, W, C]
      image_grid_thw: jaxtyping.Array | None = None,  # [B, 3]
      output_hidden_states: bool = False,
  ) -> tuple[jaxtyping.Array, Cache | None]:
    """Qwen2.5 VL model forward pass.

    Args:
      input_tokens: input sequence of tokens (may include vision placeholders)
      positions: input absolute positions
      cache: Attention KV cache or None
      attention_mask: transformer input mask
      pixel_values: vision inputs (images/videos)
      image_grid_thw: temporal/height/width grid for vision inputs
      output_hidden_states: whether to output the hidden states

    Returns:
      predicted_logits, new_cache
    """
    new_cache = None if cache is None else {}

    # Encode text tokens
    x = self.embedder.encode(input_tokens)

    # Process vision inputs if provided
    if pixel_values is not None and self.config.vision_config is not None:
      vision_features = self.vision_encoder(pixel_values, image_grid_thw)
      # TODO: Integrate vision features into text embeddings
      # This requires identifying vision placeholder tokens and replacing them
      # For now, we'll skip this integration step in the basic implementation

    # Generate position embeddings (standard RoPE for text-only)
    sin, cos = _generate_pos_embeddings(
        positions, self.config.head_dim, self.config.rope_theta
    )
    sin, cos = sin.astype(x.dtype), cos.astype(x.dtype)

    # Apply text decoder layers
    use_multimodal_rope = pixel_values is not None
    for i, layer in enumerate(self.layers):
      layer_name = f'layer_{i}'
      layer_cache = cache[layer_name] if cache else None
      layer_cache, x = layer(
          x,
          layer_cache,
          attention_mask,
          sin,
          cos,
          use_multimodal_rope,
      )
      if cache is not None:
        new_cache[layer_name] = layer_cache

    x = self.final_norm(x)
    if output_hidden_states:
      self.sow(nnx.Intermediate, 'all_hidden_states', x)

    if self.config.use_tied_embedding:
      logits = self.embedder.decode(x)
    else:
      logits = self.lm_head(x)

    return logits, new_cache

  def get_model_input(self):
    """Returns a dummy model input for the transformer."""
    dummy_batch_size = 2
    dummy_seq_len = 1
    return {
        'input_tokens': jnp.ones(
            (dummy_batch_size, dummy_seq_len), dtype=jnp.int32
        ),
        'positions': jnp.ones(
            (dummy_batch_size, dummy_seq_len), dtype=jnp.int32
        ),
        'cache': None,
        'attention_mask': jnp.ones(
            (dummy_batch_size, 1, dummy_seq_len), dtype=jnp.bool
        ),
        'pixel_values': None,
        'image_grid_thw': None,
    }

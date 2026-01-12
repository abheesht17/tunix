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

"""Utils for loading and converting Qwen2.5 VL PT weights."""

import jax
import jax.numpy as jnp
from tunix.models import safetensors_loader
from tunix.models.qwen2_5_vl import model as model_lib


def _get_key_and_transform_mapping(cfg: model_lib.ModelConfig):
  """Mapping of PyTorch keys to NNX keys with transformations.

  Returns mapping of: torch_key_pattern -> (nnx_key_pattern, (permute, reshape))
  """
  mapping = {
      # Text embeddings
      r"model\.embed_tokens\.weight": ("embedder.input_embedding", None),
      # Vision encoder - patch embedding (Conv3D weights)
      r"visual\.patch_embed\.proj\.weight": (
          "vision_encoder.patch_embed.proj",
          # Conv3D in PyTorch: [out_channels, in_channels, T, H, W]
          # Conv3D in JAX: [T, H, W, in_channels, out_channels]
          ((2, 3, 4, 1, 0), None),
      ),
      # Vision encoder - transformer blocks
      r"visual\.blocks\.([0-9]+)\.norm1\.weight": (
          r"vision_encoder.blocks.\1.norm1.w",
          None,
      ),
      r"visual\.blocks\.([0-9]+)\.norm2\.weight": (
          r"vision_encoder.blocks.\1.norm2.w",
          None,
      ),
      # Vision attention - QKV projection (combined weight)
      r"visual\.blocks\.([0-9]+)\.attn\.qkv\.weight": (
          r"vision_encoder.blocks.\1.attn.qkv.kernel",
          ((1, 0), None),
      ),
      r"visual\.blocks\.([0-9]+)\.attn\.qkv\.bias": (
          r"vision_encoder.blocks.\1.attn.qkv.bias",
          None,
      ),
      # Vision attention - output projection
      r"visual\.blocks\.([0-9]+)\.attn\.proj\.weight": (
          r"vision_encoder.blocks.\1.attn.o_proj.kernel",
          ((1, 0), None),
      ),
      # Vision MLP
      r"visual\.blocks\.([0-9]+)\.mlp\.fc1\.weight": (
          r"vision_encoder.blocks.\1.mlp.fc1.kernel",
          ((1, 0), None),
      ),
      r"visual\.blocks\.([0-9]+)\.mlp\.fc1\.bias": (
          r"vision_encoder.blocks.\1.mlp.fc1.bias",
          None,
      ),
      r"visual\.blocks\.([0-9]+)\.mlp\.fc2\.weight": (
          r"vision_encoder.blocks.\1.mlp.fc2.kernel",
          ((1, 0), None),
      ),
      r"visual\.blocks\.([0-9]+)\.mlp\.fc2\.bias": (
          r"vision_encoder.blocks.\1.mlp.fc2.bias",
          None,
      ),
      # Vision patch merger
      r"visual\.merger\.ln_q\.weight": (
          "vision_encoder.merger.ln_q.w",
          None,
      ),
      r"visual\.merger\.mlp\.0\.weight": (
          "vision_encoder.merger.mlp_1.kernel",
          ((1, 0), None),
      ),
      r"visual\.merger\.mlp\.0\.bias": (
          "vision_encoder.merger.mlp_1.bias",
          None,
      ),
      r"visual\.merger\.mlp\.2\.weight": (
          "vision_encoder.merger.mlp_2.kernel",
          ((1, 0), None),
      ),
      r"visual\.merger\.mlp\.2\.bias": (
          "vision_encoder.merger.mlp_2.bias",
          None,
      ),
      # Text decoder - attention projection weights
      r"model\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": (
          r"layers.\1.attn.q_proj.w",
          ((1, 0), (cfg.embed_dim, cfg.num_heads, cfg.head_dim)),
      ),
      r"model\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": (
          r"layers.\1.attn.k_proj.w",
          ((1, 0), (cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim)),
      ),
      r"model\.layers\.([0-9]+)\.self_attn\.v_proj\.weight": (
          r"layers.\1.attn.v_proj.w",
          ((1, 0), (cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim)),
      ),
      r"model\.layers\.([0-9]+)\.self_attn\.o_proj\.weight": (
          r"layers.\1.attn.o_proj.w",
          ((1, 0), (cfg.num_heads, cfg.head_dim, cfg.embed_dim)),
      ),
      # Text decoder - attention biases
      r"model\.layers\.([0-9]+)\.self_attn\.q_proj\.bias": (
          r"layers.\1.attn.q_bias",
          None,
      ),
      r"model\.layers\.([0-9]+)\.self_attn\.k_proj\.bias": (
          r"layers.\1.attn.k_bias",
          None,
      ),
      r"model\.layers\.([0-9]+)\.self_attn\.v_proj\.bias": (
          r"layers.\1.attn.v_bias",
          None,
      ),
      # Text decoder - MLP
      r"model\.layers\.([0-9]+)\.mlp\.gate_proj\.weight": (
          r"layers.\1.mlp.gate_proj.kernel",
          ((1, 0), None),
      ),
      r"model\.layers\.([0-9]+)\.mlp\.up_proj\.weight": (
          r"layers.\1.mlp.up_proj.kernel",
          ((1, 0), None),
      ),
      r"model\.layers\.([0-9]+)\.mlp\.down_proj\.weight": (
          r"layers.\1.mlp.down_proj.kernel",
          ((1, 0), None),
      ),
      # Text decoder - layer norms
      r"model\.norm\.weight": ("final_norm.w", None),
      r"model\.layers\.([0-9]+)\.input_layernorm\.weight": (
          r"layers.\1.input_layernorm.w",
          None,
      ),
      r"model\.layers\.([0-9]+)\.post_attention_layernorm\.weight": (
          r"layers.\1.post_attention_layernorm.w",
          None,
      ),
      # LM head
      r"lm_head\.weight": ("lm_head.w", ((1, 0), None)),
  }
  return mapping


def create_model_from_safe_tensors(
    file_dir: str,
    config: model_lib.ModelConfig,
    mesh: jax.sharding.Mesh | None = None,
    dtype: jnp.dtype | None = None,
) -> model_lib.Qwen2_5_VL:
  """Load tensors from the safetensors file and create a Qwen2.5 VL model.

  Args:
    file_dir: Directory containing safetensors files
    config: Model configuration
    mesh: JAX sharding mesh (optional)
    dtype: Data type for model weights (optional)

  Returns:
    Qwen2_5_VL model with loaded weights
  """
  return safetensors_loader.load_and_create_model(
      file_dir=file_dir,
      model_class=model_lib.Qwen2_5_VL,
      config=config,
      key_mapping=_get_key_and_transform_mapping,
      mesh=mesh,
      preprocess_fn=None,
      dtype=dtype,
  )

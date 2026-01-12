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

"""Utils for loading and converting Qwen2-VL weights from HuggingFace."""

from typing import Any, Dict

import jax
import jax.numpy as jnp

from tunix.models import safetensors_loader
from tunix.models.qwen2_vl import model as model_lib
from tunix.models.qwen2_vl.config import Qwen2VLConfig


def _get_vision_key_mapping(vision_config):
  """Get key mapping for vision encoder weights."""
  # Mapping from HuggingFace keys to our JAX model keys
  mapping = {
      # Patch embedding
      r"visual\.patch_embed\.proj\.weight": (
          "visual.patch_embed.proj.kernel",
          ((2, 3, 4, 1, 0), None),  # Conv3D: [out, in, t, h, w] -> [t, h, w, in, out]
      ),
      # Vision blocks
      r"visual\.blocks\.([0-9]+)\.norm1\.weight": (
          r"visual.blocks.\1.norm1.scale",
          None,
      ),
      r"visual\.blocks\.([0-9]+)\.norm1\.bias": (
          r"visual.blocks.\1.norm1.bias",
          None,
      ),
      r"visual\.blocks\.([0-9]+)\.attn\.qkv\.weight": (
          r"visual.blocks.\1.attn.qkv.kernel",
          ((1, 0), None),
      ),
      r"visual\.blocks\.([0-9]+)\.attn\.qkv\.bias": (
          r"visual.blocks.\1.attn.qkv.bias",
          None,
      ),
      r"visual\.blocks\.([0-9]+)\.attn\.proj\.weight": (
          r"visual.blocks.\1.attn.proj.kernel",
          ((1, 0), None),
      ),
      r"visual\.blocks\.([0-9]+)\.attn\.proj\.bias": (
          r"visual.blocks.\1.attn.proj.bias",
          None,
      ),
      r"visual\.blocks\.([0-9]+)\.norm2\.weight": (
          r"visual.blocks.\1.norm2.scale",
          None,
      ),
      r"visual\.blocks\.([0-9]+)\.norm2\.bias": (
          r"visual.blocks.\1.norm2.bias",
          None,
      ),
      r"visual\.blocks\.([0-9]+)\.mlp\.fc1\.weight": (
          r"visual.blocks.\1.mlp.fc1.kernel",
          ((1, 0), None),
      ),
      r"visual\.blocks\.([0-9]+)\.mlp\.fc1\.bias": (
          r"visual.blocks.\1.mlp.fc1.bias",
          None,
      ),
      r"visual\.blocks\.([0-9]+)\.mlp\.fc2\.weight": (
          r"visual.blocks.\1.mlp.fc2.kernel",
          ((1, 0), None),
      ),
      r"visual\.blocks\.([0-9]+)\.mlp\.fc2\.bias": (
          r"visual.blocks.\1.mlp.fc2.bias",
          None,
      ),
      # Spatial merger
      r"visual\.merger\.weight": (
          "visual.merger.kernel",
          ((1, 0), None),
      ),
      r"visual\.merger\.bias": (
          "visual.merger.bias",
          None,
      ),
  }
  return mapping


def _get_connector_key_mapping():
  """Get key mapping for vision-language connector."""
  return {
      # Connector layer norm
      r"connector\.ln_q\.weight": (
          "connector.merger.ln_q.scale",
          None,
      ),
      r"connector\.ln_q\.bias": (
          "connector.merger.ln_q.bias",
          None,
      ),
      # Connector MLP
      r"connector\.mlp\.([0-9]+)\.weight": (
          r"connector.merger.mlp.\1.kernel",
          ((1, 0), None),
      ),
      r"connector\.mlp\.([0-9]+)\.bias": (
          r"connector.merger.mlp.\1.bias",
          None,
      ),
  }


def _get_key_and_transform_mapping(config: Qwen2VLConfig):
  """Get complete key mapping for Qwen2-VL model.

  Args:
    config: Qwen2VL configuration

  Returns:
    Dictionary mapping HuggingFace keys to (JAX keys, transformations)
  """
  # Start with language model mappings (from qwen2.params)
  from tunix.models.qwen2.params import _get_key_and_transform_mapping as get_qwen2_mapping

  mapping = get_qwen2_mapping(config.text_config)

  # Rename to include 'language_model' prefix for our structure
  language_mapping = {}
  for hf_key, (jax_key, transform) in mapping.items():
    # Skip lm_head as it's part of language_model in our implementation
    new_jax_key = f"language_model.{jax_key}"
    language_mapping[hf_key] = (new_jax_key, transform)

  # Add vision encoder mappings
  vision_mapping = _get_vision_key_mapping(config.vision_config)

  # Add connector mappings
  connector_mapping = _get_connector_key_mapping()

  # Combine all mappings
  complete_mapping = {**language_mapping, **vision_mapping, **connector_mapping}

  return complete_mapping


def create_model_from_safe_tensors(
    file_dir: str,
    config: Qwen2VLConfig,
    mesh: jax.sharding.Mesh | None = None,
    dtype: jnp.dtype | None = None,
) -> model_lib.Qwen2VLForConditionalGeneration:
  """Load Qwen2-VL weights from safetensors and create model.

  Args:
    file_dir: Directory containing safetensors files
    config: Qwen2VL configuration
    mesh: Optional JAX mesh for sharding
    dtype: Optional dtype for model weights

  Returns:
    Qwen2VLForConditionalGeneration model with loaded weights
  """
  return safetensors_loader.load_and_create_model(
      file_dir=file_dir,
      model_class=model_lib.Qwen2VLForConditionalGeneration,
      config=config,
      key_mapping=_get_key_and_transform_mapping,
      mesh=mesh,
      preprocess_fn=None,
      dtype=dtype,
  )


def create_model_from_huggingface(
    model_name: str,
    config: Qwen2VLConfig | None = None,
    mesh: jax.sharding.Mesh | None = None,
    dtype: jnp.dtype | None = None,
) -> model_lib.Qwen2VLForConditionalGeneration:
  """Load Qwen2-VL model from HuggingFace Hub.

  Args:
    model_name: HuggingFace model name (e.g., "Qwen/Qwen2.5-VL-7B-Instruct")
    config: Optional config (will be inferred from model if not provided)
    mesh: Optional JAX mesh for sharding
    dtype: Optional dtype for model weights

  Returns:
    Qwen2VLForConditionalGeneration model with loaded weights
  """
  # TODO: Implement HuggingFace download and loading
  # For now, this is a placeholder
  raise NotImplementedError(
      "HuggingFace loading not yet implemented. "
      "Please download the model manually and use create_model_from_safe_tensors."
  )

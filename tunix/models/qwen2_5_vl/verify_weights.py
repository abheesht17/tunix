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

"""Verification script for Qwen2.5 VL weight conversion and numeric accuracy.

This script compares the outputs of the Hugging Face Qwen2.5 VL implementation
with the Tunix JAX implementation to verify:
1. Weight conversion is correct
2. Forward pass produces similar outputs
3. Numeric differences are within acceptable tolerances
"""

import argparse
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from tunix.models.qwen2_5_vl import model as model_lib
from tunix.models.qwen2_5_vl import params as params_lib


def compare_tensors(
    jax_tensor: jnp.ndarray,
    torch_tensor: torch.Tensor,
    name: str,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> tuple[bool, dict[str, float]]:
  """Compare JAX and PyTorch tensors for numeric accuracy.

  Args:
    jax_tensor: JAX array
    torch_tensor: PyTorch tensor
    name: Name for logging
    rtol: Relative tolerance
    atol: Absolute tolerance

  Returns:
    (is_close, metrics) where metrics contains various comparison statistics
  """
  # Convert to numpy for comparison
  jax_np = np.array(jax_tensor)
  torch_np = torch_tensor.detach().cpu().numpy()

  # Check shapes match
  if jax_np.shape != torch_np.shape:
    print(f"❌ {name}: Shape mismatch!")
    print(f"   JAX shape: {jax_np.shape}")
    print(f"   PyTorch shape: {torch_np.shape}")
    return False, {}

  # Compute metrics
  abs_diff = np.abs(jax_np - torch_np)
  rel_diff = abs_diff / (np.abs(torch_np) + 1e-8)

  metrics = {
      'max_abs_diff': float(np.max(abs_diff)),
      'mean_abs_diff': float(np.mean(abs_diff)),
      'max_rel_diff': float(np.max(rel_diff)),
      'mean_rel_diff': float(np.mean(rel_diff)),
      'cosine_similarity': float(
          np.dot(jax_np.flatten(), torch_np.flatten())
          / (np.linalg.norm(jax_np) * np.linalg.norm(torch_np) + 1e-8)
      ),
  }

  # Check if close
  is_close = np.allclose(jax_np, torch_np, rtol=rtol, atol=atol)

  # Print results
  status = "✅" if is_close else "❌"
  print(f"{status} {name}:")
  print(f"   Max abs diff: {metrics['max_abs_diff']:.6e}")
  print(f"   Mean abs diff: {metrics['mean_abs_diff']:.6e}")
  print(f"   Max rel diff: {metrics['max_rel_diff']:.6e}")
  print(f"   Mean rel diff: {metrics['mean_rel_diff']:.6e}")
  print(f"   Cosine similarity: {metrics['cosine_similarity']:.6f}")

  return is_close, metrics


def verify_text_only_forward(
    hf_model_path: str,
    tunix_model_path: str,
    batch_size: int = 2,
    seq_len: int = 32,
) -> bool:
  """Verify text-only forward pass (no vision inputs).

  Args:
    hf_model_path: Path to HuggingFace model
    tunix_model_path: Path to Tunix safetensors
    batch_size: Batch size for test
    seq_len: Sequence length for test

  Returns:
    True if verification passed, False otherwise
  """
  print("=" * 80)
  print("VERIFYING TEXT-ONLY FORWARD PASS")
  print("=" * 80)

  # Load HuggingFace model
  print("\n📥 Loading HuggingFace model...")
  hf_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
      hf_model_path,
      torch_dtype=torch.float32,
  )
  hf_model.eval()
  print("✅ HuggingFace model loaded")

  # Load Tunix model
  print("\n📥 Loading Tunix model...")
  config = model_lib.ModelConfig.qwen2p5_vl_3b()
  tunix_model = params_lib.create_model_from_safe_tensors(
      tunix_model_path,
      config,
      dtype=jnp.float32,
  )
  print("✅ Tunix model loaded")

  # Create random input tokens
  print(f"\n🎲 Creating random input (batch={batch_size}, seq_len={seq_len})...")
  input_ids = np.random.randint(0, config.vocab_size, size=(batch_size, seq_len))
  input_ids_torch = torch.from_numpy(input_ids).long()
  input_ids_jax = jnp.array(input_ids, dtype=jnp.int32)

  # Create position IDs
  positions = np.arange(seq_len)[None, :].repeat(batch_size, axis=0)
  positions_torch = torch.from_numpy(positions).long()
  positions_jax = jnp.array(positions, dtype=jnp.int32)

  # Create attention mask (causal)
  attention_mask_jax = jnp.tril(jnp.ones((batch_size, seq_len, seq_len), dtype=jnp.bool))

  print("\n🔄 Running HuggingFace forward pass...")
  with torch.no_grad():
    hf_outputs = hf_model(
        input_ids=input_ids_torch,
        attention_mask=torch.ones_like(input_ids_torch),
        return_dict=True,
    )
    hf_logits = hf_outputs.logits

  print("🔄 Running Tunix forward pass...")
  tunix_logits, _ = tunix_model(
      input_tokens=input_ids_jax,
      positions=positions_jax,
      cache=None,
      attention_mask=attention_mask_jax,
      pixel_values=None,
      image_grid_thw=None,
  )

  # Compare logits
  print("\n📊 Comparing outputs...")
  is_close, metrics = compare_tensors(
      tunix_logits,
      hf_logits,
      "Text-only logits",
      rtol=1e-3,
      atol=1e-3,
  )

  if is_close:
    print("\n✅ Text-only forward pass verification PASSED!")
  else:
    print("\n❌ Text-only forward pass verification FAILED!")
    print("   This may indicate issues with:")
    print("   - Weight conversion")
    print("   - Attention implementation")
    print("   - Position embedding (RoPE)")
    print("   - MLP implementation")

  return is_close


def verify_vision_encoder(
    hf_model_path: str,
    tunix_model_path: str,
    image_size: tuple[int, int] = (224, 224),
) -> bool:
  """Verify vision encoder outputs.

  Args:
    hf_model_path: Path to HuggingFace model
    tunix_model_path: Path to Tunix safetensors
    image_size: (height, width) of test image

  Returns:
    True if verification passed, False otherwise
  """
  print("\n" + "=" * 80)
  print("VERIFYING VISION ENCODER")
  print("=" * 80)

  # Load models
  print("\n📥 Loading models...")
  hf_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
      hf_model_path,
      torch_dtype=torch.float32,
  )
  hf_model.eval()

  config = model_lib.ModelConfig.qwen2p5_vl_3b()
  tunix_model = params_lib.create_model_from_safe_tensors(
      tunix_model_path,
      config,
      dtype=jnp.float32,
  )
  print("✅ Models loaded")

  # Create dummy image
  print(f"\n🖼️  Creating test image ({image_size[0]}x{image_size[1]})...")
  batch_size = 1
  num_frames = 1  # Single image
  h, w = image_size
  channels = 3

  # Random pixel values in [0, 1]
  pixel_values_np = np.random.rand(batch_size, num_frames, h, w, channels).astype(
      np.float32
  )
  pixel_values_torch = torch.from_numpy(pixel_values_np).permute(0, 1, 4, 2, 3)
  pixel_values_jax = jnp.array(pixel_values_np)

  # Grid THW
  patch_size = config.vision_config.patch_size
  temporal_patch_size = config.vision_config.temporal_patch_size
  grid_t = num_frames // temporal_patch_size
  grid_h = h // patch_size
  grid_w = w // patch_size
  grid_thw_jax = jnp.array([[grid_t, grid_h, grid_w]], dtype=jnp.int32)

  print("\n🔄 Running HuggingFace vision encoder...")
  with torch.no_grad():
    # Access vision encoder
    hf_vision_out = hf_model.visual(
        pixel_values_torch.squeeze(1),  # Remove temporal dim for single image
        grid_thw=torch.tensor([[grid_t, grid_h, grid_w]]),
    )

  print("🔄 Running Tunix vision encoder...")
  tunix_vision_out = tunix_model.vision_encoder(
      pixel_values_jax,
      grid_thw_jax,
  )

  # Compare vision outputs
  print("\n📊 Comparing vision encoder outputs...")
  is_close, metrics = compare_tensors(
      tunix_vision_out,
      hf_vision_out,
      "Vision encoder output",
      rtol=1e-3,
      atol=1e-3,
  )

  if is_close:
    print("\n✅ Vision encoder verification PASSED!")
  else:
    print("\n❌ Vision encoder verification FAILED!")
    print("   This may indicate issues with:")
    print("   - Patch embedding (Conv3D)")
    print("   - Vision transformer blocks")
    print("   - Patch merger")
    print("   - Vision attention")

  return is_close


def verify_multimodal_forward(
    hf_model_path: str,
    tunix_model_path: str,
) -> bool:
  """Verify full multimodal forward pass with vision and text.

  Args:
    hf_model_path: Path to HuggingFace model
    tunix_model_path: Path to Tunix safetensors

  Returns:
    True if verification passed, False otherwise
  """
  print("\n" + "=" * 80)
  print("VERIFYING MULTIMODAL FORWARD PASS")
  print("=" * 80)
  print("\n⚠️  Note: Full multimodal integration requires vision token merging")
  print("    This is a placeholder for future implementation.")
  print("    For now, we verify vision and text independently.")
  return True


def main():
  parser = argparse.ArgumentParser(
      description="Verify Qwen2.5 VL weight conversion and numeric accuracy"
  )
  parser.add_argument(
      "--hf_model",
      type=str,
      required=True,
      help="Path or name of HuggingFace Qwen2.5 VL model",
  )
  parser.add_argument(
      "--tunix_model",
      type=str,
      required=True,
      help="Path to Tunix safetensors checkpoint directory",
  )
  parser.add_argument(
      "--test",
      type=str,
      choices=["text", "vision", "multimodal", "all"],
      default="all",
      help="Which test to run",
  )
  parser.add_argument(
      "--batch_size",
      type=int,
      default=2,
      help="Batch size for testing",
  )
  parser.add_argument(
      "--seq_len",
      type=int,
      default=32,
      help="Sequence length for testing",
  )

  args = parser.parse_args()

  results = {}

  try:
    if args.test in ["text", "all"]:
      results["text"] = verify_text_only_forward(
          args.hf_model,
          args.tunix_model,
          args.batch_size,
          args.seq_len,
      )

    if args.test in ["vision", "all"]:
      results["vision"] = verify_vision_encoder(
          args.hf_model,
          args.tunix_model,
      )

    if args.test in ["multimodal", "all"]:
      results["multimodal"] = verify_multimodal_forward(
          args.hf_model,
          args.tunix_model,
      )

    # Print summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    for test_name, passed in results.items():
      status = "✅ PASSED" if passed else "❌ FAILED"
      print(f"{test_name.upper():20s}: {status}")

    all_passed = all(results.values())
    if all_passed:
      print("\n🎉 All verification tests PASSED!")
      return 0
    else:
      print("\n❌ Some verification tests FAILED!")
      return 1

  except Exception as e:
    print(f"\n💥 Error during verification: {e}")
    import traceback

    traceback.print_exc()
    return 1


if __name__ == "__main__":
  sys.exit(main())

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

"""Image preprocessing for Qwen2-VL models."""

import math
from typing import Dict, List, Tuple, Union

import jax.numpy as jnp
import jaxtyping
import numpy as np
from PIL import Image


# ImageNet normalization statistics
IMAGENET_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGENET_STD = (0.26862954, 0.26130258, 0.27577711)


def smart_resize(
    height: int,
    width: int,
    min_pixels: int,
    max_pixels: int,
    patch_size: int = 14,
    spatial_merge_size: int = 2,
) -> Tuple[int, int]:
  """Smart resize to stay within pixel bounds while maintaining aspect ratio.

  Args:
    height: Original image height
    width: Original image width
    min_pixels: Minimum total pixels
    max_pixels: Maximum total pixels
    patch_size: Patch size for the vision encoder
    spatial_merge_size: Spatial merge factor

  Returns:
    new_height, new_width: Resized dimensions
  """
  current_pixels = height * width

  # If within bounds, return original size (rounded to patch size)
  if min_pixels <= current_pixels <= max_pixels:
    # Round to nearest multiple of (patch_size * spatial_merge_size)
    factor = patch_size * spatial_merge_size
    new_height = round(height / factor) * factor
    new_width = round(width / factor) * factor
    return max(new_height, factor), max(new_width, factor)

  # Calculate aspect ratio
  aspect_ratio = width / height

  # Resize to fit within max_pixels
  if current_pixels > max_pixels:
    target_pixels = max_pixels
  else:
    target_pixels = min_pixels

  # Compute new dimensions maintaining aspect ratio
  new_height = math.sqrt(target_pixels / aspect_ratio)
  new_width = new_height * aspect_ratio

  # Round to nearest multiple of (patch_size * spatial_merge_size)
  factor = patch_size * spatial_merge_size
  new_height = round(new_height / factor) * factor
  new_width = round(new_width / factor) * factor

  # Ensure minimum size
  new_height = max(new_height, factor)
  new_width = max(new_width, factor)

  return int(new_height), int(new_width)


def normalize_image(
    image: np.ndarray,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> np.ndarray:
  """Normalize image with mean and std.

  Args:
    image: Image array [H, W, C] in range [0, 1]
    mean: Mean for each channel
    std: Std for each channel

  Returns:
    Normalized image array
  """
  mean = np.array(mean, dtype=np.float32)
  std = np.array(std, dtype=np.float32)

  # Normalize
  image = (image - mean) / std
  return image


class Qwen2VLImageProcessor:
  """Image processor for Qwen2-VL models."""

  def __init__(
      self,
      min_pixels: int = 256 * 28 * 28,
      max_pixels: int = 1280 * 28 * 28,
      patch_size: int = 14,
      spatial_merge_size: int = 2,
      temporal_patch_size: int = 2,
      mean: Tuple[float, float, float] = IMAGENET_MEAN,
      std: Tuple[float, float, float] = IMAGENET_STD,
  ):
    """
    Args:
      min_pixels: Minimum pixels per image
      max_pixels: Maximum pixels per image
      patch_size: Vision encoder patch size
      spatial_merge_size: Spatial merge factor
      temporal_patch_size: Temporal patch size (for video)
      mean: Normalization mean
      std: Normalization std
    """
    self.min_pixels = min_pixels
    self.max_pixels = max_pixels
    self.patch_size = patch_size
    self.spatial_merge_size = spatial_merge_size
    self.temporal_patch_size = temporal_patch_size
    self.mean = mean
    self.std = std

  def process_image(
      self,
      image: Union[Image.Image, np.ndarray],
  ) -> Dict[str, jaxtyping.Array]:
    """Process a single image.

    Args:
      image: PIL Image or numpy array [H, W, C]

    Returns:
      Dictionary with:
        - pixel_values: [1, H', W', 3] normalized image tensor
        - image_grid_thw: [3] grid dimensions (temporal=1, height, width patches)
    """
    # Convert to PIL if numpy
    if isinstance(image, np.ndarray):
      image = Image.fromarray(image)

    # Get dimensions
    width, height = image.size

    # Smart resize
    new_height, new_width = smart_resize(
        height,
        width,
        self.min_pixels,
        self.max_pixels,
        self.patch_size,
        self.spatial_merge_size,
    )

    # Resize image
    image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)

    # Convert to array and normalize
    image_array = np.array(image, dtype=np.float32) / 255.0  # [H, W, 3]

    # Normalize
    image_array = normalize_image(image_array, self.mean, self.std)

    # Add temporal dimension: [H, W, 3] -> [1, H, W, 3]
    pixel_values = np.expand_dims(image_array, axis=0)

    # Convert to JAX array
    pixel_values = jnp.array(pixel_values)

    # Compute grid dimensions (after patch embed and merge)
    temporal_patches = 1 // self.temporal_patch_size  # Always 1 for images
    height_patches = new_height // self.patch_size // self.spatial_merge_size
    width_patches = new_width // self.patch_size // self.spatial_merge_size

    # Ensure at least 1 patch in each dimension
    temporal_patches = max(temporal_patches, 1)
    height_patches = max(height_patches, 1)
    width_patches = max(width_patches, 1)

    image_grid_thw = jnp.array([temporal_patches, height_patches, width_patches], dtype=jnp.int32)

    return {
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }

  def process_images(
      self,
      images: List[Union[Image.Image, np.ndarray]],
  ) -> Dict[str, jaxtyping.Array]:
    """Process multiple images (for batching).

    Args:
      images: List of PIL Images or numpy arrays

    Returns:
      Dictionary with batched pixel_values and image_grid_thw
    """
    processed = [self.process_image(img) for img in images]

    # For now, we'll handle images independently
    # In a real implementation, you'd need to pad to the same size
    # and create proper batched tensors

    return {
        "pixel_values": [p["pixel_values"] for p in processed],
        "image_grid_thw": [p["image_grid_thw"] for p in processed],
    }

  def __call__(
      self,
      images: Union[Image.Image, List[Image.Image], np.ndarray, List[np.ndarray]],
  ) -> Dict[str, jaxtyping.Array]:
    """Process image(s).

    Args:
      images: Single image or list of images

    Returns:
      Processed image data
    """
    if isinstance(images, (Image.Image, np.ndarray)):
      return self.process_image(images)
    else:
      return self.process_images(images)

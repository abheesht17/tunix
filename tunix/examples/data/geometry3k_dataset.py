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

"""Geometry3K dataset loader for vision-language training."""

import logging
from typing import Any, Dict

from datasets import load_dataset
import grain
import jax.numpy as jnp
from PIL import Image

from tunix.models.qwen2_vl.image_processing import Qwen2VLImageProcessor


# System prompt for geometry problems
SYSTEM_PROMPT = """You are given a geometry problem with an image. \
Analyze the image and the problem carefully, then provide the final answer. \
The answer should be a numerical value, expression, or formula as appropriate."""

# Template for geometry problems (Qwen2-VL format)
TEMPLATE = """<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{problem}<|im_end|>
<|im_start|>assistant
"""


class Geometry3KMapDataset(grain.MapDataset):
  """Grain MapDataset wrapper for Geometry3K with image support."""

  def __init__(
      self,
      split: str = "train",
      image_processor: Qwen2VLImageProcessor | None = None,
      tokenizer=None,
      shuffle_seed: int = 42,
  ):
    """
    Args:
      split: Dataset split ("train", "validation", or "test")
      image_processor: Image processor for vision models
      tokenizer: Optional tokenizer for applying chat templates
      shuffle_seed: Seed for shuffling
    """
    self.split = split
    self.shuffle_seed = shuffle_seed

    # Load from HuggingFace
    logging.info(f"Loading Geometry3K dataset, split: {split}")
    self.hf_dataset = load_dataset("hiyouga/geometry3k", split=split)

    # Create image processor if not provided
    if image_processor is None:
      self.image_processor = Qwen2VLImageProcessor()
    else:
      self.image_processor = image_processor

    self.tokenizer = tokenizer

    # Shuffle dataset
    if shuffle_seed is not None:
      self.hf_dataset = self.hf_dataset.shuffle(seed=shuffle_seed)

  def __len__(self) -> int:
    return len(self.hf_dataset)

  def __getitem__(self, idx: int) -> Dict[str, Any]:
    """Get a single example from the dataset.

    Args:
      idx: Index of the example

    Returns:
      Dictionary containing:
        - prompts: Formatted prompt with <image> token
        - pixel_values: Processed image tensor [1, H, W, 3]
        - image_grid_thw: Image grid dimensions [3]
        - problem: Original problem text
        - answer: Ground truth answer
    """
    sample = self.hf_dataset[idx]

    # Extract fields
    problem = sample["problem"]  # Already contains "<image>" tag
    answer = sample["answer"]
    images = sample["images"]  # List of PIL Images

    # Process first image (Geometry3K has one image per problem)
    image = images[0] if isinstance(images, list) else images
    processed_image = self.image_processor.process_image(image)

    # Apply template
    if self.tokenizer is not None and hasattr(
        self.tokenizer, "apply_chat_template"
    ):
      # Use tokenizer's chat template
      prompt = self.tokenizer.apply_chat_template(
          [
              {"role": "system", "content": SYSTEM_PROMPT},
              {"role": "user", "content": problem},
          ],
          tokenize=False,
          add_generation_prompt=True,
      )
    else:
      # Use default template
      prompt = TEMPLATE.format(system_prompt=SYSTEM_PROMPT, problem=problem)

    return {
        "prompts": prompt,
        "pixel_values": processed_image["pixel_values"],
        "image_grid_thw": processed_image["image_grid_thw"],
        "problem": problem,
        "answer": answer,
    }


def normalize_math_answer(answer: str) -> str:
  """Normalize LaTeX math answers for comparison.

  Args:
    answer: Raw answer string potentially containing LaTeX

  Returns:
    Normalized answer string
  """
  # Remove extra spaces
  answer = " ".join(answer.split())

  # Normalize LaTeX formatting
  answer = answer.replace("\\left", "").replace("\\right", "")
  answer = answer.replace("{ ", "{").replace(" }", "}")
  answer = answer.replace("\\ ", "\\").replace("  ", " ")

  # Strip whitespace
  answer = answer.strip()

  return answer


def extract_answer_from_response(response: str) -> str:
  """Extract the final answer from a model response.

  Args:
    response: Model generated response

  Returns:
    Extracted answer string
  """
  # Simple extraction - look for the last line or number
  # TODO: Implement more sophisticated answer extraction
  lines = response.strip().split("\n")

  # Return last non-empty line
  for line in reversed(lines):
    line = line.strip()
    if line:
      return line

  return response.strip()


def create_dataset(
    data_source: str,
    dataset: str,
    tokenizer=None,
    split: str = "train",
    image_processor: Qwen2VLImageProcessor | None = None,
) -> Geometry3KMapDataset:
  """Creates a Geometry3K dataset.

  Args:
    data_source: The source of dataset. For Geometry3K, use "hf" (HuggingFace).
    dataset: The name of the dataset. Should be "geometry3k" or
      "hiyouga/geometry3k".
    tokenizer: The tokenizer to use for processing prompts. If no tokenizer is
      provided, the fixed template is used.
    split: The dataset split to use (e.g., "train", "validation", "test").
    image_processor: Optional image processor. If not provided, a default
      Qwen2VLImageProcessor will be created.

  Returns:
    A Geometry3KMapDataset instance.

  Raises:
    ValueError: If the dataset is not supported.
  """
  if data_source == "hf" and ("geometry3k" in dataset.lower()):
    ds = Geometry3KMapDataset(
        split=split,
        image_processor=image_processor,
        tokenizer=tokenizer,
    )
    return ds
  else:
    raise ValueError(
        f"Unsupported combination of dataset='{dataset}' and"
        f" data_source='{data_source}'. For Geometry3K, use data_source='hf'"
        f" and dataset='geometry3k' or 'hiyouga/geometry3k'."
    )

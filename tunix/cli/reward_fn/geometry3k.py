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

"""Reward function for Geometry3K dataset."""

import re
from typing import List

import jax.numpy as jnp

from tunix.examples.data.geometry3k_dataset import normalize_math_answer, extract_answer_from_response


def geometry3k_reward_fn(
    responses: List[str],
    ground_truth_answers: List[str],
    **kwargs,
) -> jnp.ndarray:
  """Reward function for Geometry3K geometry problems.

  Args:
    responses: List of model generated responses
    ground_truth_answers: List of ground truth answers
    **kwargs: Additional arguments (unused)

  Returns:
    Array of rewards, +1 for correct answer, -1 for incorrect
  """
  rewards = []

  for response, gt_answer in zip(responses, ground_truth_answers):
    # Extract answer from response
    predicted_answer = extract_answer_from_response(response)

    # Normalize both answers
    pred_normalized = normalize_math_answer(predicted_answer)
    gt_normalized = normalize_math_answer(gt_answer)

    # Compare
    if pred_normalized == gt_normalized:
      reward = 1.0
    else:
      # Try numerical comparison as fallback
      try:
        pred_num = float(pred_normalized)
        gt_num = float(gt_normalized)
        # Allow small numerical error
        if abs(pred_num - gt_num) < 1e-6:
          reward = 1.0
        else:
          reward = -1.0
      except (ValueError, TypeError):
        # Not a simple number, use string comparison
        reward = -1.0

    rewards.append(reward)

  return jnp.array(rewards, dtype=jnp.float32)

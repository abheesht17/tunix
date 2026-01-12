#!/usr/bin/env python3
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

"""Example script for training Qwen2.5-VL-7B on Geometry3K using GRPO.

This replicates the EasyR1 example:
https://github.com/hiyouga/EasyR1#tutorial-run-qwen25-vl-grpo-on-geometry3k-dataset-in-just-3-steps

Usage:
  python scripts/grpo_qwen2_5_vl_geometry3k.py \\
    --config configs/qwen2_5_vl_geometry3k.yaml \\
    --mesh_shape "(1,8)"
"""

import os
from absl import app, flags

import jax
from jax import sharding as shd

from tunix.cli.config import HyperParameters
from tunix.examples.data import geometry3k_dataset
from tunix.cli.reward_fn import geometry3k as reward_fn
from tunix.models import automodel
from tunix.rl.grpo import grpo_learner

# Flags
FLAGS = flags.FLAGS
flags.DEFINE_string(
    "config",
    "configs/qwen2_5_vl_geometry3k.yaml",
    "Path to configuration file",
)
flags.DEFINE_string(
    "model_path",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "HuggingFace model path or local directory",
)
flags.DEFINE_string(
    "mesh_shape",
    "(1,8)",
    "Mesh shape for distributed training, e.g., '(1,8)' for 8 TPUs",
)
flags.DEFINE_string(
    "mesh_axis_names",
    "('fsdp','tp')",
    "Mesh axis names",
)
flags.DEFINE_integer(
    "num_epochs",
    15,
    "Number of training epochs",
)


def create_mesh(mesh_shape_str: str, axis_names_str: str):
  """Create JAX mesh from string configuration.

  Args:
    mesh_shape_str: String like "(1,8)"
    axis_names_str: String like "('fsdp','tp')"

  Returns:
    jax.sharding.Mesh
  """
  # Parse mesh shape
  mesh_shape = eval(mesh_shape_str)  # e.g., (1, 8)

  # Parse axis names
  axis_names = eval(axis_names_str)  # e.g., ('fsdp', 'tp')

  # Create mesh
  devices = jax.devices()
  if len(devices) != mesh_shape[0] * mesh_shape[1]:
    raise ValueError(
        f"Number of devices ({len(devices)}) does not match "
        f"mesh shape {mesh_shape} (product: {mesh_shape[0] * mesh_shape[1]})"
    )

  mesh = shd.Mesh(devices, axis_names)
  return mesh


def main(argv):
  del argv  # Unused

  # Load configuration
  print(f"Loading configuration from: {FLAGS.config}")
  config = HyperParameters.from_yaml(FLAGS.config)

  # Override config with flags if provided
  if FLAGS.model_path:
    config.model.model_name = FLAGS.model_path

  # Create mesh
  print(f"Creating mesh: shape={FLAGS.mesh_shape}, axes={FLAGS.mesh_axis_names}")
  mesh = create_mesh(FLAGS.mesh_shape, FLAGS.mesh_axis_names)

  print(f"Available devices: {jax.devices()}")
  print(f"Mesh: {mesh}")

  # Load model
  print(f"Loading model: {config.model.model_name}")
  model = automodel.AutoModel.from_pretrained(
      config.model.model_name,
      model_source=automodel.ModelSource.HUGGINGFACE,
      mesh=mesh,
  )

  # Load datasets
  print("Loading Geometry3K dataset...")
  train_dataset = geometry3k_dataset.create_dataset(
      data_source="hf",
      dataset="hiyouga/geometry3k",
      split="train",
      tokenizer=None,  # Will be set from model
  )

  val_dataset = geometry3k_dataset.create_dataset(
      data_source="hf",
      dataset="hiyouga/geometry3k",
      split="test",
      tokenizer=None,
  )

  print(f"Train dataset size: {len(train_dataset)}")
  print(f"Val dataset size: {len(val_dataset)}")

  # Create GRPO learner
  print("Creating GRPO learner...")
  learner = grpo_learner.GRPOLearner(
      model=model,
      config=config.grpo,
      reward_fn=reward_fn.geometry3k_reward_fn,
      mesh=mesh,
  )

  # Train
  print(f"Starting training for {FLAGS.num_epochs} epochs...")
  learner.train(
      train_dataset=train_dataset,
      val_dataset=val_dataset,
      num_epochs=FLAGS.num_epochs,
      save_dir=config.training.checkpoint_dir,
  )

  print("Training complete!")
  print(f"Checkpoints saved to: {config.training.checkpoint_dir}")


if __name__ == "__main__":
  app.run(main)

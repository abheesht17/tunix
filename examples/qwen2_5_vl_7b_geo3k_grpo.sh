#!/bin/bash
# Training script for Qwen2.5-VL-7B on Geometry3K using GRPO
# This replicates the EasyR1 example: https://github.com/hiyouga/EasyR1

set -x

# Model configuration
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"

# Run GRPO training
python3 -m tunix.cli.grpo_main \
  config=configs/qwen2_5_vl_geometry3k.yaml \
  model.model_name=${MODEL_PATH} \
  data.dataset_name=hiyouga/geometry3k \
  data.dataset_split_train=train \
  data.dataset_split_val=test \
  training.experiment_name=qwen2_5_vl_7b_geo_grpo \
  mesh.shape="(1,8)" \
  mesh.axis_names="('fsdp','tp')"

echo "Training complete!"
echo "To merge the checkpoint for inference, run:"
echo "python3 scripts/model_merger.py --checkpoint_dir checkpoints/qwen2_5_vl_geometry3k"

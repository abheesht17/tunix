# Vision-Language Model Support in Tunix

Tunix now supports vision-language models for multimodal post-training, starting with Qwen2.5-VL.

## Overview

This implementation replicates the [EasyR1 Qwen2.5-VL GRPO example](https://github.com/hiyouga/EasyR1#tutorial-run-qwen25-vl-grpo-on-geometry3k-dataset-in-just-3-steps) in JAX/Flax, enabling TPU-optimized training of vision-language models.

## Supported Models

- **Qwen2.5-VL-7B-Instruct**: Vision-language model with dynamic resolution support
- **Qwen2.5-VL-3B-Instruct**: Smaller vision-language model (coming soon)

## Features

- ✅ Native JAX/Flax implementation of Qwen2-VL vision encoder
- ✅ Dynamic image resolution handling (256-1280 visual tokens)
- ✅ Vision-language connector for multimodal fusion
- ✅ GRPO training support for vision-language tasks
- ✅ Geometry3K dataset loader with image processing
- ✅ Efficient image preprocessing with smart resizing

## Quick Start

### 1. Installation

```bash
# Install Tunix with vision dependencies
pip install -e ".[dev]"

# Additional dependencies for vision
pip install Pillow datasets
```

### 2. Run GRPO Training on Geometry3K

The simplest way to get started:

```bash
bash examples/qwen2_5_vl_7b_geo3k_grpo.sh
```

Or use the Python script for more control:

```bash
python scripts/grpo_qwen2_5_vl_geometry3k.py \
  --config configs/qwen2_5_vl_geometry3k.yaml \
  --mesh_shape "(1,8)" \
  --num_epochs 15
```

### 3. Configuration

The configuration file [configs/qwen2_5_vl_geometry3k.yaml](../configs/qwen2_5_vl_geometry3k.yaml) contains all training parameters:

```yaml
# Model
model:
  model_name: "Qwen/Qwen2.5-VL-7B-Instruct"
  vision:
    min_pixels: 262144  # 256 * 28 * 28
    max_pixels: 1024000  # 1280 * 28 * 28
    freeze_vision_tower: false

# Data
data:
  dataset_name: "hiyouga/geometry3k"
  batch_size: 16

# GRPO
grpo:
  num_generations: 2
  beta: 0.01
  epsilon: 0.2
```

Key parameters to adjust:
- `batch_size`: Reduce if you encounter OOM errors
- `min_pixels` / `max_pixels`: Control image resolution and memory usage
- `freeze_vision_tower`: Set to `true` to only train the language model
- `mesh.shape`: Adjust based on your hardware (TPU/GPU count)

## Architecture

### Vision Encoder

The Qwen2-VL vision encoder is a Vision Transformer (ViT) with:
- **32 transformer layers** (for 7B model)
- **Dynamic resolution support**: Images are resized to fit within pixel bounds while maintaining aspect ratio
- **2D RoPE**: Rotary position embeddings for spatial awareness
- **Spatial merging**: 2x2 patch merging for efficiency

### Vision-Language Connector

A simple but effective MLP-based connector:
1. Layer normalization on vision features
2. Linear projection to text hidden size
3. GELU activation
4. Final linear projection

### Image Processing

Smart image preprocessing:
- Automatic resizing to stay within `min_pixels` and `max_pixels`
- Aspect ratio preservation
- Rounding to patch size multiples (14 x 2 = 28)
- ImageNet normalization

## Geometry3K Dataset

The Geometry3K dataset contains:
- **3,001 geometry problems** with images
- **Fields**: `images`, `problem`, `answer`
- **Splits**: train (2,100), validation (300), test (601)

### Example

```python
{
  "images": [<PIL.Image>],
  "problem": "<image>Find x.",
  "answer": "3"
}
```

Answers include numbers, radicals, fractions, and expressions like `"2 \\sqrt { 221 }"`.

## Reward Function

The Geometry3K reward function (`tunix/cli/reward_fn/geometry3k.py`):
- Normalizes LaTeX formatting
- Compares predicted vs ground truth answers
- Returns +1 for correct, -1 for incorrect
- Handles numerical comparison with tolerance

## Memory Optimization

### Tips for Large Images

1. **Reduce max_pixels**:
```yaml
vision:
  max_pixels: 512000  # Smaller resolution
```

2. **Enable gradient checkpointing**:
```yaml
training:
  gradient_checkpointing: true
```

3. **Freeze vision tower**:
```yaml
vision:
  freeze_vision_tower: true
```

4. **Use FSDP with CPU offloading**:
```yaml
fsdp:
  cpu_offload_ref: true
```

5. **Reduce batch size**:
```yaml
data:
  batch_size: 8  # Or even smaller
```

## Advanced Usage

### Custom Vision-Language Datasets

To use your own dataset:

```python
from tunix.examples.data import geometry3k_dataset
from tunix.models.qwen2_vl.image_processing import Qwen2VLImageProcessor

# Create image processor
image_processor = Qwen2VLImageProcessor(
    min_pixels=256 * 28 * 28,
    max_pixels=1280 * 28 * 28,
)

# Create custom dataset
class MyVLDataset(grain.MapDataset):
    def __getitem__(self, idx):
        # Load your data
        image = ...  # PIL Image
        text = ...   # Text prompt with <image> token

        # Process image
        processed = image_processor.process_image(image)

        return {
            "prompts": text,
            "pixel_values": processed["pixel_values"],
            "image_grid_thw": processed["image_grid_thw"],
            "answer": ...,
        }
```

### Custom Reward Functions

Create your own reward function:

```python
# my_reward_fn.py
import jax.numpy as jnp

def my_reward_fn(responses, ground_truths, **kwargs):
    rewards = []
    for response, gt in zip(responses, ground_truths):
        # Your logic here
        reward = 1.0 if response == gt else -1.0
        rewards.append(reward)
    return jnp.array(rewards)
```

Update config:

```yaml
reward:
  reward_fn_module: "path.to.my_reward_fn"
  reward_fn_name: "my_reward_fn"
```

## Performance Benchmarks

Expected performance on Geometry3K (preliminary):

| Model | Hardware | Batch Size | Training Time | Accuracy |
|-------|----------|------------|---------------|----------|
| Qwen2.5-VL-7B | 8x TPU v4 | 16 | ~2 hours | TBD |

*Note: Benchmarks to be updated after initial training runs*

## Troubleshooting

### OOM Errors

If you encounter out-of-memory errors:

1. Reduce `max_pixels` to limit image sizes
2. Reduce `batch_size`
3. Enable `gradient_checkpointing`
4. Freeze the vision tower
5. Use smaller sequence lengths

### Vision Encoder Not Training

Make sure:
- `freeze_vision_tower: false` in config
- Gradients are flowing (check with gradient norms)
- Learning rate is appropriate (try 1e-6 to 1e-5)

### Image Not Loading

Ensure:
- Dataset contains valid PIL Images
- `<image>` token is in the prompt
- Image processor is configured correctly

## Limitations

Current limitations:
- ⚠️ Multi-image support not fully implemented
- ⚠️ vLLM rollout may not support vision models on TPU
- ⚠️ Model conversion from HuggingFace needs testing
- ⚠️ Video inputs not yet supported

## Roadmap

Planned improvements:
- [ ] Full multi-image support
- [ ] Video input support
- [ ] Additional vision-language models (PaliGemma, LLaVA, etc.)
- [ ] vLLM/SGLang integration for vision models
- [ ] Quantization support for vision encoder
- [ ] More vision-language datasets

## References

- [Qwen2-VL Paper](https://arxiv.org/abs/2409.12191)
- [Qwen2.5-VL Technical Report](https://arxiv.org/pdf/2502.13923)
- [EasyR1 Repository](https://github.com/hiyouga/EasyR1)
- [Geometry3K Dataset](https://huggingface.co/datasets/hiyouga/geometry3k)

## Citation

If you use this implementation, please cite:

```bibtex
@misc{tunix2025,
  title={Tunix},
  author={Bao, Tianshu and Wang, Lance and Sharma, Abheesht and others},
  year={2025},
  howpublished={\url{https://github.com/google/tunix}},
}

@article{qwen2vl2024,
  title={Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution},
  author={Wang, Peng and others},
  journal={arXiv preprint arXiv:2409.12191},
  year={2024}
}
```

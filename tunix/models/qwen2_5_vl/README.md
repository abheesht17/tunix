# Qwen2.5 VL Implementation

This directory contains the Tunix JAX/Flax implementation of Qwen2.5 VL (Vision-Language) 3B model.

## Model Architecture

Qwen2.5 VL is a multimodal model that combines:
- **Vision Encoder**: Processes images/videos using 3D convolution patch embedding and vision transformer
- **Text Decoder**: Transformer-based language model with GQA (Grouped Query Attention)
- **Multimodal Integration**: 3D RoPE (Rotary Position Embedding) for joint vision-text understanding

### Key Components

1. **VisionPatchEmbed**: 3D convolutional patch embedding
   - Converts raw images/videos into patch embeddings
   - Kernel: [temporal_patch_size, patch_size, patch_size]
   - Stride equals kernel size (non-overlapping patches)

2. **VisionTransformer**: Vision encoder with 32 transformer blocks
   - Hidden size: 3584
   - Number of heads: 16
   - MLP intermediate size: 3420
   - Uses GELU activation

3. **VisionPatchMerger**: Reduces vision sequence length
   - Merges spatial patches by factor of 2x2
   - Two-layer MLP with GELU activation
   - RMSNorm before merging

4. **TextDecoderLayer**: Language model decoder
   - 80 layers
   - Hidden dimension: 8192
   - MLP dimension: 29568
   - 64 attention heads, 8 KV heads (GQA)
   - SiLU activation in MLP

5. **Multimodal RoPE**: 3D rotary position embedding
   - Splits embedding dimensions into temporal, height, width components
   - Section sizes: (16, 24, 24) for 3B model
   - Enables position-aware multimodal attention

## Model Configuration

### Qwen2.5 VL 3B

**Text Config:**
- Layers: 80
- Vocab size: 152,064
- Embedding dim: 8,192
- Hidden dim: 29,568
- Attention heads: 64
- KV heads: 8 (GQA ratio: 8:1)
- RoPE theta: 1,000,000
- RMSNorm epsilon: 1e-5

**Vision Config:**
- Depth: 32 transformer blocks
- Hidden size: 3,584
- Attention heads: 16
- MLP intermediate: 3,420
- Patch size: 14x14
- Temporal patch size: 2
- Spatial merge size: 2
- RMSNorm epsilon: 1e-6

**Special Tokens:**
- Image token: 151,655
- Video token: 151,656
- Vision start token: 151,652
- Vision end token: 151,653

## File Structure

```
qwen2_5_vl/
├── __init__.py          # Package initialization
├── model.py             # Core model implementation
├── params.py            # Weight conversion from PyTorch
├── verify_weights.py    # Verification script
└── README.md            # This file
```

## Usage

### Loading the Model

```python
from tunix.models.qwen2_5_vl import ModelConfig, create_model_from_safe_tensors
import jax.numpy as jnp

# Create configuration
config = ModelConfig.qwen2p5_vl_3b()

# Load from safetensors
model = create_model_from_safe_tensors(
    file_dir="/path/to/safetensors",
    config=config,
    dtype=jnp.bfloat16,
)
```

### Text-Only Forward Pass

```python
import jax.numpy as jnp

batch_size = 2
seq_len = 32

# Input tokens
input_tokens = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
positions = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)

# Causal attention mask
attention_mask = jnp.tril(jnp.ones((batch_size, seq_len, seq_len), dtype=jnp.bool))

# Forward pass
logits, cache = model(
    input_tokens=input_tokens,
    positions=positions,
    cache=None,
    attention_mask=attention_mask,
    pixel_values=None,
    image_grid_thw=None,
)
```

### Multimodal Forward Pass (with Vision)

```python
import jax.numpy as jnp

# Text inputs
batch_size = 1
seq_len = 64
input_tokens = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
positions = jnp.arange(seq_len)[None, :]
attention_mask = jnp.tril(jnp.ones((batch_size, seq_len, seq_len), dtype=jnp.bool))

# Vision inputs (single image)
num_frames = 1
height, width = 224, 224
channels = 3
pixel_values = jnp.ones((batch_size, num_frames, height, width, channels))

# Grid dimensions after patching
patch_size = config.vision_config.patch_size
temporal_patch_size = config.vision_config.temporal_patch_size
grid_t = num_frames // temporal_patch_size
grid_h = height // patch_size
grid_w = width // patch_size
image_grid_thw = jnp.array([[grid_t, grid_h, grid_w]], dtype=jnp.int32)

# Multimodal forward pass
logits, cache = model(
    input_tokens=input_tokens,
    positions=positions,
    cache=None,
    attention_mask=attention_mask,
    pixel_values=pixel_values,
    image_grid_thw=image_grid_thw,
)
```

## Weight Conversion

Weights are converted from HuggingFace PyTorch format to Tunix JAX format using the mapping in `params.py`.

### Key Conversions:

1. **Conv3D weights**: `[C_out, C_in, T, H, W]` → `[T, H, W, C_in, C_out]`
2. **Linear weights**: Transpose `[out, in]` → `[in, out]`
3. **Attention projections**: Reshape into `[embed_dim, num_heads, head_dim]`
4. **Vision QKV**: Combined projection weight handling

## Verification

Use the provided verification script to compare outputs with HuggingFace:

```bash
python -m tunix.models.qwen2_5_vl.verify_weights \
    --hf_model "Qwen/Qwen2.5-VL-3B" \
    --tunix_model "/path/to/tunix/checkpoint" \
    --test all
```

### Verification Tests:

1. **Text-only**: Verifies text decoder and attention
2. **Vision encoder**: Verifies vision transformer pipeline
3. **Multimodal**: Verifies full vision-text integration

## Implementation Notes

### Differences from HuggingFace

1. **3D Convolution**: Implemented using `jax.lax.conv_general_dilated` instead of `nn.Conv3d`
2. **Activation Checkpointing**: Configurable via `RematConfig`
3. **Sharding**: Built-in support for FSDP and tensor parallelism
4. **KV Cache**: JAX-native cache implementation with dynamic updates

### Current Limitations

1. **Vision-Text Integration**: Basic implementation provided; full placeholder token replacement needs enhancement
2. **Flash Attention**: Standard attention implementation; flash attention can be added
3. **Window Attention**: Full attention used; window patterns can be optimized

### Performance Optimizations

The implementation includes:
- Configurable sharding for distributed training
- Activation checkpointing (remat) support
- Efficient KV cache for generation
- JAX JIT compilation compatible

## Testing

Run the verification script to ensure correctness:

```bash
# Test text decoder only
python -m tunix.models.qwen2_5_vl.verify_weights \
    --hf_model "Qwen/Qwen2.5-VL-3B" \
    --tunix_model "/path/to/checkpoint" \
    --test text

# Test vision encoder only
python -m tunix.models.qwen2_5_vl.verify_weights \
    --hf_model "Qwen/Qwen2.5-VL-3B" \
    --tunix_model "/path/to/checkpoint" \
    --test vision

# Test everything
python -m tunix.models.qwen2_5_vl.verify_weights \
    --hf_model "Qwen/Qwen2.5-VL-3B" \
    --tunix_model "/path/to/checkpoint" \
    --test all
```

## References

- [HuggingFace Qwen2.5 VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B)
- [Qwen2.5 VL Paper](https://arxiv.org/abs/2409.12191)
- [Original Implementation](https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen2_5_vl)

## Citation

```bibtex
@article{qwen2.5vl,
  title={Qwen2.5-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution},
  author={Qwen Team},
  journal={arXiv preprint arXiv:2409.12191},
  year={2024}
}
```

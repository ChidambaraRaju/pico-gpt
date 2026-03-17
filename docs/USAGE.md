# Pico-GPT Usage Guide

## Installation

```bash
git clone <repo-url>
cd pico-gpt
pip install -r requirements.txt
```

## Hardware Requirements

- **GPU:** NVIDIA A100 (20GB) or equivalent recommended
- **RAM:** 32GB minimum
- **Storage:** 50GB for dataset and checkpoints
- **OS:** Linux (CUDA support)

## Preparing the Dataset

The dataset preprocessing script downloads and tokenizes the OpenWebText dataset:

```bash
python scripts/prepare_data.py \
    --output-dir data \
    --shard-size 5000000 \
    --total-tokens 100000000 \
    --val-tokens 5000000
```

This creates:
- `data/train_*.bin` - Training shards
- `data/val.bin` - Validation data
- `data/preprocessing_state.json` - Resume state

The script supports resume capability. If interrupted, run again to continue from where it stopped.

## Training

### Basic Training

```bash
python scripts/train.py \
    --data-dir data \
    --output-dir checkpoints
```

### With Custom Configuration

```bash
python scripts/train.py \
    --data-dir data \
    --output-dir checkpoints \
    --max-steps 200000 \
    --resume checkpoints/checkpoint_50000.pt
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | `data` | Directory containing binary shards |
| `--output-dir` | `checkpoints` | Output directory for checkpoints |
| `--resume` | `None` | Path to checkpoint to resume from |
| `--max-steps` | `200000` | Maximum training steps |

## Checkpoints

The trainer saves two types of checkpoints:

1. **Best Model:** `best_model.pt` (lowest validation loss)
2. **Periodic:** `checkpoint_*.pt` (every 1000 steps)

### Resuming from Checkpoint

```bash
python scripts/train.py --resume checkpoints/best_model.pt
```

## Generation

### Basic Generation

```bash
python scripts/generate.py \
    --model checkpoints/best_model.pt \
    --prompt "The future of AI is"
```

### Custom Parameters

```bash
python scripts/generate.py \
    --model checkpoints/best_model.pt \
    --prompt "Explain quantum computing" \
    --max-tokens 200 \
    --temperature 0.7
```

### Generation Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `checkpoints/best_model.pt` | Path to model checkpoint |
| `--prompt` | (default prompt) | Input text prompt |
| `--max-tokens` | `100` | Maximum tokens to generate |
| `--temperature` | `0.8` | Sampling temperature (lower = more deterministic) |

## Exporting to Hugging Face

### Export Only

```bash
python scripts/export_hf.py \
    --checkpoint checkpoints/best_model.pt \
    --output hf_model
```

This creates:
- `hf_model/model.safetensors` - Model weights
- `hf_model/config.json` - Model configuration
- `hf_model/tokenizer_config.json` - Tokenizer metadata
- `hf_model/README.md` - Model card

### Export and Upload

```bash
python scripts/export_hf.py \
    --checkpoint checkpoints/best_model.pt \
    --output hf_model \
    --upload username/pico-gpt \
    --private
```

Requires Hugging Face authentication:

```bash
huggingface-cli login
```

## Training Tips

1. **Monitor Loss:** Watch for stable decrease in training loss
2. **Validation Check:** Validation loss should track with training loss
3. **GPU Utilization:** Use `nvidia-smi` to monitor GPU usage
4. **Disk Space:** Ensure sufficient space for checkpoints (~5GB)

## Troubleshooting

### Out of Memory

Reduce batch size:

```bash
python scripts/train.py --data-dir data --output-dir checkpoints
# Edit scripts/train.py to reduce batch_size and micro_batch_size
```

### CUDA Out of Memory

If using a smaller GPU, modify `TrainingConfig`:

```python
batch_size = 32
micro_batch_size = 4
```

### Dataset Not Found

Ensure you've run the preprocessing script:

```bash
python scripts/prepare_data.py
```

### Checkpoint Corrupted

Delete the corrupted checkpoint and resume from a previous one:

```bash
rm checkpoints/checkpoint_50000.pt
python scripts/train.py --resume checkpoints/checkpoint_40000.pt
```

## Performance

Expected training performance on A100:

| Configuration | Tokens/sec | Steps/sec | Training Time |
|---------------|------------|-----------|---------------|
| Batch 64, BF16 | ~200K | ~3 | ~22 hours |
| Batch 32, BF16 | ~150K | ~2.5 | ~30 hours |
| Batch 64, FP32 | ~100K | ~1.6 | ~45 hours |

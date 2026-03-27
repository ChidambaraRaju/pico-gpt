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
    --total-tokens 1000000000 \
    --val-tokens 50000000
```

For faster testing with a smaller dataset:

```bash
python scripts/prepare_data.py \
    --output-dir data \
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
    --lr 3e-4 \
    --checkpoint-interval 1000
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | `data` | Directory containing binary shards |
| `--output-dir` | `checkpoints` | Output directory for checkpoints |
| `--max-steps` | `200000` | Maximum training steps |
| `--lr` | `3e-4` | Learning rate |
| `--checkpoint-interval` | `1000` | Save checkpoint every N steps |

> **Note:** Checkpoint resumption is not yet implemented. Training always starts from scratch.

## Checkpoints

The trainer saves the following outputs:

1. **PyTorch Checkpoints:** `checkpoint_step_N.pt` (every N steps, default 1000)
2. **Safetensors Export:** `model_step_N.safetensors` (for Hugging Face compatibility)
3. **Training Log:** `training_log.csv` (step, loss, elapsed time)

### Checkpoint Contents

Each PyTorch checkpoint contains:
- `model_state_dict`: Model weights
- `config`: Model configuration
- `step`: Current training step
- `training_config`: Training hyperparameters (final checkpoint only)

## Generation

### Basic Generation

```bash
python scripts/generate.py \
    --model checkpoints/checkpoint_step_1000.pt \
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
| `--prompt` | `The future of artificial intelligence is` | Input text prompt |
| `--max-tokens` | `100` | Maximum tokens to generate |
| `--temperature` | `0.8` | Sampling temperature (lower = more deterministic) |

## Exporting to Hugging Face

### Export Only

```bash
python scripts/export_hf.py \
    --checkpoint checkpoints/checkpoint_step_100000.pt \
    --output hf_model \
    --training-log checkpoints/training_log.csv
```

This creates:
- `hf_model/model.safetensors` - Model weights
- `hf_model/config.json` - Model configuration
- `hf_model/training_config.json` - Training hyperparameters
- `hf_model/training_log.csv` - Training metrics (if provided)
- `hf_model/samples.txt` - Generated text samples
- `hf_model/tokenizer_config.json` - Tokenizer metadata
- `hf_model/special_tokens_map.json` - Special tokens
- `hf_model/README.md` - Model card

### Export and Upload

```bash
python scripts/export_hf.py \
    --checkpoint checkpoints/checkpoint_step_100000.pt \
    --output hf_model \
    --training-log checkpoints/training_log.csv \
    --upload username/pico-gpt \
    --private
```

### Export Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | Required | Path to model checkpoint |
| `--output` | `hf_model` | Output directory |
| `--training-log` | `None` | Path to training_log.csv file |
| `--upload` | `None` | Upload to Hugging Face (repo_id) |
| `--private` | `False` | Make repository private |

Requires Hugging Face authentication:

```bash
huggingface-cli login
```

## Training Tips

1. **Monitor Loss:** Watch for stable decrease in training loss
2. **GPU Utilization:** Use `nvidia-smi` to monitor GPU usage
3. **Disk Space:** Ensure sufficient space for checkpoints (~5GB)
4. **Training Log:** Check `training_log.csv` for loss progression

## Troubleshooting

### Out of Memory

Reduce batch size by modifying the training configuration:

```bash
# Use a smaller batch size
python scripts/train.py --data-dir data --output-dir checkpoints
```

If using a smaller GPU, modify `TrainingConfig` in `pico_gpt/config.py`:

### CUDA Out of Memory

If using a smaller GPU, reduce the batch size in the `TrainingConfig`:

```python
batch_size = 32
```

### Dataset Not Found

Ensure you've run the preprocessing script:

```bash
python scripts/prepare_data.py
```

### Checkpoint Corrupted

If a checkpoint is corrupted, you can use an earlier checkpoint or restart training:

```bash
# Use an earlier valid checkpoint for generation
python scripts/generate.py --model checkpoints/checkpoint_step_40000.pt

# Or restart training from scratch
python scripts/train.py --data-dir data --output-dir checkpoints_new
```

## Performance

Expected training performance on A100:

| Configuration | Tokens/sec | Steps/sec | Training Time |
|---------------|------------|-----------|---------------|
| Batch 64, FP32 | ~100K | ~1.6 | ~45 hours |
| Batch 32, FP32 | ~75K | ~1.2 | ~60 hours |

> **Note:** Performance metrics are estimates. Actual performance depends on hardware and dataset characteristics.

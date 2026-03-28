"""
Export utilities for Hugging Face Hub.

Reference: @instructions/07_huggingface_export.md
"""

import json
from pathlib import Path
from typing import Optional
import torch
import tiktoken
from safetensors.torch import save_file

from pico_gpt.model import GPT
from pico_gpt.config import ModelConfig
from pico_gpt.tokenizer_utils import export_tokenizer_metadata


def generate_samples(
    model: GPT,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 100,
    temperature: float = 0.8,
) -> list[str]:
    """
    Generate text samples using the trained model.

    Args:
        model: Trained GPT model
        tokenizer: tiktoken tokenizer
        prompts: List of prompts to generate from
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        List of generated texts
    """
    model.eval()
    samples = []
    eos_token_id = tokenizer.eot_token  # Get EOS token from tiktoken

    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            # Encode prompt
            tokens = tokenizer.encode(prompt)
            # Keep last context_length tokens (handles truncation automatically)
            tokens = tokens[-model.config.context_length:]
            idx = torch.tensor([tokens], dtype=torch.long)

            # Generate with EOS token stopping
            generated = model.generate(
                idx,
                max_new_tokens,
                temperature=temperature,
                eos_token_id=eos_token_id,
            )

            # Decode
            generated_tokens = generated[0].tolist()
            generated_text = tokenizer.decode(generated_tokens)

            # Remove prompt from generated text for cleaner output
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()

            samples.append(f"### Sample {i+1}\n**Prompt:** {prompt}\n**Generated:** {generated_text}\n")

    return samples


def export_to_huggingface(
    checkpoint_path: str,
    output_dir: str,
    training_log_path: Optional[str] = None,
) -> None:
    """
    Export model to Hugging Face format.

    Args:
        checkpoint_path: Path to trained model checkpoint
        output_dir: Output directory for Hugging Face model
        training_log_path: Path to training_log.csv file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint_path_obj = Path(checkpoint_path)

    # Handle different checkpoint types
    if checkpoint_path_obj.suffix == ".safetensors":
        # Load safetensors file
        from safetensors.torch import load_file
        state_dict = load_file(str(checkpoint_path_obj))
        config = ModelConfig()
        training_config = {}
        # Create model
        model = GPT(config)
        model.load_state_dict(state_dict)
    else:
        # Load PyTorch checkpoint
        checkpoint = torch.load(str(checkpoint_path_obj), map_location="cpu", weights_only=False)
        # Get config
        if "config" in checkpoint:
            config = checkpoint["config"]
        else:
            config = ModelConfig()
        # Get training config if available
        training_config = checkpoint.get("training_config", {})
        # Load model weights
        model = GPT(config)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

    print(f"Exporting model to {output_path}...")

    # Save model weights in safetensors format
    state_dict = model.state_dict()
    save_file(state_dict, output_path / "model.safetensors")

    # Create model config.json
    config_dict = {
        "model_type": "custom_gpt",
        "vocab_size": config.vocab_size,
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "n_embd": config.n_embd,
        "context_length": config.context_length,
        "dropout": config.dropout,
        "bias": config.bias,
        "ffn_dim": config.ffn_dim,
    }

    with open(output_path / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Export training config (Gap #12)
    if training_config:
        with open(output_path / "training_config.json", "w") as f:
            json.dump(training_config, f, indent=2)
        print(f"Saved training config: training_config.json")

    # Copy training log if provided (Gap #10)
    if training_log_path:
        log_src = Path(training_log_path)
        if log_src.exists():
            import shutil
            shutil.copy(log_src, output_path / "training_log.csv")
            print(f"Copied training log: training_log.csv")

    # Generate samples (Gap #10)
    print("Generating sample texts...")
    tokenizer = tiktoken.get_encoding("gpt2")

    prompts = [
        "The future of artificial intelligence is",
        "Once upon a time",
        "The best way to learn programming is",
        "In the field of machine learning",
        "One of the most important concepts is",
    ]

    samples = generate_samples(
        model, tokenizer, prompts,
        max_new_tokens=50,
        temperature=0.8,
    )

    samples_file = output_path / "samples.txt"
    with open(samples_file, "w") as f:
        f.write("Generated Samples\n")
        f.write("=" * 50 + "\n\n")
        f.write("\n".join(samples))
    print(f"Saved samples: samples.txt")

    # Export tokenizer metadata
    export_tokenizer_metadata(str(output_path), model_max_length=config.context_length)

    # Create model card
    create_model_card(output_path, config, checkpoint_path, training_config)

    print(f"\nExport complete! Model saved to {output_path}")
    print("\nTo upload to Hugging Face:")
    print(f"  huggingface-cli upload-model {output_path}")


def create_model_card(
    output_path: Path,
    config: ModelConfig,
    checkpoint_path: str,
    training_config: dict,
) -> None:
    """Create model card README.md."""

    # Extract training info
    lr = training_config.get("learning_rate", "3e-4")
    weight_decay = training_config.get("weight_decay", "0.1")
    max_steps = training_config.get("max_steps", "N/A")
    final_loss = training_config.get("final_loss", "N/A")
    training_time = training_config.get("training_time_seconds", "N/A")

    # Calculate parameter count
    param_count = config.n_embd * config.vocab_size * 2  # wte + lm_head (no weight tying)
    param_count += config.n_layer * (
        4 * config.n_embd * config.n_embd +  # attention c_attn + c_proj
        2 * config.n_embd * 4 * config.n_embd +  # mlp c_fc + c_proj
        2 * config.n_embd  # layer norms
    )

    readme_content = f"""---
license: mit
tags:
- pytorch
- causal-lm
- gpt
- small-language-model
- decoder-only
language:
- en
pipeline_tag: text-generation
---

# Pico-GPT

A small GPT-style decoder-only language model (~{param_count/1e6:.1f}M parameters) trained from scratch on OpenWebText.

## Model Details

| Property | Value |
|----------|--------|
| **Architecture** | Decoder-only Transformer with Pre-LayerNorm |
| **Parameters** | ~{param_count:,} |
| **Layers** | {config.n_layer} |
| **Hidden Size** | {config.n_embd} |
| **FFN Hidden Size** | {config.ffn_dim} |
| **Attention Heads** | {config.n_head} |
| **Head Dimension** | {config.n_embd // config.n_head} |
| **Context Length** | {config.context_length} tokens |
| **Vocabulary** | {config.vocab_size} (GPT-2) |
| **Flash Attention** | {'✅ Enabled' if config.flash_attention else '❌ Disabled'} |
| **Dropout** | {config.dropout} |
| **Bias** | {'Disabled' if not config.bias else 'Enabled'} |

## Training Objective

The model was trained using **causal language modeling (next-token prediction)**. The loss function is cross-entropy over the vocabulary.

For a given sequence of tokens `x_1, x_2, ..., x_n`, the model is trained to predict `x_{{i+1}}` given `x_1, ..., x_i`.

## Dataset

### Source
- **Dataset:** OpenWebText
- **Hugging Face:** `Skylion007/openwebtext`
- **Mode:** Streaming preprocessing
- **License:** Same as OpenAI's GPT-2 dataset

### Preprocessing Pipeline
- **Tokenizer:** GPT-2 (tiktoken)
- **Tokenization:** Streaming, incremental
- **EOS Token:** Appended after each document
- **Text Cleaning:** Minimal (strip whitespace, skip empty strings)
- **Sharding:** Binary shards (uint16), 5M tokens per shard
- **Train/Val Split:** Deterministic split by token count
- **Memory Mapping:** Enabled for efficient loading

### Dataset Statistics
- **Total Tokens Collected:** 1B tokens
- **Training Tokens:** 950M tokens
- **Validation Tokens:** 50M tokens
- **Training Shards:** ~190 files (train_000.bin to train_189.bin)
- **Validation Shard:** val.bin
- **Data Type:** uint16 (supports memory mapping)

## Training Configuration

### Hyperparameters
| Parameter | Value |
|-----------|--------|
| **Optimizer** | AdamW |
| **Learning Rate** | {lr} |
| **Weight Decay** | {weight_decay} |
| **Betas** | (0.9, 0.95) |
| **Max Steps** | {max_steps} |
| **Batch Size** | 64 |
| **Context Window** | {config.context_length} |
| **Gradient Clipping** | 1.0 |
| **Checkpoint Interval** | {training_config.get('checkpoint_interval', 'N/A')} |
| **Log Interval** | {training_config.get('log_interval', 'N/A')} |

### Training Results
| Metric | Value |
|--------|--------|
| **Final Training Loss** | {final_loss} |
| **Training Time** | {training_time if training_time == 'N/A' else f'{training_time/60:.1f} minutes'} |
| **Hardware** | NVIDIA A100 (20GB) or equivalent |

## Model Files

| File | Description |
|-------|-------------|
| `model.safetensors` | Model weights in safetensors format (secure, fast loading) |
| `config.json` | Model architecture configuration |
| `training_config.json` | Training hyperparameters and results |
| `training_log.csv` | Training metrics over time (step, loss, elapsed_time) |
| `samples.txt` | Sample generations from the trained model |
| `tokenizer_config.json` | Tokenizer configuration |
| `special_tokens_map.json` | Special tokens mapping |

## Installation

This model requires the custom `pico_gpt` package to load and run.

```bash
git clone https://github.com/ChidambaraRaju/pico-gpt.git
cd pico-gpt
pip install -e .
```

Or install dependencies only:
```bash
pip install torch tiktoken safetensors
```

## Usage

### Loading with safetensors:

```python
import torch
from safetensors.torch import load_file
import json

# Load config
with open("config.json", "r") as f:
    config = json.load(f)

# Load weights
state_dict = load_file("model.safetensors")

# Create model (requires custom model class from pico_gpt/model.py)
from pico_gpt.model import GPT
from pico_gpt.config import ModelConfig

model = GPT(ModelConfig(**config))
model.load_state_dict(state_dict)
model.eval()
```

### Text Generation:

```python
import torch
import tiktoken

# Load tokenizer
enc = tiktoken.get_encoding("gpt2")

# Prepare prompt
prompt = "The future of artificial intelligence is"
tokens = enc.encode(prompt)
tokens = tokens[-context_length:]  # Truncate to context length if needed
idx = torch.tensor([tokens], dtype=torch.long)

# Generate
with torch.no_grad():
    generated = model.generate(
        idx,
        max_new_tokens=100,
        temperature=0.8,
        eos_token_id=enc.eot_token,
    )

# Decode result
generated_text = enc.decode(generated[0].tolist())
print(generated_text)
```

### Loading Checkpoint:

```python
import torch

# Load checkpoint
checkpoint = torch.load("checkpoint_step_<N>.pt", map_location="cpu")
model_state = checkpoint["model_state_dict"]
config = checkpoint["config"]

# Load training config if needed
training_config = checkpoint.get("training_config", {{}})

# Use with custom GPT class
from pico_gpt.model import GPT
from pico_gpt.config import ModelConfig

model = GPT(config)
model.load_state_dict(model_state)
```

## Limitations

- **Small Model Size:** ~{param_count/1e6:.1f}M parameters limits reasoning capability
- **Short Context:** {config.context_length} token context window limits long-range dependencies
- **Single Dataset:** Trained only on web text (OpenWebText subset)
- **No Instruction Tuning:** Not aligned for chat/instruction following
- **Potential Biases:** May contain biases present in the training data
- **No Weight Tying:** Embedding and output layers have separate parameters

## Future Work

- [ ] Convert to native Hugging Face GPT-2 architecture
- [ ] Increase model size and context length
- [ ] Add instruction tuning / alignment
- [ ] Evaluation on downstream benchmarks (perplexity, etc.)
- [ ] Fine-tune for specific tasks
- [ ] Implement more sampling strategies (top-k, top-p)
- [ ] Add support for streaming inference

## Citation

```bibtex
@misc{{pico-gpt,
  title={{Pico-GPT: A Small Language Model from Scratch}},
  author={{Your Name}},
  year={{2026}},
  howpublished={{\\url{{https://huggingface.co/YOUR_USERNAME/pico-gpt}}}},
}}
```

## Acknowledgments

- This project uses the **GPT-2 tokenizer** from OpenAI's `tiktoken` library
- Dataset: **OpenWebText** by Skylion007
- Architecture inspired by **GPT**, **GPT-2**, and **nanoGPT**

---

*For training details, see `training_config.json` and `training_log.csv`.*
*Model files use the safetensors format for safe and efficient loading.*
"""

    with open(output_path / "README.md", "w") as f:
        f.write(readme_content)

    print("Created model card: README.md")


def upload_to_hub(
    repo_id: str,
    model_dir: str,
    private: bool = False,
) -> None:
    """
    Upload model to Hugging Face Hub.

    Requires `huggingface-hub` package.

    Args:
        repo_id: Repository ID (e.g., "username/pico-gpt")
        model_dir: Directory containing model files
        private: Whether repository is private
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("Please install huggingface-hub: pip install huggingface_hub")
        return

    api = HfApi()

    # Create repository if it doesn't exist
    try:
        create_repo(repo_id=repo_id, private=private, exist_ok=True)
        print(f"Created repository: {repo_id}")
    except Exception as e:
        print(f"Repository may already exist or error occurred: {e}")

    # Upload all files in the directory
    print(f"Uploading files from {model_dir} to {repo_id}...")
    api.upload_folder(
        repo_id=repo_id,
        folder_path=model_dir,
        repo_type="model",
    )

    print(f"\nModel uploaded to: https://huggingface.co/{repo_id}")
    print("\nFiles uploaded:")
    for file in Path(model_dir).iterdir():
        if file.is_file():
            print(f"  - {file.name}")

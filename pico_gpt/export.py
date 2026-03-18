"""
Export utilities for Hugging Face Hub.

Reference: @instructions/07_huggingface_export.md
"""

import json
from pathlib import Path
from typing import Optional
import torch
from safetensors.torch import save_file

from pico_gpt.model import GPT
from pico_gpt.config import ModelConfig
from pico_gpt.tokenizer_utils import export_tokenizer_metadata


def export_to_huggingface(
    checkpoint_path: str,
    output_dir: str,
) -> None:
    """
    Export model to Hugging Face format.

    Args:
        checkpoint_path: Path to trained model checkpoint
        output_dir: Output directory for Hugging Face model
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Get config
    if "config" in checkpoint:
        config = checkpoint["config"]
    else:
        config = ModelConfig()

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

    # Create config.json
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

    # Export tokenizer metadata
    export_tokenizer_metadata(str(output_path), model_max_length=config.context_length)

    # Create model card
    create_model_card(output_path, config, checkpoint_path)

    print(f"\nExport complete! Model saved to {output_path}")
    print("\nTo upload to Hugging Face:")
    print(f"  huggingface-cli upload-model {output_path}")


def create_model_card(output_path: Path, config: ModelConfig, checkpoint_path: str) -> None:
    """Create model card README.md."""
    readme_content = f"""---
license: mit
tags:
- pytorch
- causal-lm
- gpt
- small-language-model
language:
- en
---

# Pico-GPT

A small GPT-style decoder-only language model (~35M parameters) trained from scratch on OpenWebText.

## Model Details

- **Architecture:** Decoder-only Transformer with Pre-LayerNorm
- **Parameters:** ~35M
- **Layers:** {config.n_layer}
- **Hidden Size:** {config.n_embd}
- **Attention Heads:** {config.n_head}
- **Context Length:** {config.context_length} tokens
- **Vocabulary:** {config.vocab_size:,} (GPT-2)

## Training

- **Dataset:** OpenWebText subset (~100M tokens)
- **Training Split:** 95M tokens
- **Validation Split:** 5M tokens
- **Hardware:** NVIDIA A100 (20GB)
- **Mixed Precision:** BF16
- **Training Time:** ~22 hours
- **Optimizer:** AdamW (lr=3e-4, weight_decay=0.1)
- **Learning Rate:** Cosine decay with 2K step warmup

## Usage

### Loading with safetensors:

```python
import torch
from safetensors.torch import load_file
import json

# Load config
with open("config.json", "r") as f:
    config = json.load(f)

# Load weights (lm_head.weight is tied to wte.weight, not saved separately)
state_dict = load_file("model.safetensors")

# Create model and load state_dict
# (requires custom model class from pico_gpt/model.py)
```

### Generation:

```python
import tiktoken

# Load tokenizer (GPT-2)
enc = tiktoken.get_encoding("gpt2")

# Encode prompt
prompt = "The future of AI is"
tokens = enc.encode(prompt)

# Your model forward pass would go here
```

## Limitations

- Small model size (35M) limits reasoning capability
- Short context window (128 tokens)
- Trained only on web text (OpenWebText)
- No instruction tuning or alignment

## Future Work

- Convert to native Hugging Face GPT-2 architecture
- Increase model size and context length
- Add instruction tuning
- Evaluation on downstream tasks

## Citation

```bibtex
@misc{{pico-gpt,
  title={{Pico-GPT: A Small Language Model from Scratch}},
  author={{Your Name}},
  year={{2026}}
}}
```

---

This model uses the GPT-2 tokenizer from OpenAI's `tiktoken` library.
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
        print(f"Repository may already exist: {e}")

    # Upload files
    api.upload_folder(
        repo_id=repo_id,
        folder_path=model_dir,
        repo_type="model",
    )

    print(f"\nModel uploaded to: https://huggingface.co/{repo_id}")

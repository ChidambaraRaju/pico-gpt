# Pico-GPT API Reference

## Core Modules

### `pico_gpt.config`

Configuration classes for model and training.

#### `ModelConfig`

```python
@dataclass
class ModelConfig:
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    vocab_size: int = 50257
    context_length: int = 128
    ffn_dim: int = 1536
    dropout: float = 0.1
    bias: bool = False
    flash_attention: bool = True
```

#### `TrainingConfig`

```python
@dataclass
class TrainingConfig:
    data_dir: str = "data"
    shard_size: int = 5_000_000
    total_tokens: int = 100_000_000
    val_tokens: int = 5_000_000
    batch_size: int = 64
    micro_batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    warmup_steps: int = 2000
    min_lr: float = 3e-5
    max_steps: int = 200_000
    eval_interval: int = 200
    checkpoint_interval: int = 1000
    grad_clip: float = 1.0
    checkpoint_dir: str = "checkpoints"
    resume_from: Optional[str] = None
    log_interval: int = 1
    output_dir: str = "outputs"
    use_bf16: bool = True
```

### `pico_gpt.tokenizer`

Tokenizer wrapper for OpenAI tiktoken.

#### `GPT2Tokenizer`

```python
class GPT2Tokenizer:
    """Wrapper for tiktoken GPT-2 tokenizer."""

    def __init__(self) -> None:
        """Initialize tokenizer."""

    def encode(self, text: str) -> List[int]:
        """Encode text to token ids."""

    def decode(self, tokens: List[int]) -> str:
        """Decode token ids to text."""

    def truncate(self, tokens: List[int], max_length: int) -> List[int]:
        """Truncate tokens from left to max_length."""
```

### `pico_gpt.model`

Model architecture components.

#### `GPT`

```python
class GPT(nn.Module):
    """GPT-style decoder-only transformer."""

    def __init__(self, config: ModelConfig) -> None:
        """Initialize model."""

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            idx: Input token ids (B, T)
            targets: Target token ids (B, T), optional

        Returns:
            logits (B, T, vocab_size), loss or None
        """

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            idx: Input token ids (B, T)
            max_new_tokens: Maximum new tokens
            temperature: Sampling temperature

        Returns:
            Generated tokens (B, T + max_new_tokens)
        """
```

### `pico_gpt.dataloader`

Memory-mapped dataset loader.

#### `MemoryMappedDataset`

```python
class MemoryMappedDataset:
    """Memory-mapped dataset for efficient training."""

    def __init__(
        self,
        data_dir: str | Path,
        context_length: int,
        batch_size: int,
        split: str = "train"
    ) -> None:
        """Initialize dataset loader."""

    def get_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a random batch of (x, y) pairs."""

    def __len__(self) -> int:
        """Number of valid start positions."""
```

### `pico_gpt.trainer`

Training loop utilities.

#### `Trainer`

```python
class Trainer:
    """Training loop for Pico-GPT."""

    def __init__(
        self,
        model: GPT,
        train_loader: MemoryMappedDataset,
        val_loader: Optional[MemoryMappedDataset],
        output_dir: str,
        max_steps: int,
        **kwargs
    ) -> None:
        """Initialize trainer."""

    def train_step(self) -> Dict:
        """Perform one training step."""

    @torch.no_grad()
    def validate(self) -> Optional[float]:
        """Run validation loop."""

    def train(self) -> None:
        """Main training loop."""

    def save_checkpoint(self, filename: str) -> None:
        """Save training checkpoint."""

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
```

## Scripts

### `scripts/prepare_data.py`

Preprocess OpenWebText dataset.

```bash
python scripts/prepare_data.py \
    --output-dir data \
    --shard-size 5000000 \
    --total-tokens 100000000 \
    --val-tokens 5000000
```

### `scripts/train.py`

Train model.

```bash
python scripts/train.py \
    --data-dir data \
    --output-dir checkpoints \
    --max-steps 200000
```

### `scripts/generate.py`

Generate text from trained model.

```bash
python scripts/generate.py \
    --model checkpoints/best_model.pt \
    --prompt "The future of AI is" \
    --max-tokens 100 \
    --temperature 0.8
```

### `scripts/export_hf.py`

Export to Hugging Face format.

```bash
python scripts/export_hf.py \
    --checkpoint checkpoints/best_model.pt \
    --output hf_model \
    --upload username/pico-gpt
```

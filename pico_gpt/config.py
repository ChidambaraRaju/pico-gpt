"""Configuration module for Pico-GPT model and training."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    # Architecture
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    vocab_size: int = 50257
    context_length: int = 128

    # Feedforward
    ffn_dim: int = 1536  # 4 * n_embd

    # Regularization
    dropout: float = 0.1

    # Attention
    bias: bool = False
    flash_attention: bool = True

    def __post_init__(self):
        """Validate configuration."""
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.ffn_dim == 4 * self.n_embd, "ffn_dim should be 4 * n_embd"


@dataclass
class TrainingConfig:
    """Training loop configuration."""

    # Dataset
    data_dir: str = "data"
    shard_size: int = 5_000_000  # tokens per shard
    total_tokens: int = 100_000_000  # total training tokens
    val_tokens: int = 5_000_000  # validation tokens

    # Training hyperparameters
    batch_size: int = 64
    micro_batch_size: int = 8
    gradient_accumulation_steps: int = field(init=False)

    # Optimizer
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)

    # Learning rate schedule
    warmup_steps: int = 2000
    min_lr: float = 3e-5

    # Training
    max_steps: int = 200_000
    eval_interval: int = 200
    checkpoint_interval: int = 1000
    grad_clip: float = 1.0

    # Checkpoints
    checkpoint_dir: str = "checkpoints"
    resume_from: Optional[str] = None

    # Logging
    log_interval: int = 1
    output_dir: str = "outputs"

    # Mixed precision
    use_bf16: bool = True

    def __post_init__(self):
        """Validate and compute derived values."""
        self.gradient_accumulation_steps = self.batch_size // self.micro_batch_size
        assert self.batch_size % self.micro_batch_size == 0, "batch_size must be divisible by micro_batch_size"


@dataclass
class GenerationConfig:
    """Text generation configuration."""

    model_path: str = "checkpoints/best_model.pt"
    max_new_tokens: int = 100
    temperature: float = 0.8
    prompt: str = "The future of artificial intelligence is"

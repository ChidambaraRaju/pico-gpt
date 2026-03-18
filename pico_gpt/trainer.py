"""Minimal training utilities for Pico-GPT."""
import torch
import time
from pathlib import Path
from safetensors.torch import save_file

from pico_gpt.model import GPT
from pico_gpt.dataloader import MemoryMappedDataset
from pico_gpt.config import ModelConfig


class Trainer:
    """
    Minimal training loop for Pico-GPT.

    Features:
    - Basic forward/backward pass
    - Gradient clipping
    - Simple checkpointing (PyTorch format)
    - Safetensors export for Hugging Face
    - Minimal logging
    """

    def __init__(
        self,
        model: GPT,
        train_loader: MemoryMappedDataset,
        output_dir: str,
        config: ModelConfig,
        max_steps: int,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        checkpoint_interval: int = 1000,
        log_interval: int = 100,
    ):
        """
        Initialize trainer.

        Args:
            model: GPT model to train
            train_loader: Training dataset loader
            output_dir: Directory for outputs and checkpoints
            config: Model configuration
            max_steps: Maximum training steps
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            checkpoint_interval: Save checkpoint every N steps
            log_interval: Print logs every N steps
        """
        self.model = model
        self.train_loader = train_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config

        self.max_steps = max_steps
        self.checkpoint_interval = checkpoint_interval
        self.log_interval = log_interval

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Simple AdamW optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )

    def train(self) -> None:
        """Main training loop."""
        self.model.train()
        print(f"Training on {self.device}")
        print(f"Max steps: {self.max_steps}")

        start_time = time.time()

        for step in range(self.max_steps):
            # Get batch
            x, y = self.train_loader.get_batch()
            x = torch.from_numpy(x).to(self.device)
            y = torch.from_numpy(y).to(self.device)

            # Forward pass
            logits, loss = self.model(x, targets=y)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Logging
            if (step + 1) % self.log_interval == 0:
                elapsed = time.time() - start_time
                print(f"Step {step+1:6d} | Loss: {loss.item():.4f} | Time: {elapsed:.1f}s")

            # Checkpointing (PyTorch format - fast)
            if (step + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint(step + 1)

        # Final save
        self.save_checkpoint(self.max_steps)
        # Also save final model as safetensors for HF export
        self.save_safetensors(self.max_steps)
        print(f"\nTraining complete!")

    def save_checkpoint(self, step: int) -> None:
        """Save model checkpoint in PyTorch format."""
        checkpoint = {
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
        }
        path = self.output_dir / f"checkpoint_step_{step}.pt"
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

    def save_safetensors(self, step: int) -> None:
        """
        Save model in safetensors format for Hugging Face export.
        """
        state_dict = self.model.state_dict()
        path = self.output_dir / f"model_step_{step}.safetensors"
        save_file(state_dict, path)
        print(f"Saved safetensors: {path}")

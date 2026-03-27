"""Minimal training utilities for Pico-GPT."""
import torch
import time
import csv
from pathlib import Path
from safetensors.torch import save_file
from datetime import datetime
from tqdm import tqdm

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
    - Training log CSV export
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
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

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

        # Training log for CSV export
        self.training_log = []

        # Setup CSV log file
        self.log_file = self.output_dir / "training_log.csv"
        self._init_log_file()

    def _init_log_file(self) -> None:
        """Initialize the training log CSV file."""
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "loss", "elapsed_time"])
        print(f"Training log will be saved to: {self.log_file}")

    def _log_step(self, step: int, loss: float, elapsed: float) -> None:
        """Log a training step to CSV."""
        self.training_log.append({
            "step": step,
            "loss": loss,
            "elapsed_time": elapsed,
        })
        # Append to CSV
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([step, f"{loss:.4f}", f"{elapsed:.1f}"])

    def train(self) -> None:
        """Main training loop."""
        self.model.train()
        print(f"Training on {self.device}")
        print(f"Max steps: {self.max_steps}")

        start_time = time.time()

        pbar = tqdm(total=self.max_steps, desc="Training", unit="step")
        for step in pbar:
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
                pbar.set_postfix(loss=f"{loss.item():.4f}", time=f"{elapsed:.1f}s")
                # Log to CSV
                self._log_step(step + 1, loss.item(), elapsed)

            # Checkpointing (PyTorch format - fast)
            if (step + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint(step + 1)
        pbar.close()

        # Final save
        final_loss = loss.item()
        final_time = time.time() - start_time
        self._log_step(self.max_steps, final_loss, final_time)

        self.save_checkpoint(self.max_steps, training_config={
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_steps": self.max_steps,
            "checkpoint_interval": self.checkpoint_interval,
            "log_interval": self.log_interval,
            "final_loss": final_loss,
            "training_time_seconds": final_time,
        })
        # Also save final model as safetensors for HF export
        self.save_safetensors(self.max_steps)
        print(f"\nTraining complete!")
        print(f"Final loss: {final_loss:.4f}")
        print(f"Total time: {final_time:.1f}s")

    def save_checkpoint(self, step: int, training_config: dict = None) -> None:
        """Save model checkpoint in PyTorch format."""
        checkpoint = {
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
        }
        if training_config:
            checkpoint["training_config"] = training_config
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

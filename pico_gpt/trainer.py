"""Training utilities for Pico-GPT.
Reference: @instructions/05_training_pipeline.md
"""
import math
from typing import Optional, List, Dict
import torch
import torch.nn as nn
import csv
import time
from pathlib import Path

from pico_gpt.model import GPT
from pico_gpt.dataloader import MemoryMappedDataset


class CosineAnnealingWarmupScheduler:
    """
    Cosine learning rate schedule with linear warmup.

    LR increases linearly during warmup, then decays following
    a cosine curve to minimum learning rate.

    Reference: @instructions/05_training_pipeline.md
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        max_lr: float,
        min_lr: float,
    ):
        """
        Initialize scheduler.

        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of warmup steps
            max_steps: Total training steps
            max_lr: Peak learning rate after warmup
            min_lr: Minimum learning rate after decay
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = 0

    def get_lr_for_step(self, step: int) -> float:
        """
        Get learning rate for a given step.

        Args:
            step: Current training step

        Returns:
            Learning rate for this step
        """
        if step < self.warmup_steps:
            # Linear warmup
            return self.max_lr * (step + 1) / self.warmup_steps
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return self.min_lr + (self.max_lr - self.min_lr) * cosine_decay

    def step(self) -> None:
        """Update optimizer learning rate."""
        lr = self.get_lr_for_step(self.current_step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.current_step += 1

    def state_dict(self) -> dict:
        """Return scheduler state."""
        return {
            "current_step": self.current_step,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load scheduler state."""
        self.current_step = state_dict["current_step"]


class Trainer:
    """
    Training loop for Pico-GPT.

    Features:
    - BF16 mixed precision training
    - Gradient accumulation
    - Gradient clipping
    - Periodic validation
    - Checkpointing
    - Logging

    Reference: @instructions/05_training_pipeline.md
    """

    def __init__(
        self,
        model: GPT,
        train_loader: MemoryMappedDataset,
        val_loader: Optional[MemoryMappedDataset],
        output_dir: str,
        max_steps: int,
        batch_size: int = 64,
        micro_batch_size: int = 8,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        warmup_steps: int = 2000,
        min_lr: float = 3e-5,
        eval_interval: int = 200,
        checkpoint_interval: int = 1000,
        grad_clip: float = 1.0,
        use_bf16: bool = True,
        resume_from: Optional[str] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: GPT model to train
            train_loader: Training dataset loader
            val_loader: Validation dataset loader (optional)
            output_dir: Directory for outputs and checkpoints
            max_steps: Maximum training steps
            batch_size: Total batch size
            micro_batch_size: Batch size per GPU step
            learning_rate: Peak learning rate
            weight_decay: Weight decay for optimizer
            warmup_steps: Number of warmup steps
            min_lr: Minimum learning rate
            eval_interval: Validation interval in steps
            checkpoint_interval: Checkpoint interval in steps
            grad_clip: Gradient clipping threshold
            use_bf16: Whether to use BF16 mixed precision
            resume_from: Path to checkpoint to resume from
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.max_steps = max_steps
        self.eval_interval = eval_interval
        self.checkpoint_interval = checkpoint_interval
        self.grad_clip = grad_clip
        self.use_bf16 = use_bf16 and torch.cuda.is_bf16_supported()

        # Gradient accumulation
        self.batch_size = batch_size
        self.micro_batch_size = micro_batch_size
        self.grad_accum_steps = batch_size // micro_batch_size

        # Optimizer setup
        # Exclude LayerNorm and embedding weights from weight decay
        decay_params = []
        no_decay_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "ln" in name or "wte" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            betas=(0.9, 0.95),
        )

        # Learning rate scheduler
        self.scheduler = CosineAnnealingWarmupScheduler(
            optimizer=self.optimizer,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            max_lr=learning_rate,
            min_lr=min_lr,
        )

        # Training state
        self.step = 0
        self.best_val_loss = float("inf")
        self.logs: List[Dict] = []

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Resume if checkpoint provided
        if resume_from:
            self.load_checkpoint(resume_from)

    def train_step(self) -> Dict:
        """
        Perform one training step with gradient accumulation.

        Returns:
            Dictionary of metrics
        """
        self.model.train()
        total_loss = 0.0

        for _ in range(self.grad_accum_steps):
            # Get batch
            x, y = self.train_loader.get_batch()
            x = torch.from_numpy(x).to(self.device)
            y = torch.from_numpy(y).to(self.device)

            # Forward pass with autocast
            if self.use_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, loss = self.model(x, targets=y)
                    loss = loss / self.grad_accum_steps
            else:
                logits, loss = self.model(x, targets=y)
                loss = loss / self.grad_accum_steps

            total_loss += loss.item()

            # Backward pass
            loss.backward()

        # Gradient clipping
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()

        self.step += 1

        return {
            "loss": total_loss,
            "lr": self.scheduler.get_lr_for_step(self.step),
            "step": self.step,
        }

    @torch.no_grad()
    def validate(self) -> Optional[float]:
        """
        Run validation loop.

        Returns:
            Average validation loss, or None if no val_loader
        """
        if self.val_loader is None:
            return None

        self.model.eval()
        total_loss = 0.0
        num_batches = 100  # Limit validation batches

        for _ in range(num_batches):
            x, y = self.val_loader.get_batch()
            x = torch.from_numpy(x).to(self.device)
            y = torch.from_numpy(y).to(self.device)

            if self.use_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, loss = self.model(x, targets=y)
            else:
                _, loss = self.model(x, targets=y)

            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        return avg_loss

    def train(self) -> None:
        """
        Main training loop.

        Reference: @instructions/05_training_pipeline.md
        """
        print(f"Starting training for {self.max_steps} steps...")
        print(f"Device: {self.device}")
        print(f"BF16: {self.use_bf16}")
        print(f"Batch size: {self.batch_size} (micro: {self.micro_batch_size}, grad_acc: {self.grad_accum_steps})")

        start_time = time.time()

        while self.step < self.max_steps:
            # Training step
            metrics = self.train_step()

            # Log
            self.logs.append({
                "step": metrics["step"],
                "train_loss": metrics["loss"],
                "lr": metrics["lr"],
            })

            if metrics["step"] % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Step {metrics['step']:6d} | Loss: {metrics['loss']:.4f} | LR: {metrics['lr']:.2e} | Time: {elapsed:.1f}s")

            # Validation
            if self.val_loader and metrics["step"] % self.eval_interval == 0:
                val_loss = self.validate()
                print(f"  -> Val Loss: {val_loss:.4f}")

                self.logs[-1]["val_loss"] = val_loss

                # Update best loss
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best_model.pt")

            # Checkpoint
            if metrics["step"] % self.checkpoint_interval == 0:
                self.save_checkpoint(f"checkpoint_{metrics['step']}.pt")

            # Save logs
            if metrics["step"] % 100 == 0:
                self.save_logs()

        # Final save
        self.save_checkpoint("final_model.pt")
        self.save_logs()

        print(f"\nTraining complete! Best val loss: {self.best_val_loss:.4f}")

    def save_checkpoint(self, filename: str) -> None:
        """Save training checkpoint."""
        checkpoint = {
            "step": self.step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
        }
        torch.save(checkpoint, self.output_dir / filename)
        print(f"Saved checkpoint: {filename}")

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.step = checkpoint["step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        print(f"Loaded checkpoint from {path}, resuming at step {self.step}")

    def save_logs(self) -> None:
        """Save training logs to CSV."""
        log_path = self.output_dir / "training_log.csv"
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "train_loss", "val_loss", "lr"])
            writer.writeheader()
            writer.writerows(self.logs)

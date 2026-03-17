"""
Training script for Pico-GPT.
Reference: @instructions/05_training_pipeline.md
"""
import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))
from pico_gpt.config import ModelConfig, TrainingConfig
from pico_gpt.model import GPT
from pico_gpt.dataloader import MemoryMappedDataset
from pico_gpt.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train Pico-GPT")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max steps")
    args = parser.parse_args()

    # Load or create config
    if args.config:
        # Load from file (simplified)
        model_config = ModelConfig()
        training_config = TrainingConfig()
    else:
        model_config = ModelConfig()
        training_config = TrainingConfig()

    # Override max steps if specified
    if args.max_steps:
        training_config.max_steps = args.max_steps

    print(f"Model config: {model_config}")
    print(f"Training config: {training_config}")

    # Create model
    model = GPT(model_config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Create data loaders
    train_loader = MemoryMappedDataset(
        data_dir=args.data_dir,
        context_length=model_config.context_length,
        batch_size=training_config.batch_size,
        split="train",
    )

    val_loader = MemoryMappedDataset(
        data_dir=args.data_dir,
        context_length=model_config.context_length,
        batch_size=training_config.batch_size,
        split="val",
    )

    print(f"Training tokens: {train_loader.n_tokens:,}")
    print(f"Validation tokens: {val_loader.n_tokens:,}")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=args.output_dir,
        max_steps=training_config.max_steps,
        batch_size=training_config.batch_size,
        micro_batch_size=training_config.micro_batch_size,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        warmup_steps=training_config.warmup_steps,
        min_lr=training_config.min_lr,
        eval_interval=training_config.eval_interval,
        checkpoint_interval=training_config.checkpoint_interval,
        grad_clip=training_config.grad_clip,
        use_bf16=training_config.use_bf16,
        resume_from=args.resume,
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    main()

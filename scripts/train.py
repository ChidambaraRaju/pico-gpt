"""
Training script for Pico-GPT.
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
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max steps")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--checkpoint-interval", type=int, default=None, help="Override checkpoint interval")
    args = parser.parse_args()

    # Load config
    model_config = ModelConfig()
    training_config = TrainingConfig()

    # Override if specified
    if args.max_steps:
        training_config.max_steps = args.max_steps

    print(f"Model config: {model_config}")
    print(f"Training config: max_steps={training_config.max_steps}, lr={training_config.learning_rate}")

    # Create model
    model = GPT(model_config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Create data loader
    train_loader = MemoryMappedDataset(
        data_dir=args.data_dir,
        context_length=model_config.context_length,
        batch_size=training_config.batch_size,
        split="train",
    )

    print(f"Training tokens: {train_loader.n_tokens:,}")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        output_dir=args.output_dir,
        config=model_config,
        max_steps=training_config.max_steps,
        learning_rate=args.lr or training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        checkpoint_interval=args.checkpoint_interval or training_config.checkpoint_interval,
        log_interval=100,
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    main()

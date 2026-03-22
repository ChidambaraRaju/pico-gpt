"""
Dataset preprocessing script for Pico-GPT.

Streams OpenWebText dataset, tokenizes using tiktoken,
and saves binary shards for training.

Reference: @instructions/02_dataset_preparation.md
"""

import sys
from pathlib import Path
import argparse

from datasets import load_dataset
import tiktoken

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from pico_gpt.data import PreprocessingState, TokenBuffer


def clean_text(text: str) -> str:
    """
    Apply minimal text cleaning.

    Rules:
    - Strip leading/trailing whitespace
    - Skip empty strings
    - Normalize multiple spaces to single space

    Reference: @instructions/02_dataset_preparation.md
    """
    text = text.strip()
    if not text:
        return ""

    # Normalize multiple spaces to single space
    while "  " in text:
        text = text.replace("  ", " ")

    return text


def prepare_dataset(
    output_dir: str = "data",
    shard_size: int = 5_000_000,
    total_tokens: int = 1_000_000_000,
    val_tokens: int = 50_000_000,
    resume: bool = True,
) -> None:
    """
    Prepare dataset by streaming and tokenizing OpenWebText.

    Args:
        output_dir: Directory to save binary shards
        shard_size: Number of tokens per shard
        total_tokens: Total tokens to process
        val_tokens: Number of tokens for validation split
        resume: Whether to resume from existing state
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    state_path = output_path / "preprocessing_state.json"

    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    eos_token = enc.eot_token

    # Load or initialize state
    if resume and state_path.exists():
        state = PreprocessingState.load(state_path)
        print(f"Resuming from step: {state.total_tokens:,} tokens processed")
    else:
        state = PreprocessingState(shard_index=0, tokens_written=0, total_tokens=0)

    # Initialize token buffer
    buffer = TokenBuffer(
        output_dir=output_path,
        shard_size=shard_size,
        total_tokens=total_tokens,
        val_tokens=val_tokens,
    )

    # Skip processed shards if resuming
    if resume and state.shard_index > 0:
        buffer.shard_index = state.shard_index

    # Load dataset in streaming mode
    print("Loading OpenWebText dataset (streaming)...")
    dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

    # Skip already processed rows (approximate)
    if resume and state.total_tokens > 0:
        # Estimate tokens per row to skip
        avg_tokens_per_row = 300  # Rough estimate
        rows_to_skip = state.total_tokens // avg_tokens_per_row
        print(f"Skipping approximately {rows_to_skip:,} rows...")
        dataset = dataset.skip(rows_to_skip)

    print(f"Processing up to {total_tokens:,} tokens...")
    print(f"Validation split: {val_tokens:,} tokens")

    # Process documents
    for idx, example in enumerate(dataset):
        if state.total_processed >= total_tokens:
            break

        # Clean and tokenize
        text = clean_text(example["text"])
        if not text:
            continue

        tokens = enc.encode_ordinary(text)
        tokens.append(eos_token)  # Append EOS after each document

        # Add tokens to buffer
        can_continue = buffer.add_tokens(tokens)
        state.total_processed += len(tokens)

        # Update state periodically
        if idx % 10000 == 0:
            state.tokens_written = len(buffer.buffer)
            state.total_tokens = state.total_processed
            state.save(state_path)

        if not can_continue:
            break

    # Finalize remaining tokens
    buffer.finalize()

    # Save final state
    state.shard_index = buffer.shard_index
    state.tokens_written = 0
    state.total_tokens = buffer.total_processed
    state.save(state_path)

    print(f"\nDataset preparation complete!")
    print(f"Total tokens processed: {buffer.total_processed:,}")
    print(f"Train tokens: {min(buffer.total_processed, total_tokens - val_tokens):,}")
    print(f"Val tokens: {min(val_tokens, buffer.total_processed):,}")


def main():
    parser = argparse.ArgumentParser(description="Prepare OpenWebText dataset for Pico-GPT training")
    parser.add_argument("--output-dir", default="data", help="Output directory for binary shards")
    parser.add_argument("--shard-size", type=int, default=5_000_000, help="Tokens per shard")
    parser.add_argument("--total-tokens", type=int, default=1_000_000_000, help="Total tokens to process")
    parser.add_argument("--val-tokens", type=int, default=50_000_000, help="Validation tokens")
    parser.add_argument("--no-resume", action="store_true", help="Start from scratch")
    args = parser.parse_args()

    prepare_dataset(
        output_dir=args.output_dir,
        shard_size=args.shard_size,
        total_tokens=args.total_tokens,
        val_tokens=args.val_tokens,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()

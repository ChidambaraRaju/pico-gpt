"""Dataset preprocessing utilities.

Reference: @instructions/02_dataset_preparation.md
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List
import numpy as np


@dataclass
class PreprocessingState:
    """State for dataset preprocessing resume capability."""

    shard_index: int  # Current shard being filled
    tokens_written: int  # Tokens in current shard
    total_tokens: int  # Total tokens processed across all shards
    total_processed: int = 0  # Cumulative tokens processed (for resume tracking)

    def save(self, path: Path) -> None:
        """Save state to JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "PreprocessingState":
        """Load state from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def is_shard_complete(self, shard_size: int) -> bool:
        """Check if current shard is complete."""
        return self.tokens_written >= shard_size

    def tokens_until_shard_complete(self, shard_size: int) -> int:
        """Get remaining tokens needed to complete current shard."""
        remaining = shard_size - self.tokens_written
        return max(0, remaining)


class TokenBuffer:
    """
    Buffer for accumulating tokens during streaming preprocessing.

    Handles:
    - Accumulating tokens from documents
    - Writing complete shards to disk
    - Managing train/validation split

    Reference: @instructions/02_dataset_preparation.md
    """

    def __init__(
        self,
        output_dir: Path,
        shard_size: int = 5_000_000,
        total_tokens: int = 100_000_000,
        val_tokens: int = 5_000_000,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.shard_size = shard_size
        self.total_tokens = total_tokens
        self.val_tokens = val_tokens
        self.train_tokens = total_tokens - val_tokens

        self.buffer: List[int] = []
        self.total_processed = 0
        self.shard_index = 0
        self.tokens_written_to_disk = 0  # Track cumulative tokens written

    def add_tokens(self, tokens: List[int]) -> bool:
        """
        Add tokens to buffer and write shards as needed.

        Returns:
            True if more tokens can be added, False if target reached
        """
        if self.total_processed >= self.total_tokens:
            return False

        remaining_needed = self.total_tokens - self.total_processed
        tokens_to_add = tokens[:remaining_needed]
        self.buffer.extend(tokens_to_add)
        self.total_processed += len(tokens_to_add)

        # Write shards as they fill up
        while len(self.buffer) >= self.shard_size and self.total_processed < self.total_tokens:
            shard_tokens = self.buffer[:self.shard_size]
            self.buffer = self.buffer[self.shard_size:]
            train_tokens_written = self._write_shard(shard_tokens, self.tokens_written_to_disk)
            self.tokens_written_to_disk += train_tokens_written

        return True

    def _write_shard(self, tokens: List[int], position: int) -> int:
        """
        Write a shard to disk (either train or validation).

        Args:
            tokens: The token list to write
            position: Position of first token in the overall stream

        Returns:
            Number of training tokens written (for position tracking)
        """
        # Determine split based on position in the stream
        # Train: [0, total_tokens - val_tokens)
        # Val: [total_tokens - val_tokens, total_tokens)
        train_end = self.total_tokens - self.val_tokens

        if position + len(tokens) <= train_end:
            # Pure training shard
            self._write_train_shard(tokens, self.shard_index)
            self.shard_index += 1
            return len(tokens)
        elif position >= train_end:
            # Pure validation shard
            self._write_val_shard(tokens)
            return 0
        else:
            # Split shard - part train, part validation
            remaining_train = train_end - position
            train_tokens = tokens[:remaining_train]
            val_tokens = tokens[remaining_train:]
            self._write_train_shard(train_tokens, self.shard_index)
            self.shard_index += 1
            # Validation appends to val.bin
            self._write_val_shard(val_tokens)
            return len(train_tokens)

    def _write_train_shard(self, tokens: List[int], index: int) -> None:
        """Write training shard to disk."""
        path = self.output_dir / f"train_{index:03d}.bin"
        np.array(tokens, dtype=np.uint16).tofile(path)

    def _write_val_shard(self, tokens: List[int]) -> None:
        """Write validation shard to disk (append mode)."""
        path = self.output_dir / "val.bin"
        # Append if file exists, otherwise create new
        if path.exists():
            with open(path, 'ab') as f:
                np.array(tokens, dtype=np.uint16).tofile(f)
        else:
            np.array(tokens, dtype=np.uint16).tofile(path)

    def finalize(self) -> None:
        """Write any remaining tokens in buffer."""
        if self.buffer and self.total_processed <= self.total_tokens:
            train_end = self.total_tokens - self.val_tokens
            if self.tokens_written_to_disk < train_end:
                train_tokens_written = self._write_shard(self.buffer, self.tokens_written_to_disk)
                self.tokens_written_to_disk += train_tokens_written
            else:
                self._write_val_shard(self.buffer)

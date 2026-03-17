"""Tokenizer wrapper using OpenAI tiktoken GPT-2 encoding."""

from typing import List
import tiktoken


class GPT2Tokenizer:
    """
    Lightweight wrapper for OpenAI tiktoken GPT-2 tokenizer.

    This wrapper provides a consistent interface for:
    - Encoding text to token ids
    - Decoding token ids to text
    - Truncating prompts to context length

    Reference: @instructions/03_tokenizer_usage.md
    """

    def __init__(self):
        """Initialize GPT-2 tokenizer."""
        self.enc = tiktoken.get_encoding("gpt2")
        self.eos_token_id = self.enc.eot_token
        self.vocab_size = self.enc.n_vocab

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token ids using ordinary encoding.

        Args:
            text: Input text string

        Returns:
            List of token ids
        """
        return self.enc.encode_ordinary(text)

    def decode(self, tokens: List[int]) -> str:
        """
        Decode token ids to text.

        Args:
            tokens: List of token ids

        Returns:
            Decoded text string
        """
        return self.enc.decode(tokens)

    def truncate(self, tokens: List[int], max_length: int) -> List[int]:
        """
        Truncate tokens from left to max_length.

        Used for prompt truncation during generation.

        Args:
            tokens: List of token ids
            max_length: Maximum allowed length

        Returns:
            Truncated list of token ids (last max_length tokens)
        """
        if len(tokens) <= max_length:
            return tokens
        return tokens[-max_length:]

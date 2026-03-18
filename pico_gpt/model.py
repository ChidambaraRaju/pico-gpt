"""GPT-style Transformer architecture.

Reference: @instructions/04_model_architecture.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from pico_gpt.config import ModelConfig


class CausalSelfAttention(nn.Module):
    """
    Multi-head self-attention with fused QKV projection.

    Uses PyTorch's scaled_dot_product_attention for Flash Attention support.

    Reference: @instructions/04_model_architecture.md
    """

    def __init__(self, n_embd: int, n_head: int, dropout: float, bias: bool, context_length: int):
        """
        Initialize attention module.

        Args:
            n_embd: Embedding dimension
            n_head: Number of attention heads
            dropout: Dropout rate
            bias: Whether to use bias in linear layers
            context_length: Maximum sequence length
        """
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"

        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.dropout = dropout

        # Fused QKV projection: (n_embd -> 3 * n_embd)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)

        # Output projection: (n_embd -> n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)

        # Residual connection dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with fused QKV and Flash Attention.

        Args:
            x: Input tensor of shape (B, T, C)

        Returns:
            Output tensor of shape (B, T, C)
        """
        B, T, C = x.size()

        # Fused QKV projection: (B, T, 3*C)
        qkv = self.c_attn(x)

        # Split into Q, K, V: each (B, T, C)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head: (B, T, n_head, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)

        # Transpose to (B, n_head, T, head_dim) for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention with causal mask
        # Enables Flash Attention on supported hardware
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        # Transpose back to (B, T, n_head, head_dim)
        y = y.transpose(1, 2)

        # Merge heads: (B, T, C)
        y = y.contiguous().view(B, T, C)

        # Output projection
        y = self.c_proj(y)

        # Apply residual dropout
        y = self.resid_dropout(y)

        return y


class MLP(nn.Module):
    """
    Feedforward network for transformer blocks.

    Structure: Linear -> GELU -> Linear -> Dropout

    Reference: @instructions/04_model_architecture.md
    """

    def __init__(self, n_embd: int, ffn_dim: int, dropout: float, bias: bool):
        """
        Initialize MLP.

        Args:
            n_embd: Input/output dimension
            ffn_dim: Hidden dimension (typically 4 * n_embd)
            dropout: Dropout rate
            bias: Whether to use bias in linear layers
        """
        super().__init__()

        self.c_fc = nn.Linear(n_embd, ffn_dim, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(ffn_dim, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, T, C)

        Returns:
            Output tensor of shape (B, T, C)
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm transformer block.

    Structure:
        x = x + Attention(LN(x))
        x = x + MLP(LN(x))

    Reference: @instructions/04_model_architecture.md
    """

    def __init__(self, n_embd: int, n_head: int, ffn_dim: int, dropout: float, bias: bool, context_length: int):
        """
        Initialize transformer block.

        Args:
            n_embd: Embedding dimension
            n_head: Number of attention heads
            ffn_dim: Feedforward hidden dimension
            dropout: Dropout rate
            bias: Whether to use bias
            context_length: Maximum sequence length
        """
        super().__init__()

        # Pre-LayerNorm for attention
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, bias, context_length)

        # Pre-LayerNorm for MLP
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, ffn_dim, dropout, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, T, C)

        Returns:
            Output tensor of shape (B, T, C)
        """
        # Attention with residual connection
        x = x + self.attn(self.ln_1(x))

        # MLP with residual connection
        x = x + self.mlp(self.ln_2(x))

        return x


class GPT(nn.Module):
    """
    GPT-style decoder-only transformer model.

    Architecture:
        Token Embedding
        + Positional Embedding
            ↓
        Transformer Block × n_layer
            ↓
        Final LayerNorm
            ↓
        Language Modeling Head (tied weights)

    Reference: @instructions/04_model_architecture.md
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize GPT model.

        Args:
            config: ModelConfig instance
        """
        super().__init__()
        self.config = config

        # Token and positional embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.context_length, config.n_embd)

        # Dropout for embeddings
        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                n_embd=config.n_embd,
                n_head=config.n_head,
                ffn_dim=config.ffn_dim,
                dropout=config.dropout,
                bias=config.bias,
                context_length=config.context_length,
            )
            for _ in range(config.n_layer)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Language modeling head
        if config.weight_tying:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            self.lm_head.weight = self.wte.weight
        else:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """
        Forward pass.

        Args:
            idx: Input token ids of shape (B, T)
            targets: Target token ids of shape (B, T), optional

        Returns:
            logits: Shape (B, T, vocab_size)
            loss: Cross-entropy loss if targets provided, else None
        """
        B, T = idx.size()

        assert T <= self.config.context_length, f"Sequence length {T} exceeds context length {self.config.context_length}"

        # Token and positional embeddings
        tok_emb = self.wte(idx)  # (B, T, n_embd)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # (T,)
        pos_emb = self.wpe(pos)  # (T, n_embd)
        x = self.drop(tok_emb + pos_emb)  # (B, T, n_embd)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)  # (B, T, n_embd)

        # Final layer norm
        x = self.ln_f(x)  # (B, T, n_embd)

        # LM head
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Cross-entropy loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            idx: Input token ids of shape (B, T)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated tokens of shape (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # If context exceeded, truncate from left
            idx_cond = idx if idx.size(1) <= self.config.context_length else idx[:, -self.config.context_length:]

            # Forward pass
            logits, _ = self(idx_cond)

            # Only use logits for last position
            logits = logits[:, -1, :] / temperature

            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample next token
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            idx = torch.cat([idx, idx_next], dim=1)

        return idx

# Pico-GPT Architecture

## Overview

Pico-GPT is a decoder-only transformer language model with approximately 35M parameters. It follows GPT architecture design principles with modern optimizations.

## Model Architecture

```
Input Token IDs (B, T)
        ↓
Token Embedding (50257 x 384)
Positional Embedding (128 x 384)
        ↓
Embedding Sum + Dropout
        ↓
┌─────────────────────────────────┐
│  Transformer Block × 6          │
│  ┌─────────────────────────┐    │
│  │ LN1 → Attention        │    │
│  │ (Fused QKV + Flash Attn)│    │
│  └─────────────────────────┘    │
│  ┌─────────────────────────┐    │
│  │ LN2 → MLP (4x hidden)  │    │
│  └─────────────────────────┘    │
└─────────────────────────────────┘
        ↓
Final LayerNorm
        ↓
LM Head (384 x 50257, weight-tied)
        ↓
Logits (B, T, 50257)
```

## Components

### Attention Module

- **Type:** Multi-head self-attention with causal masking
- **QKV Projection:** Fused single linear layer (3*C)
- **Flash Attention:** Enabled via PyTorch SDPA
- **Heads:** 6
- **Head Dim:** 64 (384 / 6)

### Transformer Block

- **Norm:** Pre-LayerNorm (before attention and MLP)
- **Residual Connections:** Both attention and MLP
- **MLP:** Linear → GELU → Linear (4x hidden dimension)

### Embeddings

- **Token Embedding:** Learned (vocab_size x n_embd)
- **Positional Embedding:** Learned (context_length x n_embd)
- **Weight Tying:** LM head shares weights with token embedding

## Training

### Optimization

- **Optimizer:** AdamW
- **Weight Decay:** 0.1 (excluded from LayerNorm/embeddings)
- **Learning Rate:** 3e-4 peak, cosine decay to 3e-5
- **Warmup:** 2K steps (linear)
- **Gradient Clipping:** 1.0
- **Mixed Precision:** BF16 (autocast)

### Data Pipeline

- **Format:** Memory-mapped binary shards (uint16)
- **Shard Size:** 5M tokens
- **Sampling:** Random window sampling
- **Gradient Accumulation:** Configurable (default: 8 micro-batches)

### Regularization

- **Dropout:** 0.1 (embeddings, residual connections, attention)
- **Label Smoothing:** None

## Inference

### Generation

- **Method:** Temperature sampling
- **Default Temperature:** 0.8
- **Stopping:** EOS token or max_new_tokens
- **Context Management:** Truncate from left when exceeding context_length

## Key Design Decisions

1. **Pre-LayerNorm:** Improves training stability
2. **Fused QKV:** Reduces memory allocations, improves efficiency
3. **Flash Attention:** Leverages optimized kernels on A100
4. **Learned Positional Embeddings:** Simple, effective for short context
5. **Weight Tying:** Reduces parameters, regularizes model

## References

- Attention Is All You Need (Vaswani et al., 2017)
- Improving Language Understanding by Generative Pre-Training (Radford et al., 2018)
- GPT-2: Language Models are Unsupervised Multitask Learners (Radford et al., 2019)
- nanoGPT (karpathy, 2023)

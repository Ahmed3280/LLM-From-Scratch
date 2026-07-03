# LLM From Scratch

A decoder-only Transformer implemented from scratch in PyTorch to understand how modern Large Language Models (LLMs) work under the hood.

The goal of this project was not to use existing LLM libraries, but to implement the core building blocks of a GPT-style language model and understand how they work together to perform next-token prediction.

## Features

- Character-level tokenizer
- Token & positional embeddings
- Scaled dot-product self-attention
- Multi-head self-attention
- Feed-forward networks (FFN)
- Residual connections
- Layer Normalization
- Decoder-only Transformer architecture
- Autoregressive text generation
- Training on the Tiny Shakespeare dataset

## Architecture

The model follows the standard decoder-only Transformer architecture:

```
Input Tokens
      │
      ▼
Token Embeddings + Positional Embeddings
      │
      ▼
N × Transformer Blocks
    ├── Multi-Head Self Attention
    ├── Feed Forward Network
    ├── Residual Connections
    └── Layer Normalization
      │
      ▼
Linear Language Modeling Head
      │
      ▼
Next Token Prediction
```

## What I Learned

This project helped me understand the implementation details behind modern LLMs, including:

- How embeddings represent tokens and positions.
- How Query, Key, and Value projections enable self-attention.
- Why causal masking is required for autoregressive generation.
- How multi-head attention captures different relationships simultaneously.
- Why residual connections and Layer Normalization stabilize training.
- How stacking Transformer blocks increases the model's expressive power.
- How GPT-style models generate text one token at a time.

## Tech Stack

- Python
- PyTorch

## Future Improvements

- Top-k / Top-p sampling
- Temperature scaling
- Model checkpointing
- Byte Pair Encoding (BPE) tokenizer
- LoRA fine-tuning
- Hugging Face compatibility

## References

- Andrej Karpathy — Let's Build GPT
- Attention Is All You Need (2017)

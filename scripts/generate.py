"""
Text generation script for Pico-GPT.

Reference: @instructions/06_text_generation.md
"""
import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))
from pico_gpt.model import GPT
from pico_gpt.config import ModelConfig
from pico_gpt.tokenizer import GPT2Tokenizer
import torch


def load_model(checkpoint_path: str) -> tuple[GPT, GPT2Tokenizer]:
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint

    Returns:
        Model and tokenizer
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Use default config (always available from pico_gpt.config)
    config = ModelConfig()

    # Create model
    model = GPT(config)

    # Load state dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Create tokenizer
    tokenizer = GPT2Tokenizer()

    return model, tokenizer


def generate(
    model: GPT,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    """
    Generate text from prompt.

    Reference: @instructions/06_text_generation.md

    Args:
        model: Trained model
        tokenizer: GPT-2 tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Generated text
    """
    # Encode prompt
    tokens = tokenizer.encode(prompt)

    # Truncate if longer than context
    tokens = tokenizer.truncate(tokens, model.config.context_length)

    # Convert to tensor
    x = torch.tensor([tokens], dtype=torch.long)

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    x = x.to(device)

    # Generate
    with torch.no_grad():
        output = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature)

    # Decode
    generated_tokens = output[0].tolist()
    generated_text = tokenizer.decode(generated_tokens)

    return generated_text


def main():
    parser = argparse.ArgumentParser(description="Generate text with Pico-GPT")
    parser.add_argument("--model", type=str, default="checkpoints/best_model.pt", help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="The future of artificial intelligence is", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    model, tokenizer = load_model(args.model)

    print(f"Generating with temperature {args.temperature}...")
    print(f"Prompt: {args.prompt}")
    print("-" * 50)

    generated = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    print(generated)
    print("-" * 50)


if __name__ == "__main__":
    main()

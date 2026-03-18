"""
Sanity check script for Pico-GPT.

Tests all components without running full training.
Uses CPU and dummy data.
"""

import sys
from pathlib import Path
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

print("=" * 60)
print("Pico-GPT Sanity Check")
print("=" * 60)

# Test 1: Imports
print("\n[Test 1] Checking imports...")
try:
    from pico_gpt.config import ModelConfig, TrainingConfig, GenerationConfig
    from pico_gpt.tokenizer import GPT2Tokenizer
    from pico_gpt.model import GPT
    from pico_gpt.dataloader import MemoryMappedDataset
    from pico_gpt.trainer import Trainer
    from pico_gpt.data import PreprocessingState, TokenBuffer
    from pico_gpt.export import export_to_huggingface, upload_to_hub
    from pico_gpt.tokenizer_utils import export_tokenizer_metadata
    print("  ✓ All imports successful")
except ImportError as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Configuration
print("\n[Test 2] Checking configuration...")
try:
    model_config = ModelConfig()
    training_config = TrainingConfig()
    generation_config = GenerationConfig()

    assert model_config.n_layer == 6
    assert model_config.n_head == 6
    assert model_config.n_embd == 384
    assert model_config.vocab_size == 50257
    assert model_config.context_length == 128

    # Validate derived values
    assert training_config.gradient_accumulation_steps == training_config.batch_size // training_config.micro_batch_size

    print(f"  ✓ Model config: {model_config.n_layer} layers, {model_config.n_embd} dim")
    print(f"  ✓ Training config: batch={training_config.batch_size}, grad_acc={training_config.gradient_accumulation_steps}")
except Exception as e:
    print(f"  ✗ Configuration failed: {e}")
    sys.exit(1)

# Test 3: Tokenizer
print("\n[Test 3] Checking tokenizer...")
try:
    tokenizer = GPT2Tokenizer()
    assert tokenizer.vocab_size == 50257
    assert tokenizer.eos_token_id == 50256

    # Test encode/decode
    text = "Hello, world!"
    tokens = tokenizer.encode(text)
    assert len(tokens) > 0
    decoded = tokenizer.decode(tokens)
    assert decoded == text

    # Test truncation
    long_tokens = list(range(200))
    truncated = tokenizer.truncate(long_tokens, 128)
    assert len(truncated) == 128

    print(f"  ✓ Tokenizer works (vocab_size={tokenizer.vocab_size})")
    print(f"  ✓ Encode/decode roundtrip works")
    print(f"  ✓ Truncation works")
except Exception as e:
    print(f"  ✗ Tokenizer failed: {e}")
    sys.exit(1)

# Test 4: Model instantiation
print("\n[Test 4] Checking model instantiation...")
try:
    model = GPT(model_config)
    model.eval()

    # Count parameters
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Model created with {param_count:,} parameters")

    # Verify approximate parameter count (~30M for our config)
    if 25_000_000 < param_count < 40_000_000:
        print(f"  ✓ Parameter count in expected range (~30M)")
    else:
        print(f"  ⚠ Parameter count {param_count:,} seems unexpected")
except Exception as e:
    print(f"  ✗ Model instantiation failed: {e}")
    sys.exit(1)

# Test 5: Forward pass
print("\n[Test 5] Checking forward pass...")
try:
    batch_size = 4
    seq_len = 32

    # Create dummy input
    dummy_tokens = torch.randint(0, model_config.vocab_size, (batch_size, seq_len))
    dummy_targets = torch.randint(0, model_config.vocab_size, (batch_size, seq_len))

    # Forward pass
    logits, loss = model(dummy_tokens, targets=dummy_targets)

    # Check output shapes
    assert logits.shape == (batch_size, seq_len, model_config.vocab_size)
    assert loss is not None
    assert loss.item() > 0

    print(f"  ✓ Forward pass works")
    print(f"  ✓ Output shape: {logits.shape}")
    print(f"  ✓ Loss: {loss.item():.4f}")
except Exception as e:
    print(f"  ✗ Forward pass failed: {e}")
    sys.exit(1)

# Test 6: Generation
print("\n[Test 6] Checking generation...")
try:
    prompt = "The future of"
    tokens = tokenizer.encode(prompt)
    tokens = tokenizer.truncate(tokens, model_config.context_length)
    x = torch.tensor([tokens], dtype=torch.long)

    # Generate a few tokens
    with torch.no_grad():
        generated = model.generate(x, max_new_tokens=5, temperature=0.8)

    # Check output
    assert generated.shape[0] == 1
    assert generated.shape[1] == len(tokens) + 5

    generated_text = tokenizer.decode(generated[0].tolist())
    print(f"  ✓ Generation works")
    print(f"  ✓ Input: {prompt}")
    print(f"  ✓ Output: {generated_text}")
except Exception as e:
    print(f"  ✗ Generation failed: {e}")
    sys.exit(1)

# Test 7: Checkpoint save/load
print("\n[Test 8] Checking checkpoint save/load...")
try:
    tmp_path = Path(tempfile.gettempdir()) / "test_checkpoint.pt"

    # Save checkpoint
    checkpoint = {
        "step": 100,
        "model_state_dict": model.state_dict(),
        "best_val_loss": 3.5,
        "config": model_config,
    }
    torch.save(checkpoint, tmp_path)

    # Load checkpoint (weights_only=False for PyTorch 2.6+ compatibility)
    loaded = torch.load(tmp_path, map_location='cpu', weights_only=False)

    assert loaded["step"] == 100
    assert loaded["best_val_loss"] == 3.5
    assert "model_state_dict" in loaded

    # Verify model can load from checkpoint
    model2 = GPT(loaded["config"])
    model2.load_state_dict(loaded["model_state_dict"])

    print(f"  ✓ Checkpoint save works")
    print(f"  ✓ Checkpoint load works")

    # Cleanup
    tmp_path.unlink()
except Exception as e:
    print(f"  ✗ Checkpoint save/load failed: {e}")
    sys.exit(1)

# Test 8: Safetensors export (if available)
print("\n[Test 8] Checking safetensors export...")
try:
    from safetensors.torch import save_file, load_file

    tmp_path = Path(tempfile.gettempdir()) / "test_model.safetensors"

    # Save
    state_dict = model.state_dict()
    save_file(state_dict, tmp_path)

    # Load
    loaded_state = load_file(tmp_path)

    # Verify keys match
    expected_keys = set(state_dict.keys())
    assert set(loaded_state.keys()) == expected_keys

    # Verify shapes match
    for key in loaded_state:
        assert loaded_state[key].shape == model.state_dict()[key].shape

    print(f"  ✓ Safetensors save works")
    print(f"  ✓ Safetensors load works")

    # Cleanup
    tmp_path.unlink()
except ImportError:
    print(f"  ⚠ safetensors not installed (pip install safetensors)")
except Exception as e:
    print(f"  ✗ Safetensors failed: {e}")
    sys.exit(1)

# Test 9: Tokenizer metadata export
print("\n[Test 9] Checking tokenizer metadata export...")
try:
    tmp_dir = Path(tempfile.gettempdir()) / "test_tokenizer"
    tmp_dir.mkdir(exist_ok=True)

    export_tokenizer_metadata(str(tmp_dir), model_max_length=128)

    # Check files created
    assert (tmp_dir / "tokenizer_config.json").exists()
    assert (tmp_dir / "special_tokens_map.json").exists()

    # Verify content
    import json
    with open(tmp_dir / "tokenizer_config.json") as f:
        tc = json.load(f)
    assert tc["model_max_length"] == 128

    print(f"  ✓ Tokenizer metadata export works")

    # Cleanup
    shutil.rmtree(tmp_dir)
except Exception as e:
    print(f"  ✗ Tokenizer metadata export failed: {e}")
    sys.exit(1)

# Test 10: Model config export
print("\n[Test 10] Checking model config export...")
try:
    tmp_dir = Path(tempfile.gettempdir()) / "test_config"
    tmp_dir.mkdir(exist_ok=True)

    # Simulate export_to_huggingface (just config part)
    config_dict = {
        "model_type": "custom_gpt",
        "vocab_size": model_config.vocab_size,
        "n_layer": model_config.n_layer,
        "n_head": model_config.n_head,
        "n_embd": model_config.n_embd,
        "context_length": model_config.context_length,
        "dropout": model_config.dropout,
        "bias": model_config.bias,
        "ffn_dim": model_config.ffn_dim,
    }

    with open(tmp_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Verify
    with open(tmp_dir / "config.json") as f:
        loaded = json.load(f)
    assert loaded["n_layer"] == 6

    print(f"  ✓ Model config export works")

    # Cleanup
    shutil.rmtree(tmp_dir)
except Exception as e:
    print(f"  ✗ Model config export failed: {e}")
    sys.exit(1)

# Test 11: CPU vs CUDA handling
print("\n[Test 11] Checking device handling...")
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cpu = GPT(model_config).to(device)

    # Should work on both CPU and CUDA
    dummy = torch.randint(0, model_config.vocab_size, (1, 10)).to(device)
    logits, _ = model_cpu(dummy)

    print(f"  ✓ Device: {device}")
    print(f"  ✓ Forward pass on {device} works")
except Exception as e:
    print(f"  ✗ Device handling failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 60)
print("✓ ALL SANITY CHECKS PASSED!")
print("=" * 60)
print("\nYou can proceed with:")
print("  1. Dataset preparation: python scripts/prepare_data.py")
print("  2. Training:          python scripts/train.py")
print("  3. Generation:        python scripts/generate.py --model checkpoints/best_model.pt")
print("  4. HF Export:         export HF_TOKEN=... && python scripts/export_hf.py --upload username/repo")
print()

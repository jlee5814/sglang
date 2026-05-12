"""
Verify an MLX-backend optimization produces output matching the unmodified baseline.

Usage:
    python scripts/mlx/verify_fusion.py \\
        --patch sglang.srt.hardware_backend.mlx.moe.fused_switch_glu:patch_switch_glu_with_fused_up_gate \\
        --model mlx-community/Qwen3-30B-A3B-4bit \\
        --prompt "The quick brown fox" \\
        --tolerance 0.0

Currently validates:
- PR #24712 FusedSwitchUpGate (bit-exact, tolerance=0.0)

Will validate:
- PR 2 custom Metal grouped GEMM (tolerance TBD; likely ~1e-3 due to accumulation order)
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Callable

import mlx.core as mx
from mlx_lm import load


def resolve_patch_fn(spec: str) -> Callable:
    """Resolve 'module.path:function_name' to a callable."""
    module_path, fn_name = spec.split(":")
    module = importlib.import_module(module_path)
    return getattr(module, fn_name)


def run_one_pass(model, tokenizer, prompt: str) -> mx.array:
    """Encode prompt and run one forward pass; return logits."""
    tokens = mx.array(tokenizer.encode(prompt))
    tokens = mx.expand_dims(tokens, 0)
    logits = model(tokens)
    mx.eval(logits)
    return logits


def verify(
    patch_fn: Callable,
    model_path: str,
    prompt: str,
    tolerance: float,
) -> int:
    """Load model twice (baseline vs patched), compare logits, return exit code."""
    print(f"Loading baseline ({model_path})...")
    model_baseline, tokenizer = load(model_path)

    print(f"Running baseline forward pass...")
    logits_baseline = run_one_pass(model_baseline, tokenizer, prompt)
    print(f"  shape: {logits_baseline.shape}")
    print(f"  last-token logits[:5]: {logits_baseline[0, -1, :5]}")

    print(f"\nLoading patched ({patch_fn.__module__}.{patch_fn.__name__})...")
    model_patched, _ = load(model_path)
    n_patched = patch_fn(model_patched)
    print(f"  patched {n_patched} sites")

    print(f"Running patched forward pass...")
    logits_patched = run_one_pass(model_patched, tokenizer, prompt)
    print(f"  shape: {logits_patched.shape}")
    print(f"  last-token logits[:5]: {logits_patched[0, -1, :5]}")

    diff = mx.abs(logits_baseline - logits_patched)
    max_diff = mx.max(diff).item()
    mean_diff = mx.mean(diff).item()

    print(f"\n=== Comparison ===")
    print(f"max abs diff:  {max_diff:.6e}")
    print(f"mean abs diff: {mean_diff:.6e}")
    print(f"tolerance:     {tolerance:.6e}")

    if max_diff <= tolerance:
        status = "BIT-EXACT" if tolerance == 0.0 else "WITHIN TOLERANCE"
        print(f"status: {status}")
        return 0
    print(f"status: FAIL")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patch", required=True,
                        help="patch function spec: 'module.path:function_name'")
    parser.add_argument("--model", default="mlx-community/Qwen3-30B-A3B-4bit")
    parser.add_argument("--prompt", default="The quick brown fox")
    parser.add_argument("--tolerance", type=float, default=0.0,
                        help="max abs diff threshold; 0.0 requires bit-exact")
    args = parser.parse_args()

    # Make the SGLang fork importable without installing it.
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "python"))

    patch_fn = resolve_patch_fn(args.patch)
    return verify(patch_fn, args.model, args.prompt, args.tolerance)


if __name__ == "__main__":
    sys.exit(main())

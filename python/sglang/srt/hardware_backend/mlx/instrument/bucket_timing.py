"""Per-layer attention vs MoE bucket timing for decode-step kernel attribution.

Activated by SGLANG_MLX_BUCKET_TIMING=1.

Patches every Qwen3MoeDecoderLayer instance via __class__ reassignment to a
subclass that times the attention half and the MoE half separately, forcing
mx.eval at each boundary to commit the lazy graph. Accumulates into
module-level buckets; dumps a sorted table at session end via atexit.

WARNING: each forced mx.eval kills async overlap. Decode latencies measured
under this patch are upper bounds, NOT representative of production
throughput. Use for kernel-attribution research only.

Tunables:
  SGLANG_MLX_BUCKET_TIMING=1         enable the patch
  SGLANG_MLX_BUCKET_WARMUP_SKIP=5    drop the first N calls per bucket in dump
"""

from __future__ import annotations

import atexit
import hashlib
import inspect
import logging
import os
import time
from collections import defaultdict
from typing import Any, Optional

import mlx.core as mx

logger = logging.getLogger(__name__)

# SHA256 of mlx_lm.models.qwen3_moe.Qwen3MoeDecoderLayer.__call__ source as of
# the version this patch was authored against. The wrapper reimplements the
# body verbatim to insert timing between halves. If mlx-lm changes the layer,
# this hash check fails and the patch refuses to install rather than silently
# diverge.
EXPECTED_DECODER_LAYER_CALL_HASH = (
    "937a607a6e7ea3c9c64d306b406009d0c0f5a32e1f9a00c792ffe7dc669ec57a"
)
EXPECTED_MOE_BLOCK_CALL_HASH = (
    "fffa3169fdc28428856db86b740ad4d284bd5a54cd73207fe95fd39271477925"
)

_buckets: dict[str, list[float]] = defaultdict(list)
_dump_registered: bool = False
_timed_class_cache: dict[type, type] = {}


def _build_timed_class(base_cls: type) -> type:
    """Subclass base_cls with a timing-instrumented __call__.

    Mirrors the upstream Qwen3MoeDecoderLayer.__call__ body verbatim, with
    mx.eval barriers at the attention/MoE split and at the end of the layer
    so each bucket records cleanly-attributed wall time.
    """
    if base_cls in _timed_class_cache:
        return _timed_class_cache[base_cls]

    class TimedLayer(base_cls):
        def __call__(
            self,
            x: mx.array,
            mask: Optional[mx.array] = None,
            cache: Optional[Any] = None,
        ) -> mx.array:
            t0 = time.perf_counter()
            r = self.self_attn(self.input_layernorm(x), mask, cache)
            h = x + r
            mx.eval(h)
            t1 = time.perf_counter()
            _buckets["attention"].append(t1 - t0)

            r = self.mlp(self.post_attention_layernorm(h))
            out = h + r
            mx.eval(out)
            t2 = time.perf_counter()
            _buckets["moe_block"].append(t2 - t1)

            return out

    TimedLayer.__name__ = f"Timed{base_cls.__name__}"
    TimedLayer.__qualname__ = TimedLayer.__name__
    _timed_class_cache[base_cls] = TimedLayer
    return TimedLayer


def _build_timed_moe_class(base_cls: type) -> type:
    """Subclass base_cls (Qwen3MoeSparseMoeBlock) with timed __call__.

    Splits the MoE block into three buckets:
      moe_routing  — gate, softmax, top-k partition, take, normalize
      switch_mlp   — the 3 gather_qmm dispatches (up, gate, down)
      moe_combine  — weighted sum across top-k experts

    Mirrors the upstream Qwen3MoeSparseMoeBlock.__call__ body verbatim, with
    one rewrite: `scores /= sum(...)` becomes `scores = scores / sum(...)`.
    mx.array is immutable so the operator forms are equivalent, but the
    explicit form is clearer in a timing wrapper.
    """
    if base_cls in _timed_class_cache:
        return _timed_class_cache[base_cls]

    class TimedMoEBlock(base_cls):
        def __call__(self, x: mx.array) -> mx.array:
            # Routing: 5 sub-buckets, plus umbrella for cross-stage validation.
            t0 = time.perf_counter()
            gates = self.gate(x)
            mx.eval(gates)
            t1 = time.perf_counter()
            _buckets["routing_gate"].append(t1 - t0)

            gates = mx.softmax(gates, axis=-1, precise=True)
            mx.eval(gates)
            t2 = time.perf_counter()
            _buckets["routing_softmax"].append(t2 - t1)

            k = self.top_k
            inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
            mx.eval(inds)
            t3 = time.perf_counter()
            _buckets["routing_argpartition"].append(t3 - t2)

            scores = mx.take_along_axis(gates, inds, axis=-1)
            mx.eval(scores)
            t4 = time.perf_counter()
            _buckets["routing_take"].append(t4 - t3)

            if self.norm_topk_prob:
                scores = scores / mx.sum(scores, axis=-1, keepdims=True)
                mx.eval(scores)
                t5 = time.perf_counter()
                _buckets["routing_normalize"].append(t5 - t4)
            else:
                t5 = t4

            # Umbrella bucket: equals sub-sum by construction (telescoping).
            # Cross-stage validator against Stage 2's moe_routing measurement.
            _buckets["moe_routing"].append(t5 - t0)

            y = self.switch_mlp(x, inds)
            mx.eval(y)
            t6 = time.perf_counter()
            _buckets["switch_mlp"].append(t6 - t5)

            y = (y * scores[..., None]).sum(axis=-2)
            mx.eval(y)
            t7 = time.perf_counter()
            _buckets["moe_combine"].append(t7 - t6)

            return y

    TimedMoEBlock.__name__ = f"Timed{base_cls.__name__}"
    TimedMoEBlock.__qualname__ = TimedMoEBlock.__name__
    _timed_class_cache[base_cls] = TimedMoEBlock
    return TimedMoEBlock


def patch_bucket_timing(model: Any) -> int:
    """Install bucket timing on Qwen3MoE decoder layers and MoE blocks.

    Two independent passes guarded by separate hash checks:
      1. Decoder-layer pass — splits each layer into attention vs moe_block
      2. MoE-block pass     — splits each MoE block into routing / switch_mlp / combine

    Each pass is gated by its own source-hash check. A failed check skips
    only that pass; the other still runs. Idempotent. Registers an atexit
    dump on first successful patch.

    Returns the total count (decoder layers + MoE blocks) patched.
    """
    try:
        from mlx_lm.models.qwen3_moe import (
            Qwen3MoeDecoderLayer,
            Qwen3MoeSparseMoeBlock,
        )
    except ImportError:
        logger.warning("patch_bucket_timing: mlx_lm.qwen3_moe not importable")
        return 0

    layer_list = getattr(getattr(model, "model", model), "layers", None)
    if not layer_list:
        logger.warning("patch_bucket_timing: no model.layers found")
        return 0

    # Decoder-layer pass
    decoder_hash = hashlib.sha256(
        inspect.getsource(Qwen3MoeDecoderLayer.__call__).encode()
    ).hexdigest()
    decoder_patched = 0
    if decoder_hash != EXPECTED_DECODER_LAYER_CALL_HASH:
        logger.error(
            "patch_bucket_timing: Qwen3MoeDecoderLayer.__call__ hash mismatch."
        )
        logger.error(f"  expected: {EXPECTED_DECODER_LAYER_CALL_HASH}")
        logger.error(f"  actual:   {decoder_hash}")
        logger.error(
            "  mlx_lm version change broke decoder-layer patch semantics. "
            "Skipping decoder-layer instrumentation."
        )
    else:
        decoder_timed_cls: Optional[type] = None
        for layer in layer_list:
            if not isinstance(layer, Qwen3MoeDecoderLayer):
                continue
            if type(layer).__name__.startswith("Timed"):
                continue
            if decoder_timed_cls is None:
                decoder_timed_cls = _build_timed_class(type(layer))
            layer.__class__ = decoder_timed_cls
            decoder_patched += 1

    # MoE-block pass
    moe_hash = hashlib.sha256(
        inspect.getsource(Qwen3MoeSparseMoeBlock.__call__).encode()
    ).hexdigest()
    moe_patched = 0
    if moe_hash != EXPECTED_MOE_BLOCK_CALL_HASH:
        logger.error(
            "patch_bucket_timing: Qwen3MoeSparseMoeBlock.__call__ hash mismatch."
        )
        logger.error(f"  expected: {EXPECTED_MOE_BLOCK_CALL_HASH}")
        logger.error(f"  actual:   {moe_hash}")
        logger.error(
            "  mlx_lm version change broke MoE-block patch semantics. "
            "Skipping MoE-block instrumentation."
        )
    else:
        moe_timed_cls: Optional[type] = None
        for layer in layer_list:
            mlp = getattr(layer, "mlp", None)
            if mlp is None or not isinstance(mlp, Qwen3MoeSparseMoeBlock):
                continue
            if type(mlp).__name__.startswith("Timed"):
                continue
            if moe_timed_cls is None:
                moe_timed_cls = _build_timed_moe_class(type(mlp))
            mlp.__class__ = moe_timed_cls
            moe_patched += 1

    total = decoder_patched + moe_patched
    if total == 0:
        logger.warning("patch_bucket_timing: nothing patched")
        return 0

    global _dump_registered
    if not _dump_registered:
        atexit.register(dump_buckets)
        _dump_registered = True

    logger.info(
        f"patch_bucket_timing: instrumented {decoder_patched} decoder layers, "
        f"{moe_patched} MoE blocks. "
        "Forced mx.eval at every bucket boundary — decode latencies are upper bounds."
    )
    return total


def dump_buckets() -> None:
    """Print a sorted table of bucket totals.

    Drops the first SGLANG_MLX_BUCKET_WARMUP_SKIP calls per bucket (default 5)
    to exclude warmup contamination — first decode step after prefill drains
    the lazy graph and is artificially heavy. Sorted by total_ms desc.
    """
    if not _buckets:
        return

    warmup_skip = int(os.environ.get("SGLANG_MLX_BUCKET_WARMUP_SKIP", "5"))

    rows = []
    for name, dts in _buckets.items():
        if len(dts) <= warmup_skip:
            continue
        dts = dts[warmup_skip:]
        n = len(dts)
        total_ms = sum(dts) * 1000.0
        mean_ms = total_ms / n
        sorted_dts = sorted(dts)
        median_ms = sorted_dts[n // 2] * 1000.0
        rows.append((name, n, total_ms, mean_ms, median_ms))
    rows.sort(key=lambda r: -r[2])

    if not rows:
        print(
            f"\n[SGLANG_MLX_BUCKET_TIMING] no buckets with > {warmup_skip} "
            "calls after warmup skip; nothing to report."
        )
        return

    print(
        f"\n[SGLANG_MLX_BUCKET_TIMING] decode-step kernel attribution "
        f"(skipped first {warmup_skip} calls per bucket)"
    )
    print(
        f"{'bucket':<20} {'calls':>8} {'total_ms':>12} "
        f"{'mean_ms':>10} {'median_ms':>12}"
    )
    for name, n, total_ms, mean_ms, median_ms in rows:
        print(
            f"{name:<20} {n:>8d} {total_ms:>12.2f} "
            f"{mean_ms:>10.3f} {median_ms:>12.3f}"
        )

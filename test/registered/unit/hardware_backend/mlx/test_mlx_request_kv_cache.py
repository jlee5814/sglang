"""Unit tests for request KV buffer sizing and the reuse pool bound."""

from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_mlx_ci

register_mlx_ci(est_time=1, suite="stage-a-unit-test-mlx")

_HAS_MLX = importlib.util.find_spec("mlx") is not None
_SKIP_REASON = "requires mlx"

if _HAS_MLX:
    import mlx.core as mx

    from sglang.srt.environ import envs
    from sglang.srt.hardware_backend.mlx.kv_cache import (
        ContiguousAttentionKVCache,
        MlxModelCacheLayout,
    )
    from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner


def _runner(num_layers: int = 2, pool_size: int = 8, init_tokens: int = 32):
    runner = object.__new__(MlxModelRunner)
    layers = [SimpleNamespace(self_attn=object()) for _ in range(num_layers)]
    runner._cache_layout = MlxModelCacheLayout.from_attention_discovery(
        layers, ["self_attn"] * num_layers
    )
    runner._max_seq_len = init_tokens
    runner._cache_pool = []
    runner._cache_pool_size = pool_size
    return runner


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestRequestKvCacheSizing(unittest.TestCase):
    def test_defaults_start_small_and_bound_the_pool(self):
        self.assertEqual(envs.SGLANG_MLX_REQ_KV_INIT_TOKENS.get(), 256)
        self.assertEqual(envs.SGLANG_MLX_REQ_KV_POOL_SIZE.get(), 8)

    def test_new_cache_uses_the_runner_initial_capacity(self):
        runner = _runner(init_tokens=32)
        cache = runner._new_native_cache()
        self.assertEqual([c.max_seq_len for c in cache], [32, 32])

    def test_prefill_longer_than_capacity_grows_and_keeps_every_token(self):
        cache = ContiguousAttentionKVCache(max_seq_len=4)
        n = 11
        keys = mx.arange(n, dtype=mx.float32).reshape(1, 1, n, 1)
        k, v = cache.update_and_fetch(keys, -keys)
        mx.eval(k, v)
        self.assertEqual(cache.offset, n)
        self.assertEqual(cache.max_seq_len, 16)
        self.assertEqual(k.shape, (1, 1, n, 1))
        self.assertEqual(k[0, 0, :, 0].tolist(), list(range(n)))
        self.assertEqual(v[0, 0, :, 0].tolist(), [-float(t) for t in range(n)])


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestRequestKvCachePool(unittest.TestCase):
    def test_release_keeps_at_most_pool_size_lists(self):
        runner = _runner(pool_size=2)
        for _ in range(5):
            runner._release_cache(runner._new_native_cache())
        self.assertEqual(len(runner._cache_pool), 2)

    def test_pool_size_zero_disables_reuse(self):
        runner = _runner(pool_size=0)
        runner._release_cache(runner._new_native_cache())
        self.assertEqual(runner._cache_pool, [])

    def test_pooled_list_is_reused_and_reset(self):
        runner = _runner(pool_size=2)
        cache = runner._new_native_cache()
        keys = mx.zeros((1, 1, 3, 2), dtype=mx.float32)
        for layer in cache:
            layer.update_and_fetch(keys, keys)
        runner._release_cache(cache)
        reused = runner._acquire_cache()
        self.assertIs(reused, cache)
        self.assertEqual([c.offset for c in reused], [0, 0])
        self.assertEqual(runner._cache_pool, [])


if __name__ == "__main__":
    unittest.main()

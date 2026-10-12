## Summary

Every PLE gather rebuilds shard offsets and checks tensor dtypes. Cache the offsets and layouts at model load. Skip repeated checks for uniform tables. Keep the fallback when touched shards differ in dtype, bit width or group size.

## Validation

M5 Max, 128 GiB; Python 3.12.14; MLX 0.32.3. Base `f3ad34d`; head `ad9ffe0`.

Synthetic 128-shard affine PLE: 5 bits, group 32, width 160. Three runs in forward/reverse/forward order; five warmups per case; 300 samples for 16/64 rows; 50 for 32000; warm OS page cache. Values below are medians of run medians.

| Case | main (ms) | branch (ms) |
|---|---:|---:|
| plan_16 | 0.0175 | 0.0047 |
| plan_64 | 0.0426 | 0.0101 |
| plan_32000 | 0.8677 | 0.7815 |

Tests cover dense/affine tables, uneven shard sizes, duplicate indices and mixed weight/scale/bias dtypes, bit widths and group sizes.

Real model: `mlx-community/Qwen3.8-Flash-Next-Uncensored-oQ5e-mtp`, revision `c7d915c`; MTP depth 3, TQ4 KV, PLE SSD, greedy, thinking off, concurrency 1. Five tasks at 2k/8k, twice; one unscored 8k/256-token warmup; zero cached prompt tokens. All 16 coding outputs passed functional tests. The generated list-based LRU violates O(1) on main and branch.

8k coding medians (main → branch): prefill 2036 → 2088 tok/s; decode 99.9 → 102.4 tok/s; client TTFT 3.947 → 3.844 s. Sampled process memory stayed near 88 GiB.

The gain applies to host planning. Full-model decode remains within the observed run variation.

Reproduce: install this branch from source, then run [ple_micro.py](https://github.com/Ambisphaeric/omlx/blob/perf/ple-benchmark-evidence/experiments/qwen4-ple/ple_micro.py) with `python ple_micro.py --out result.json`. [All timings and memory samples](https://github.com/Ambisphaeric/omlx/blob/perf/ple-benchmark-evidence/experiments/qwen4-ple/micro-summary.json); [real-model setup and results](https://github.com/Ambisphaeric/omlx/blob/perf/ple-benchmark-evidence/experiments/qwen4-ple/source-summary.json).

Focused PLE tests: 106 passed on the combined build. Default suite (`pytest -n 3 --dist loadgroup`): 18,047 passed, 238 skipped, five DSpark failures. The same five failures reproduce on unchanged main (`test_deepseek_v4_dspark.py`: 70 passed, five failed). Black and Ruff pass for the changed tests. Combined real-model prefix reuse passed with 7,993 cached tokens.

## Checklist

- [x] I read CONTRIBUTING.md.
- [x] I understand each change and can explain it in review.

## Summary

Large PLE gathers create one Future for each missing page. Split pages across at most 48 tasks. Keep 48-way read parallelism, short-read retries and task draining before the file can close. Mark pages only after all tasks succeed.

## Validation

M5 Max, 128 GiB; Python 3.12.14; MLX 0.32.3. Base `f3ad34d`; head `06c5d4d`.

Synthetic 128-shard affine PLE: 5 bits, group 32, width 160. Three runs in forward/reverse/forward order; five warmups per case; 100 samples for 16/128 pages; 50 for 2048; warm OS page cache. Values below are medians of run medians.

| Case | main (ms) | branch (ms) |
|---|---:|---:|
| page_dispatch_16 | 0.1794 | 0.1830 |
| page_dispatch_128 | 1.0772 | 0.8864 |
| page_dispatch_2048 | 16.0877 | 11.6100 |

Tests cover bounded submissions, short reads, warm skips, failed batches, partial submission and close during failed reads.

Real model: `mlx-community/Qwen3.8-Flash-Next-Uncensored-oQ5e-mtp`, revision `c7d915c`; MTP depth 3, TQ4 KV, PLE SSD, greedy, thinking off, concurrency 1. Five tasks at 2k/8k, twice; one unscored 8k/256-token warmup; zero cached prompt tokens. All 16 coding outputs passed functional tests. The generated list-based LRU violates O(1) on main and branch.

8k coding medians (main → branch): prefill 2036 → 2074 tok/s; decode 99.9 → 103.3 tok/s; client TTFT 3.947 → 3.876 s. Sampled process memory stayed near 88 GiB.

The 16-page case took 0.0036 ms longer. The gain is clearest for large missing-page sets. Small full-model rate differences do not establish a causal gain.

Reproduce: install this branch from source, then run [ple_micro.py](https://github.com/Ambisphaeric/omlx/blob/perf/ple-benchmark-evidence/experiments/qwen4-ple/ple_micro.py) with `python ple_micro.py --out result.json`. [All timings and memory samples](https://github.com/Ambisphaeric/omlx/blob/perf/ple-benchmark-evidence/experiments/qwen4-ple/micro-summary.json); [real-model setup and results](https://github.com/Ambisphaeric/omlx/blob/perf/ple-benchmark-evidence/experiments/qwen4-ple/source-summary.json).

Focused PLE tests: 106 passed on the combined build. Default suite (`pytest -n 3 --dist loadgroup`): 18,047 passed, 238 skipped, five DSpark failures. The same five failures reproduce on unchanged main (`test_deepseek_v4_dspark.py`: 70 passed, five failed). Black and Ruff pass for the changed tests. Combined real-model prefix reuse passed with 7,993 cached tokens.

## Checklist

- [x] I read CONTRIBUTING.md.
- [x] I understand each change and can explain it in review.

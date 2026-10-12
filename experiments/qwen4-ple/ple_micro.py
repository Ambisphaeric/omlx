"""Measure the unchanged production PLE methods on a 128-shard affine table.

Run with the editable oMLX environment: python ple_micro.py --out result.json.
The page cache is warm. The page-read bitmap is reset for the dispatch test.
"""
import argparse
import inspect
import json
import resource
import statistics
import struct
import tempfile
import time
from pathlib import Path

import numpy as np
from omlx.patches import mlx_vlm_qwen4_exp_compat as compat

compat.apply_mlx_vlm_qwen4_exp_compat_patch()
from mlx_vlm.models.qwen4_exp.language import DiskBackedShardedEmbedding, _SafeTensorMMap, _PLE_PAGE_SIZE


def measure(fn, repeats):
    for _ in range(5):
        fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        fn()
        times.append((time.perf_counter_ns()-start)/1e6)
    return dict(median_ms=statistics.median(times), p95_ms=float(np.percentile(times, 95)),
                repeats=repeats)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    args = p.parse_args()
    rng = np.random.default_rng(773)
    shards, rows, dims = 128, 8192, 160
    prefix = 'model.language_model.layers.2.ple.ple_embedding.ngram_embedding'
    header, arrays, weight_map = {}, [], {}
    offset = 0
    for shard in range(shards):
        for family, dtype, width in [('weight', 'U32', 25), ('scales', 'BF16', 5), ('biases', 'BF16', 5)]:
            key = f'{prefix}.shard_{shard}.{family}'
            array = rng.integers(0, 65535, size=(rows, width), dtype=np.uint32 if dtype=='U32' else np.uint16)
            header[key] = dict(dtype=dtype, shape=list(array.shape), data_offsets=[offset, offset+array.nbytes])
            offset += array.nbytes
            arrays.append(array)
            weight_map[key] = 'table.safetensors'
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        metadata = json.dumps(header).encode()
        with (root/'table.safetensors').open('wb') as f:
            f.write(struct.pack('<Q', len(metadata)))
            f.write(metadata)
            for array in arrays:
                f.write(array.tobytes())
        del arrays
        (root/'model.safetensors.index.json').write_text(json.dumps(dict(weight_map=weight_map)))
        embedding = DiskBackedShardedEmbedding(root, prefix, shards*rows, dims, shards)
        result = dict(table=dict(shards=shards, rows_per_shard=rows, dims=dims, bits=5, group_size=32),
                      source=inspect.getfile(DiskBackedShardedEmbedding),
                      warmups=5, page_cache='warm', bitmap='reset for page_dispatch')
        try:
            for count in [16, 64, 32000, 128000]:
                host = rng.integers(0, shards*rows, size=count, dtype=np.int64)
                result[f'plan_{count}'] = measure(lambda: embedding._plan(host), 300 if count<100 else 50)
                plan = embedding._plan(host)
                # Prime every page before the warm assembly test.
                embedding._assemble(host, plan)
                result[f'assemble_{count}'] = measure(lambda: embedding._assemble(host, plan), 100 if count<100 else 30)
            reader = next(iter(embedding._readers.values()))
            key = f'{prefix}.shard_0.weight'
            entry = reader._header[key]
            # Keep 2,048 distinct pages: the full model's shard has millions of
            # rows, rather than the small contiguous table used above.
            page_rows = 2049*_PLE_PAGE_SIZE//100
            page_header = json.dumps({'weight': dict(dtype='U32', shape=[page_rows, 25],
                data_offsets=[0, page_rows*100])}).encode()
            page_path = root/'pages.safetensors'
            with page_path.open('wb') as f:
                f.write(struct.pack('<Q', len(page_header)))
                f.write(page_header)
                f.truncate(8+len(page_header)+page_rows*100)
            page_reader = _SafeTensorMMap(page_path)
            row_indices = np.arange(2048, dtype=np.intp)*_PLE_PAGE_SIZE//100
            base = page_reader._data_start
            row_bytes = 100
            # Touch the same physical pages; only task dispatch varies.
            for count in [16, 48, 128, 2048]:
                selected_pages = row_indices[:count]
                def pages():
                    page_reader._seen_pages[:] = bytes(len(page_reader._seen_pages))
                    page_reader._prefetch_missing_pages(selected_pages, base, row_bytes)
                result[f'page_dispatch_{count}'] = measure(pages, 100 if count<200 else 50)
            page_reader.close()
            reader._seen_pages[:] = bytes([1])*len(reader._seen_pages)
            for count in [64, 2000, 8000]:
                selected = rng.integers(0, rows, size=count, dtype=np.intp)
                result[f'rows_copy_{count}'] = measure(lambda: reader.rows_np(key, selected), 500)
            result['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        finally:
            embedding.close()
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

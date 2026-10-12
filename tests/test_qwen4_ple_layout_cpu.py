# SPDX-License-Identifier: Apache-2.0
"""Check cached PLE layouts without loading MLX or a model."""

import contextlib
import json
import struct
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import test_qwen4_runtime_ple_fork_cpu as runtime


@contextlib.contextmanager
def table(layouts, sizes=None, dims=160):
    sizes = sizes or [4] * len(layouts)
    header, arrays, offset = {}, [], 0
    for shard, ((bits, group_size, dtypes), rows) in enumerate(zip(layouts, sizes)):
        widths = (
            [dims]
            if bits is None
            else [dims * bits // 32, dims // group_size, dims // group_size]
        )
        for family, width, dtype in zip(("weight", "scales", "biases"), widths, dtypes):
            array = np.arange(
                rows * width,
                dtype={
                    "U32": np.uint32,
                    "F32": np.float32,
                    "F16": np.float16,
                    "BF16": np.uint16,
                }[dtype],
            ).reshape(rows, width)
            key = f"table.shard_{shard}.{family}"
            header[key] = {
                "shape": list(array.shape),
                "dtype": dtype,
                "data_offsets": [offset, offset + array.nbytes],
            }
            offset += array.nbytes
            arrays.append(array)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        encoded = json.dumps(header).encode()
        (root / "table.safetensors").write_bytes(
            struct.pack("<Q", len(encoded))
            + encoded
            + b"".join(a.tobytes() for a in arrays)
        )
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {key: "table.safetensors" for key in header}})
        )
        mx = types.SimpleNamespace(
            ones=lambda shape, dtype: np.ones(shape, dtype=dtype), bfloat16=np.float16
        )
        with patch.dict(runtime.NS, {"mx": mx}):
            embedding = runtime.NS["DiskBackedShardedEmbedding"](
                root, "table", sum(sizes), dims, len(sizes)
            )
        try:
            yield embedding
        finally:
            embedding.close()


class PLELayoutTests(unittest.TestCase):
    def test_uniform_layout_does_not_query_readers_after_load(self):
        for layout in [(None, None, ("F32",)), (5, 32, ("U32", "BF16", "BF16"))]:
            with (
                self.subTest(layout=layout),
                table([layout] * 3, [5, 5, 4]) as embedding,
            ):
                for reader in embedding._readers.values():
                    reader.tensor_dtype = lambda _: self.fail("dtype query after load")
                host = np.array([13, 0, 5, 4, 10, 5, 9], dtype=np.int64)
                plan = embedding._plan(host)
                np.testing.assert_array_equal(plan[0], [2, 0, 1, 0, 2, 1, 1])
                np.testing.assert_array_equal(plan[1], [3, 0, 0, 4, 0, 0, 4])
                self.assertEqual(plan[2], [0, 1, 2])
                self.assertEqual(plan[6:], layout[:2])
                self.assertEqual(tuple(plan[5].values()), layout[2])
                buffers = embedding._assemble(host, plan)
                self.assertEqual(buffers[0].shape[0], host.size)
                np.testing.assert_array_equal(buffers[0][2], buffers[0][5])

    def test_mixed_layout_only_falls_back_when_touched_shards_differ(self):
        uniform = (4, 32, ("U32", "BF16", "BF16"))
        cases = [
            ((None, None, ("F32",)), (None, None, ("F16",))),
            (uniform, (5, 32, uniform[2])),
            (uniform, (4, 64, uniform[2])),
            (uniform, (4, 32, ("U32", "F16", "BF16"))),
            (uniform, (4, 32, ("U32", "BF16", "F16"))),
        ]
        for first, other in cases:
            with (
                self.subTest(other=other),
                table([first, first, other], dims=128) as embedding,
            ):
                self.assertIsNotNone(embedding._plan(np.array([0, 4, 0])))
                self.assertIsNotNone(embedding._plan(np.array([8, 9, 8])))
                self.assertIsNone(embedding._plan(np.array([0, 8])))


if __name__ == "__main__":
    unittest.main()

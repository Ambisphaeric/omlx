"""Alternate main and candidate on identical projection weights and activations."""
import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from omlx.patches import row_exact_qmv as main

args=argparse.ArgumentParser()
args.add_argument('--candidate',type=Path,required=True,help='Candidate row_exact_qmv.py')
args.add_argument('--output',type=Path,default=Path('micro.json'))
args=args.parse_args()
spec=importlib.util.spec_from_file_location('omlx.patches.tail_candidate', str(args.candidate))
candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)
results=[]
for k,n,bits,gs in [(2560,248320,8,64),(2560,32768,6,64),(2560,32768,5,64),(320,16387,8,64)]:
    mx.random.seed(k+n+bits)
    layer=nn.QuantizedLinear(k,n,bias=False,group_size=gs,bits=bits)
    layer.weight,layer.scales,layer.biases=mx.quantize((mx.random.normal((n,k))*.05).astype(mx.bfloat16),bits=bits,group_size=gs)
    mx.eval(layer.parameters());mx.clear_cache()
    for rows in [4,5,6,7,8,11,13]:
        x=mx.random.normal((1,rows,k)).astype(mx.bfloat16);mx.eval(x)
        expected=mx.concatenate([layer(x[:,r:r+1]) for r in range(rows)],axis=1);mx.eval(expected)
        samples={name:[] for name in ('main','candidate')}
        for block in range(4):
            variants=[('main',main),('candidate',candidate)]
            if block%2:variants.reverse()
            for name,module in variants:
                layer.__dict__.pop('_omlx_row_exact_plans',None)
                for _ in range(5):mx.eval(module.quantized_linear(layer,x))
                observed=module.quantized_linear(layer,x);mx.eval(observed)
                assert observed.shape==expected.shape and mx.array_equal(observed.view(mx.uint16),expected.view(mx.uint16)).item(),(name,k,n,bits,rows)
                for _ in range(20):
                    start=time.perf_counter_ns();mx.eval(module.quantized_linear(layer,x))
                    samples[name].append((time.perf_counter_ns()-start)/1e6)
        record=dict(k=k,n=n,bits=bits,group_size=gs,rows=rows,main_ms=statistics.median(samples['main']),candidate_ms=statistics.median(samples['candidate']),bit_exact=True)
        record['gain_pct']=100*(record['main_ms']/record['candidate_ms']-1)
        results.append(record);print(json.dumps(record),flush=True)
        args.output.write_text(json.dumps(results,indent=2))
        del x,expected,observed;mx.clear_cache()
    del layer;mx.clear_cache()

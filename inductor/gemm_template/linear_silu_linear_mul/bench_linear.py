try:
    import itt
except:
    # create a dummy itt module that contains pause and resume functions
    class itt:
        @staticmethod
        def detach():
            pass
        @staticmethod
        def pause():
            pass
        @staticmethod
        def resume():
            pass

itt.pause()

from unittest.mock import patch
import torch
from torch._inductor import config
import argparse
import torch._inductor.select_algorithm as select_algorithm
from torch.testing._internal import logging_utils
import time
import statistics
import contextlib
import gc

# Set up argument parsing
parser = argparse.ArgumentParser(description='Configure the model parameters.')
parser.add_argument('--batch-size', type=int, default=512, help='Batch size')
parser.add_argument('--in-features', type=int, default=768, help='Number of input features')
parser.add_argument('--out-features', type=int, default=3072, help='Number of output features')
parser.add_argument('--dtype', type=str, default='bfloat16', help='Data type (e.g., bfloat16, float16, float32)')
parser.add_argument('--freezing', type=bool, default=True, help='Enable or disable freezing')
parser.add_argument('--dynamic', type=bool, default=False, help='Enable or disable dynamic shape')
parser.add_argument('--verbose', action='store_true', help='Enable or disable verbose mode')
parser.add_argument('--count', type=int, default=50, help='Number of layers to run per each iteration')
parser.add_argument('--warmup', type=int, default=10, help='Number of warmup iterations')
parser.add_argument('--run', type=int, default=30, help='Number of run iterations')
parser.add_argument('--eager', action='store_true', help='Enable or disable eager mode')
parser.add_argument('--vtune', action='store_true', help='Enable or disable VTune profiling')
parser.add_argument('--aten', action='store_true', help='Use ATen GEMM backend')
parser.add_argument('--cpp', action='store_true', help='Use C++ GEMM backend')
parser.add_argument('--debug', action='store_true', help='Enable or disable debug mode')
parser.add_argument('--bias', action='store_true', help='Enable or disable bias')
parser.add_argument('--memray', type=str, default="", help='Memray profiling bin file')
parser.add_argument('--share-act', action='store_true', help='Share the same activation buffer')
parser.add_argument('--fusion', action='store_true', help='Share the same activation buffer')
parser.add_argument('--ipex', action='store_true', help='Share the same activation buffer')

args = parser.parse_args()

# Configure settings based on parsed arguments
config.freezing = args.freezing
if args.aten:
    config.max_autotune_gemm_backends = "ATEN"
if args.cpp:
    config.max_autotune_gemm_backends = "CPP"
if args.fusion:
    torch._inductor.config.enable_linear_silu_linear_mul = True

if args.debug:
    args.count = 1

batch_size = args.batch_size
in_features = args.in_features
out_features = args.out_features
dtype:torch.dtype = getattr(torch, args.dtype)
bias = args.bias

if args.ipex:
    import intel_extension_for_pytorch as ipex
    from intel_extension_for_pytorch.cpu._auto_kernel_selection import (
        _enable_tpp,
        _disable_tpp,
    )

if args.verbose:
    # Print the parameters
    print(f'batch_size: {batch_size}')
    print(f'in_features: {in_features}')
    print(f'out_features: {out_features}')
    print(f'dtype: {dtype}')
    print(f'bias: {bias}')
    print(f'freezing: {config.freezing}')
    print(f'ipex: {args.ipex}')
    print(f'aten: {args.aten}')
    print(f'silu mul fusion: {args.fusion}')

class M(torch.nn.Module):
    def __init__(self, bias):
        super().__init__()
        # self.linear = torch.nn.Linear(in_features, out_features, bias)
        self.gate_proj = torch.nn.Linear(in_features, out_features, bias=bias)
        self.up_proj = torch.nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        # return self.linear(x)
        return torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)

mods = []
for _ in range(args.count):
    if args.ipex:
        m = M(bias=bias).to(dtype=dtype).eval()
        mods.append(ipex.optimize(m, dtype=dtype))
    else:
        mods.append(M(bias=bias).to(dtype=dtype).eval())

def run_eager(x):
    y = []
    for i, mod in enumerate(mods):
        y.append(mod(x[i]))
    return y

@torch.compile(mode="max-autotune")
def run_compiled(x):
    return run_eager(x)

def run_ipex(x):
    y = []
    for i, mod in enumerate(mods):
        res = torch.ops.torch_ipex.tpp_fused_gate_up_proj(
            x[i],
            mod.gate_proj.weight,
            mod.gate_proj.bias,
            mod.up_proj.weight,
            mod.up_proj.bias,
        )
        y.append(res)
    return y

def benchmark(fn, x, iters):
    if args.ipex:
        _enable_tpp()
    gc.disable()
    durations = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(x)
        duration = (time.perf_counter() - t0) / args.count
        if args.verbose:
            print(f'duration: {duration * 1000:.4f} ms')
        durations.append(duration)
    gc.enable()
    if args.ipex:
        _disable_tpp()
    return durations

B = (batch_size,)
if args.share_act:
    v = [torch.randn(*B, in_features).to(dtype=dtype)] * args.count
else:
    v = []
    for _ in range(args.count):
        v.append(torch.randn(*B, in_features).to(dtype=dtype))
if args.dynamic:
    for x in v:
        torch._dynamo.mark_dynamic(x, 0)

logging_context = logging_utils.log_settings("+inductor") if args.debug else contextlib.nullcontext()
with torch.no_grad(), patch.object(select_algorithm, "VERIFY", dict(atol=1e-2, rtol=1e-2)), logging_context:
    if args.memray:
        import memray
        with memray.Tracker(args.memray, trace_python_allocators=True):
            result = torch.compile(mods[0], mode="max-autotune")(v[0])
    else:
        result = torch.compile(mods[0], mode="max-autotune")(v[0])
    ref = mods[0](v[0])
    assert torch.allclose(result, ref, atol=3e-2, rtol=3e-2)

    benchmark(run_ipex if args.ipex else run_compiled, v, args.warmup)
    if args.vtune:
        itt.resume()
        time.sleep(3)
    durations = benchmark(run_ipex if args.ipex else run_compiled, v, args.run)
    duration_ms = statistics.median(durations) * 1000
    M, N, K = args.batch_size, args.out_features, args.in_features
    GOPS = M*N*K*2/1000/1000/1000
    MEM = (M*N + N*K + K*M)*dtype.itemsize/1024/1024
    print(f"GEMM({M=},{N=},{K=}) compile: {duration_ms:.4f} ms ({GOPS/duration_ms:.2f} TOPS, {MEM/duration_ms:.2f} GB/s)")
    if args.eager:
        benchmark(run_eager, v, args.warmup)
        durations = benchmark(run_eager, v, args.run)
        duration_ms = statistics.median(durations) * 1000
        print(f"GEMM({M=},{N=},{K=}) compile: {duration_ms:.4f} ms ({GOPS/duration_ms:.2f} TOPS, {MEM/duration_ms:.2f} GB/s)")

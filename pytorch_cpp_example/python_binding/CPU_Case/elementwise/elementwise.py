import torch
import time 
from torch.utils.cpp_extension import load
from typing import Optional
from functools import partial

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(name='elementwise_lib', 
           sources=['elementwise.cpp'], 
           extra_cuda_cflags=[
               "-O3",
                # "-U__CUDA_NO_HALF_OPERATORS__",
                # "-U__CUDA_NO_HALF_CONVERSIONS__",
                # "-U__CUDA_NO_HALF2_OPERATORS__",
                # "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--use_fast_math",
            ], 
           extra_cflags=['-std=c++17'])


def run_benchmark(perf_func: callable, a: torch.Tensor, b: torch.Tensor, tag: str, 
                  out: Optional[torch.Tensor] = None, warmup: int = 10, 
                  iters: int = 1000, show_all: bool = False):
    # torch.dot vs custom dot_prod kernel
    if out is not None: 
        out.fill_(0)    
    # warmup
    if out is not None:
        for i in range(warmup):
            perf_func(a, b, out)
    else:
        for i in range(warmup):
            _ = perf_func(a, b) 
    # torch.cuda.synchronize()
    start = time.time()
    # iters
    if out is not None:
        for i in range(iters):
            perf_func(a, b, out)
    else:
        for i in range(iters):
            out = perf_func(a, b) 
    # torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000 # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = out.flatten().detach().cpu().numpy().tolist()[:2]
    out_val = [round(v, 8) for v in out_val]
    print(f"{out_info:>18}: {out_val}, time:{mean_time:.8f}ms")
    if show_all: print(out)
    return out, mean_time

device = torch.device("cpu")

print("-" * 80)
S, K = 4096, 4096
a = torch.randn((S, K)).to(device=device).float().contiguous()
b = torch.randn((S, K)).to(device=device).float().contiguous()
c = torch.zeros_like(a).to(device=device).float().contiguous()
run_benchmark(lib.elementwise_add_f32, a, b, "f32", c)
print("c is: {}".format(c.size()), flush=True)
print(torch.allclose(c, a + b, rtol=1e-3, atol=1e-3), flush=True)
# run_benchmark(lib.elementwise_add_f32x4, a, b, "f32x4", c)
# run_benchmark(partial(torch.add, out=c), a, b, "f32_th")

# print("-" * 80)
# a_f16 = a.half().contiguous()
# b_f16 = b.half().contiguous()
# c_f16 = c.half().contiguous()
# run_benchmark(lib.elementwise_add_f16,        a_f16, b_f16, "f16",       c_f16)
# run_benchmark(lib.elementwise_add_f16x2,      a_f16, b_f16, "f16x2",     c_f16)
# run_benchmark(lib.elementwise_add_f16x8,      a_f16, b_f16, "f16x8",     c_f16)
# run_benchmark(lib.elementwise_add_f16x8_pack, a_f16, b_f16, "f16x8pack", c_f16)
# run_benchmark(partial(torch.add, out=c_f16),  a_f16, b_f16, "f16_th")
# print("-" * 80)

# clear && rm -rf /tmp/torchinductor_leslie/* && numactl -C 56-111 -m 1 python micro_gemm_u8s8s32.py

# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align

from torch import device, empty_strided
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


# Inner Product
cpp_fused__softmax_0_inner_product = async_compile.cpp_pybinding(['const unsigned char*', 'const signed char*', 'int*'], '''
#include "/localdisk/leslie/torch_inductor_community/pytorch/torch/_inductor/codegen/cpp_prefix.h"
extern "C" void kernel(const unsigned char* __restrict__  in_ptr0,
                       const signed char* __restrict__  in_ptr1,
                       int* __restrict__  out_ptr1)
{
    for (int m=0; m<128; m++) {
        for (int n=0; n<512; n++) {
            out_ptr1[m*512 + n] = 0;
            for (int k=0; k<1024; k++) {
                out_ptr1[m*512 + n] += in_ptr0[m*1024 + k] * in_ptr1[k*512 + n];
            }
        }
    }
}
''')

# Outer Product
cpp_fused__softmax_0_outer_product = async_compile.cpp_pybinding(['const unsigned char*', 'const signed char*', 'int*'], '''
#include "/localdisk/leslie/torch_inductor_community/pytorch/torch/_inductor/codegen/cpp_prefix.h"
#include <iostream>
#include <c10/util/Unroll.h>
extern "C" void kernel(const unsigned char* __restrict__  in_ptr0,
                       const signed char* __restrict__  in_ptr1,
                       int* __restrict__  out_ptr1)
{
    constexpr auto BLOCK_M = 128;
    constexpr auto BLOCK_N = 512;
                                                 
                                                                 
    using Vectorized = at::vec::Vectorized<int>;
    constexpr auto VLEN = Vectorized::size();
    constexpr auto ROWS = BLOCK_M;
    constexpr auto COLS = BLOCK_N / VLEN;
    Vectorized va;
    at::vec::VectorizedN<int, COLS> vb;
    at::vec::VectorizedN<int, ROWS*COLS> vc;
                                                                 
    for (int k=0; k<1024; k++) {
        for (int m=0; m<128; m++) {
            for (int n=0; n<512; n++) {
                out_ptr1[m*512 + n] += in_ptr0[m*1024 + k] * in_ptr1[k*512 + n];
            }
        }
    }

                                                                 
}
''')

m, n, k = 128, 512, 1024

async_compile.wait(globals())
del async_compile

def test_correctness(args):
    arg0_1, arg1_1 = args
    args.clear()    
    ref_result = torch.matmul(arg0_1.to(torch.int32), arg1_1.to(torch.int32))
    buf2 = torch.zeros((m, n), dtype=torch.int32)
    cpp_fused__softmax_0_inner_product(arg0_1, arg1_1, buf2)
    print("Inner Product correctness is: {}".format(torch.allclose(ref_result, buf2)), flush=True)

    buf3 = torch.zeros((m, n), dtype=torch.int32)
    cpp_fused__softmax_0_outer_product(arg0_1, arg1_1, buf3)
    print("Outer Product correctness2 is: {}".format(torch.allclose(ref_result, buf3)), flush=True)

    return (buf2, )

def test_ref_performance(args):
    arg0_1, arg1_1 = args
    args.clear()    
    ref_result = torch.matmul(arg0_1.to(torch.int32), arg1_1.to(torch.int32))
    return (ref_result, )

def test_inner_product_performance(args):
    arg0_1, arg1_1 = args
    args.clear()    
    buf2 = torch.zeros((m, n), dtype=torch.int32)
    cpp_fused__softmax_0_inner_product(arg0_1, arg1_1, buf2)
    return (buf2, )

def test_outer_product_performance(args):
    arg0_1, arg1_1 = args
    args.clear()    
    buf2 = torch.zeros((m, n), dtype=torch.int32)
    cpp_fused__softmax_0_outer_product(arg0_1, arg1_1, buf2)
    return (buf2, )

def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((m, k), (k, 1), device='cpu', dtype=torch.float32).to(torch.uint8)
    arg0_2 = rand_strided((k, n), (n, 1), device='cpu', dtype=torch.float32).to(torch.int8)
    fn = lambda: test_correctness([arg0_1, arg0_2])
    fn()

    fn2 = lambda: test_ref_performance([arg0_1, arg0_2])
    print("---- ref time is: ", flush=True)
    print_performance(fn2, times=times, repeat=repeat)

    fn3 = lambda: test_inner_product_performance([arg0_1, arg0_2])
    print("---- inner product time is: ", flush=True)
    print_performance(fn3, times=times, repeat=repeat)

    fn4 = lambda: test_outer_product_performance([arg0_1, arg0_2])
    print("---- outer product time is: ", flush=True)
    print_performance(fn4, times=times, repeat=repeat)
    return 0


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)


# clear && rm -rf /tmp/torchinductor_leslie/* && numactl -C 56-111 -m 1 python micro_gemm_u8s8f32_vectorize_weight_per_channel.py

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

# Outer Product
cpp_fused__softmax_0_outer_product = async_compile.cpp_pybinding(['const bfloat16*', 'const bfloat16*', 'bfloat16*'], '''
#include "/localdisk/leslie/torch_inductor_community/pytorch/torch/_inductor/codegen/cpp_prefix.h"
#include <iostream>
#include <c10/util/Unroll.h>
                                                                                                                                                                            
extern "C" void kernel(const bfloat16* __restrict__  in_ptr0,
                       const bfloat16* __restrict__  in_ptr1,
                       bfloat16* __restrict__  out_ptr1)
{
    constexpr int BLOCK_M = 32;
    constexpr int BLOCK_N = 64;
    constexpr int BLOCK_K = 128;
    constexpr int lda = 128;
    constexpr int ldb = 64;
    constexpr int ldc = 64;

    constexpr bool accum = false;
    int alpha = 1;
                                                                                                            
    using VectorizeF32 = at::vec::Vectorized<float>;
    using VectorizeBF16 = at::vec::Vectorized<bfloat16>;
    using VectorizedS32 = at::vec::Vectorized<int>;
    using VectorizedU8 = at::vec::Vectorized<unsigned char>;
    using VectorizedS8 = at::vec::Vectorized<signed char>;

    // <TODO> Transform B to VNNI Layout from (K, N) to (K/2, N, 2)

    constexpr auto VLEN = VectorizeBF16::size();
    constexpr auto ROWS = BLOCK_M;
    constexpr auto COLS = BLOCK_N / VLEN;
    VectorizeBF16 va;
    at::vec::VectorizedN<bfloat16, COLS> vb;
    at::vec::VectorizedN<bfloat16, ROWS*COLS> vc;

    auto loadc = [&](auto i) {
        if constexpr (accum) {
            constexpr int row = i / COLS;
            constexpr int col = i % COLS;
            vc[i] = VectorizeBF16::loadu(out_ptr1 + row * ldc + col * VLEN);
        } else {
            vc[i] = VectorizeBF16(0.0);
        }
    };
    c10::ForcedUnroll<ROWS * COLS>{}(loadc);
                                       
    auto compute = [&, COLS](auto i, int k) {
        constexpr int row = i / COLS;
        constexpr int col = i % COLS;
        if constexpr (col == 0) {
            if (alpha != 1) {
               va = VectorizeBF16((in_ptr0[row * lda + k]) * alpha);
            } else {
               va = VectorizeBF16((in_ptr0[row * lda + k]));                                         
            }
        }
        if constexpr (row == 0) {
            vb[col] = VectorizeBF16::loadu(in_ptr1 + k * ldb + col * VLEN);
        }
        constexpr int idx = row * COLS + col;
        vc[idx] = at::vec::fmadd(va, vb[col], vc[idx]);
    };

    #pragma unroll(4)
    for (int k = 0; k < BLOCK_K; ++k) {
        c10::ForcedUnroll<ROWS * COLS>{}(compute, k);
    }
                                                                                      
    // Compensate and store to C
    auto storec = [&](auto i) {
        constexpr int row = i / COLS;
        constexpr int col = i % COLS;
        vc[i].store(out_ptr1 + row * ldc + col * VLEN);
    };
    c10::ForcedUnroll<ROWS * COLS>{}(storec);
                                                                                                                                                                                       
}
''')

m, n, k = 32, 64, 128
lda, ldb, ldc = 128, 64, 64

async_compile.wait(globals())
del async_compile

def test_correctness(args):
    arg0_1, arg1_1 = args
    args.clear()    
    ref_result = torch.matmul(arg0_1, arg1_1)

    buf2 = torch.zeros((m, n), dtype=torch.bfloat16)
    cpp_fused__softmax_0_outer_product(arg0_1, arg1_1, buf2)
    print("----- ref_result is: {}".format(ref_result), flush=True)
    print("----- buf2 is: {}".format(buf2), flush=True)
    print("Outer Product correctness2 is: {}".format(torch.allclose(ref_result, buf2, atol=1e-3, rtol=1e-3)), flush=True)

    return (buf2, )

def test_ref_performance(args):
    arg0_1, arg1_1 = args
    args.clear()    
    ref_result = torch.matmul(arg0_1, arg1_1)
    return (ref_result, )

def test_outer_product_performance(args):
    arg0_1, arg1_1, buf2 = args
    args.clear()    
    cpp_fused__softmax_0_outer_product(arg0_1, arg1_1, buf2)
    return (buf2, )

def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    a = rand_strided((m, k), (k, 1), device='cpu', dtype=torch.bfloat16)
    b = rand_strided((k, n), (n, 1), device='cpu', dtype=torch.bfloat16)


    fn = lambda: test_correctness([a, b])
    fn()

    # fn2 = lambda: test_ref_performance([a, b])
    # print("---- ref time is: ", flush=True)
    # print_performance(fn2, times=times, repeat=repeat)

    # buf2 = torch.zeros((m, n), dtype=torch.float32)
    # fn4 = lambda: test_outer_product_performance([a, b, buf2])
    # print("---- outer product time is: ", flush=True)
    # print_performance(fn4, times=times, repeat=repeat)
    return 0


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)

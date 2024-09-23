
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
import numpy as np

local_seed = 2024
torch.manual_seed(local_seed) # Set PyTorch seed
np.random.seed(seed=local_seed) # Set Numpy seed
random.seed(local_seed) # Set the Python seed

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
cpp_fused__softmax_0_outer_product = async_compile.cpp_pybinding(['const bfloat16*', 'const bfloat16*', 'float*'], '''
#include "/localdisk/leslie/torch_inductor_community/pytorch/torch/_inductor/codegen/cpp_prefix.h"
#include <iostream>
#include <c10/util/Unroll.h>

typedef struct tileconfig_t {
  uint8_t palette_id;
  uint8_t startRow;
  uint8_t reserved[14];
  uint16_t colb[16]; // 一般就用 前8个, 每个值 分别对应 一个 Tile 的 一行有多少 bytes 数据
  uint8_t rows[16];  // 一般就用 前8个, 每个值 分别对应 一个 Tile 的 有多少行数据
} tileconfig_t;
                                                                                                                           
static tileconfig_t tc = {0};

__m512bh cast(at::vec::Vectorized<float>& src){
    // Convert Vectorized<float> to __m512bh                                                            
    return (__m512bh)_mm512_castps_pd(src);                                        
}
                                                                                                                                                                                                                                                                                                                                                                                                                               
extern "C" void kernel(const bfloat16* __restrict__  in_ptr0,
                       const bfloat16* __restrict__  in_ptr1,
                       float* __restrict__  out_ptr1)
{
    constexpr int BLOCK_M = 16;
    constexpr int BLOCK_N = 16;
    constexpr int BLOCK_K = 32;
    constexpr int lda = 32;
    constexpr int ldb = 16;
    constexpr int ldc = 16;

    constexpr bool accum = false;
    int alpha = 1;
    

    // <TODO> Transform B to VNNI Layout from (K, N) to (K/2, N, 2)
    bfloat16 B_VNNI_Layout[BLOCK_K*BLOCK_N];
    int idx = 0;
    for (int i = 0; i < BLOCK_K; i+=2) {
        for (int j = 0; j < BLOCK_N; j++) {
            B_VNNI_Layout[idx] = in_ptr1[i*ldb + j];
            B_VNNI_Layout[idx + 1] = in_ptr1[(i+1)*ldb + j];
            idx += 2;
        }
    }

    const uint8_t TILE_M = 16;
    const uint8_t TILE_N = 16;
    const uint8_t TILE_IK = 64; // For INT8
    const uint8_t TILE_BK = 32; // For BF16
    const uint8_t KPACK = 2; // 2 for bf16; 4 for int8; with VNNI Layout
                                                                 
    // Config tiles
    tc.palette_id = 1;
    tc.startRow = 0;
    
    // Tile[0] 用作 C
    tc.rows[0] = (uint8_t)TILE_M;
    tc.colb[0] = (uint16_t)(TILE_N * sizeof(float)); // 配置 多少 bytes, at most 64 bytes
    
    // Tile[1] 用作 A
    tc.rows[1] = (uint8_t)TILE_M;
    tc.colb[1] = (uint16_t)(TILE_BK * sizeof(bfloat16)); // 配置 多少 bytes
    
    // Tile[2] 用作 B
    tc.rows[2] = (uint8_t)(TILE_BK / KPACK);
    tc.colb[2] = (uint16_t)(TILE_N * KPACK * sizeof(bfloat16));
    // 剩下5个Tile 先不用

    // Config the Tiles
    _tile_loadconfig((const void*)&tc);

    // Load Data
    // _tile_loadd(0, out_ptr1, ldc);
    _tile_zero(0);
    _tile_loadd(1, in_ptr0, TILE_BK * sizeof(bfloat16));
    _tile_loadd(2, B_VNNI_Layout, TILE_N * KPACK * sizeof(bfloat16));

    _tile_dpbf16ps(0, 1, 2);

    // float temp_out[16*16];
    _tile_stored(0, out_ptr1, ldc * sizeof(float));

    // std::cout<<"----- temp_out is: "<<temp_out[0]<<std::endl;
                                                                 
    // Release the Tile Data
    _tile_release();

    /*
    constexpr auto VLEN = VectorizeF32::size();                                                  
    constexpr auto ROWS = BLOCK_M;
    constexpr auto COLS = BLOCK_N / VLEN;
    VectorizeF32 va;
    at::vec::VectorizedN<float, COLS> vb;
    at::vec::VectorizedN<float, ROWS*COLS> vc;

    auto loadc = [&](auto i) {
        if constexpr (accum) {
            constexpr int row = i / COLS;
            constexpr int col = i % COLS;
            vc[i] = VectorizeF32::loadu(out_ptr1 + row * ldc + col * VLEN);
        } else {
            vc[i] = VectorizeF32(0.0);
        }
    };
    c10::ForcedUnroll<ROWS * COLS>{}(loadc);
                           
    auto compute = [&, COLS](auto i, int k) {
        constexpr int row = i / COLS;
        constexpr int col = i % COLS;
        if constexpr (col == 0) {
            if (alpha != 1) {
               va = VectorizeF32((((float*)in_ptr0)[row * (lda/2) + k]) * alpha);
            } else {
               va = VectorizeF32((((float*)in_ptr0)[row * (lda/2) + k]));                                         
            }
        }
        if constexpr (row == 0) {
            vb[col] = VectorizeF32::loadu(((float*)B_VNNI_Layout) + k * ldb + col * VLEN);
        }
        constexpr int idx = row * COLS + col;
        vc[idx] = _mm512_dpbf16_ps(vc[idx], cast(va), cast(vb[col]));
    };

    #pragma unroll(4)
    for (int k = 0; k < (BLOCK_K/2); k++) {
        c10::ForcedUnroll<ROWS * COLS>{}(compute, k);
    }
                                                                     
    // Compensate and store to C
    auto storec = [&](auto i) {
        constexpr int row = i / COLS;
        constexpr int col = i % COLS;
        VectorizeBF16 temp = _mm512_castsi256_si512(at::vec::cvtfp32_bf16(vc[i]));
        temp.store(out_ptr1 + row * ldc + col * VLEN, 16);
    };
    c10::ForcedUnroll<ROWS * COLS>{}(storec);
    */
                                                                                                                                                                              
}
''')

m, n, k = 16, 16, 32
lda, ldb, ldc = 32, 16, 16

async_compile.wait(globals())
del async_compile

def test_correctness(args):
    arg0_1, arg1_1 = args
    args.clear()    
    ref_result = torch.matmul(arg0_1, arg1_1).to(torch.float32)

    buf2 = torch.zeros((m, n), dtype=torch.float32)
    cpp_fused__softmax_0_outer_product(arg0_1, arg1_1, buf2)
    buf2 = buf2.to(torch.bfloat16).to(torch.float32)
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

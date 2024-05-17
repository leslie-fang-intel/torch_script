
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
    constexpr int BLOCK_M = 32;
    constexpr int BLOCK_N = 32;
    constexpr int BLOCK_K = 64;
    constexpr int lda = 64;
    constexpr int ldb = 32;
    constexpr int ldc = 32;

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

    // Tile[0-3] 用作 C
    for (int i=0;i<4;i++) {
        tc.rows[i] = (uint8_t)TILE_M;
        tc.colb[i] = (uint16_t)(TILE_N * sizeof(float)); // 配置 多少 bytes, at most 64 bytes                                                                 
    }
    
    // Tile[4] 用作 A
    tc.rows[4] = (uint8_t)TILE_M;
    tc.colb[4] = (uint16_t)(TILE_BK * sizeof(bfloat16)); // 配置 多少 bytes
    
    // Tile[5~6] 用作 B
    for (int i=5;i<7;i++) {
        tc.rows[i] = (uint8_t)(TILE_BK / KPACK);
        tc.colb[i] = (uint16_t)(TILE_N * KPACK * sizeof(bfloat16));
    }
    // 剩下1个Tile 先不用


    // Config the Tiles
    _tile_loadconfig((const void*)&tc);


    constexpr int ROWS = BLOCK_M / TILE_M; // 2
    constexpr int COLS = BLOCK_N / TILE_N; // 2


    auto loadc = [&](int i) {
        TORCH_CHECK(accum == false, "accum is not supported");
        if (i == 0){
            _tile_zero(0);                                                    
        } else if (i == 1) {
            _tile_zero(1);                                            
        } else if (i == 2) {
            _tile_zero(2);                                            
        } else if (i == 3) {
            _tile_zero(3);                                            
        } else {
            TORCH_CHECK(false, "C Tile exceed");                                                   
        }
    };
    c10::ForcedUnroll<ROWS * COLS>{}(loadc);

                                                                     
    auto compute = [&, COLS](auto i, int k) {
        constexpr int row = i / COLS;
        constexpr int col = i % COLS;
        if constexpr (col == 0) {
            TORCH_CHECK(alpha == 1);
            _tile_loadd(4, in_ptr0 + row * TILE_M * BLOCK_K + k * TILE_BK, BLOCK_K * sizeof(bfloat16));
        }
        if constexpr (row == 0) {
            if (col == 0) {
                _tile_loadd(5, B_VNNI_Layout + k * (TILE_BK / KPACK) * (BLOCK_N * KPACK) + col * TILE_N * KPACK, BLOCK_N * KPACK * sizeof(bfloat16));                                                                 
            } else {
                TORCH_CHECK(col == 1);         
                _tile_loadd(6, B_VNNI_Layout + k * (TILE_BK / KPACK) * (BLOCK_N * KPACK) + col * TILE_N * KPACK, BLOCK_N * KPACK * sizeof(bfloat16));                                      
            }
        }

        if (i == 0){
            _tile_dpbf16ps(0, 4, 5);                                                   
        } else if (i == 1) {
            _tile_dpbf16ps(1, 4, 6);                                         
        } else if (i == 2) {
            _tile_dpbf16ps(2, 4, 5);                                         
        } else if (i == 3) {
            _tile_dpbf16ps(3, 4, 6);                                         
        } else {
            TORCH_CHECK(false, "C Tile exceed");                                                   
        }
    };

    for (int k = 0; k < (BLOCK_K/TILE_BK); k++) {
        c10::ForcedUnroll<ROWS * COLS>{}(compute, k);
    }

    // Compensate and store to C
    auto storec = [&](auto i) {
        constexpr int row = i / COLS;
        constexpr int col = i % COLS;
        if (i == 0){
            _tile_stored(0, out_ptr1, ldc * sizeof(float));                                                 
        } else if (i == 1) {
            _tile_stored(1, out_ptr1 + col * TILE_N, ldc * sizeof(float));                                      
        } else if (i == 2) {
            _tile_stored(2, out_ptr1 + row * TILE_M * ldc, ldc * sizeof(float));                                      
        } else if (i == 3) {
            _tile_stored(3, out_ptr1 + row * TILE_M * ldc + col * TILE_N, ldc * sizeof(float));                                        
        } else {
            TORCH_CHECK(false, "C Tile exceed");                                                   
        }
    };
    c10::ForcedUnroll<ROWS * COLS>{}(storec);
                                                      
    // Release the Tile Data
    _tile_release();                                                                                                                                                                  
}
''')

m, n, k = 32, 32, 64
lda, ldb, ldc = 64, 32, 32

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


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


# Inner Product
cpp_fused__softmax_0_inner_product = async_compile.cpp_pybinding(['const unsigned char*', 'const signed char*', 'float*', 'const float*', 'const int*'], '''
#include "/localdisk/leslie/torch_inductor_community/pytorch/torch/_inductor/codegen/cpp_prefix.h"
extern "C" void kernel(const unsigned char* __restrict__  in_ptr0,
                       const signed char* __restrict__  in_ptr1,
                       float* __restrict__  out_ptr1,
                       const float* __restrict__ b_scale,
                       const int* __restrict__ b_zp)
{
    float a_scale = 0.02;
    int a_zp = 10;
    // float b_scale = 0.04;
    // int b_zp = 5;
    
    int M = 32;
    int N = 64;
    int K = 128;

    // Calculate the u8s8s32 with inner product
    for (int m=0; m<M; m++) {
        for (int n=0; n<N; n++) {
            out_ptr1[m*N + n] = 0.0;
            for (int k=0; k<K; k++) {
                out_ptr1[m*N + n] += in_ptr0[m*K + k] * in_ptr1[k*N + n];
            }
        }
    }
    
    // Compute the compensation of in_ptr0
    float a_compensate[M];
    for (int m=0; m<M; m++) {
        a_compensate[m] = 0.0;
        for (int k=0; k<K; k++) {
            a_compensate[m] += in_ptr0[m*K + k];
        }
    }

    // Compute the compensation of in_ptr1
    float b_compensate[N];
    for (int n=0; n<N; n++) {
        b_compensate[n] = 0.0;
        for (int k=0; k<K; k++) {
            b_compensate[n] += in_ptr1[k*N + n];
        }
    }            

    // Compensate the s32 output to f32
    for (int m=0; m<32; m++) {
        for (int n=0; n<64; n++) {
            out_ptr1[m*64 + n] = a_scale * b_scale[n] * out_ptr1[m*64 + n];
            out_ptr1[m*64 + n] -= a_scale * b_scale[n] * a_zp * b_compensate[n];
            out_ptr1[m*64 + n] -= a_scale * b_scale[n] * b_zp[n] * a_compensate[m];
            out_ptr1[m*64 + n] += a_scale * b_scale[n] * a_zp * b_zp[n] * K;
        }
    }
}
''')

# Outer Product
cpp_fused__softmax_0_outer_product = async_compile.cpp_pybinding(['const unsigned char*', 'const signed char*', 'float*', 'const float*', 'const int*'], '''
#include "/localdisk/leslie/torch_inductor_community/pytorch/torch/_inductor/codegen/cpp_prefix.h"
#include <iostream>
#include <c10/util/Unroll.h>

template <typename scalar_t>
inline void _sum_a_contiguous_kernel(
    const scalar_t* in,
    float* out,
    const int& M,
    const int& K,
    const int& ld) {
    using VectorizedA = at::vec::Vectorized<unsigned char>;
    using Vectorize_f32 = at::vec::Vectorized<float>;
    auto VLEN = Vectorize_f32::size();
    for (int m=0; m<M; m++) {
        float tmp_sum = 0.0;
        Vectorize_f32 vec_tmp_sum(tmp_sum);
        int k = 0;
        for (; k < c10::div_floor_integer(K, VLEN) * VLEN; k+=VLEN) {
            VectorizedA tmpA = VectorizedA::loadu(in + m*K + k, VLEN);
            vec_tmp_sum += convert_int8_to_float(tmpA);
        }
        tmp_sum += at::vec::vec_reduce_all<float>(
            [](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) {
                return x + y;
            },
            vec_tmp_sum);
        for (; k<K; k+=1) {
            tmp_sum += in[m*K + k];     
        }
        out[m] = tmp_sum;
    }
}
                                                                 
template <typename scalar_t>
inline void _sum_b_contiguous_kernel(
    const scalar_t* in,
    float* out,
    const int& K,
    const int& N,
    const int& ld) {
    using VectorizedB = at::vec::Vectorized<signed char>;
    using Vectorize_f32 = at::vec::Vectorized<float>;
    auto VLEN = Vectorize_f32::size();
    for (int k=0; k<K; k++) {
        int n=0;
        for (; n < c10::div_floor_integer(N, VLEN) * VLEN; n+=VLEN) {
            Vectorize_f32 temp_b_compensate;
            if (k == 0){
                temp_b_compensate = Vectorize_f32(0.0);
            } else {
                temp_b_compensate = Vectorize_f32::loadu(out + n, VLEN);                                                 
            }
            VectorizedB vb = VectorizedB::loadu(in + k * ld + n, VLEN);
            Vectorize_f32 vb_f32 = convert_int8_to_float(vb);                                                                 
            temp_b_compensate += vb_f32;                                                                 
            temp_b_compensate.store(out + n);
        }
        for (; n<N; n+=1) {
            if (k == 0) {
                out[n] = 0.0;
            }
            out[n] += in[k*ld + n];
        }
    }
}
                                                                                                                                                                            
extern "C" void kernel(const unsigned char* __restrict__  in_ptr0,
                       const signed char* __restrict__  in_ptr1,
                       float* __restrict__  out_ptr1,
                       const float* __restrict__ b_scale,
                       const int* __restrict__ b_zp                                        )
{
    constexpr auto BLOCK_M = 32;
    constexpr auto BLOCK_N = 64;
    const int K = 128;
    constexpr bool accum = false;
    int alpha = 1;
    int lda = K;
    int ldb = BLOCK_N;
    int ldc = BLOCK_N;

    float a_scale = 0.02;
    int a_zp = 10;
    // float b_scale = 0.04;
    // int b_zp = 5;
                                                 
                                                                 
    using Vectorized = at::vec::Vectorized<int>;
    using VectorizedA = at::vec::Vectorized<unsigned char>;
    using VectorizedB = at::vec::Vectorized<signed char>;
    using Vectorize_f32 = at::vec::Vectorized<float>;

    constexpr auto VLEN = Vectorized::size();
    constexpr auto ROWS = BLOCK_M;
    constexpr auto COLS = BLOCK_N / VLEN;
    Vectorized va;
    at::vec::VectorizedN<int, COLS> vb;
    at::vec::VectorizedN<int, ROWS*COLS> vc;


    auto loadc = [&](auto i) {
        if constexpr (accum) {
            constexpr int row = i / COLS;
            constexpr int col = i % COLS;
            vc[i] = Vectorized::loadu(out_ptr1 + row * ldc + col * VLEN);
        } else {
            vc[i] = Vectorized(0);
        }
    };
    c10::ForcedUnroll<ROWS * COLS>{}(loadc);
                                                                 
    auto compute = [&, COLS](auto i, int k) {
        constexpr int row = i / COLS;
        constexpr int col = i % COLS;
        if constexpr (col == 0) {
            if (alpha != 1) {
                // convert scalar u8 to int32
               va = Vectorized(int(in_ptr0[row * lda + k]) * alpha);
            } else {
               va = Vectorized(int(in_ptr0[row * lda + k]));                                                  
            }
        }
        if constexpr (row == 0) {
            // convert vectorized s8 to int32
            vb[col] = at::vec::convert_to_int32(in_ptr1 + k * ldb + col * VLEN);
        }
        constexpr int idx = row * COLS + col;
        // Do s32s32s32 fmadd
        vc[idx] = at::vec::fmadd(va, vb[col], vc[idx]);
    };

    #pragma unroll(4)
    for (int k = 0; k < K; ++k) {
        c10::ForcedUnroll<ROWS * COLS>{}(compute, k);
    }


    // Compute the compensation of in_ptr0
    // Scalar Version
    //float a_compensate[BLOCK_M];
    //for (int m=0; m<BLOCK_M; m++) {
    //    a_compensate[m] = 0.0;
    //    for (int k=0; k<K; k++) {
    //        a_compensate[m] += in_ptr0[m*K + k];
    //    }
    //}

    // Vectorize Version
    float a_compensate[BLOCK_M];
    _sum_a_contiguous_kernel(
        in_ptr0,
        a_compensate,
        BLOCK_M,
        K,
        lda);
                                                           
    // Compute the compensation of in_ptr1
    // Scalar Version
    //float b_compensate[BLOCK_N];
    //for (int n=0; n<BLOCK_N; n++) {
    //    b_compensate[n] = 0.0;
    //    for (int k=0; k<K; k++) {
    //        b_compensate[n] += in_ptr1[k*BLOCK_N + n];
    //    }
    //}

    // Vectorized Version
    float b_compensate[BLOCK_N];                               
    _sum_b_contiguous_kernel<signed char>(
        in_ptr1,
        b_compensate,
        K,
        BLOCK_N,
        ldb);
                                                                                      
    // Compensate and store to C
    auto storec = [&](auto i) {
        constexpr int row = i / COLS;
        constexpr int col = i % COLS;
        
        Vectorize_f32 temp = _mm512_cvtepi32_ps(vc[i]);

        Vectorize_f32 b_scales_v = Vectorize_f32::loadu(b_scale + col * VLEN, VLEN);
        Vectorize_f32 b_zp_v = _mm512_cvtepi32_ps(Vectorized::loadu(b_zp + col * VLEN, VLEN));

        temp *= Vectorize_f32(a_scale) * b_scales_v;    
        temp -= Vectorize_f32(a_scale * a_zp) * b_scales_v * Vectorize_f32::loadu(b_compensate + col * VLEN, VLEN);   
        // We can skip below 2 if b_zp is 0
        temp -= Vectorize_f32(a_scale * a_compensate[row]) * b_scales_v * b_zp_v;  
        temp += Vectorize_f32(a_scale * a_zp * K) * b_scales_v * b_zp_v;   

        temp.store(out_ptr1 + row * ldc + col * VLEN);
    };
    c10::ForcedUnroll<ROWS * COLS>{}(storec);
                                                                                                                                                                                       
}
''')

m, n, k = 32, 64, 128

async_compile.wait(globals())
del async_compile

def test_correctness(args):
    arg0_1, arg1_1, a_scale, a_zp, b_scale, b_zp = args
    args.clear()    
    ref_result = torch.matmul(arg0_1.to(torch.float32), arg1_1.to(torch.float32))


    qarg0_1 = torch.clamp(torch.round(arg0_1 / a_scale) + a_zp, 0, 255).to(torch.uint8)
    # qarg1_1 = torch.clamp(torch.round(arg1_1 / b_scale) + b_zp, -128, 127).to(torch.int8)
    qarg1_1 = torch.ops.quantized_decomposed.quantize_per_channel(
        arg1_1,
        b_scale,
        b_zp,
        1,
        -128,
        127,
        torch.int8,
    )
    
    buf2 = torch.zeros((m, n), dtype=torch.float32)
    cpp_fused__softmax_0_inner_product(qarg0_1, qarg1_1, buf2, b_scale, b_zp)

    # print("----- ref_result is: {}".format(ref_result), flush=True)
    # print("----- buf2 is: {}".format(buf2), flush=True)
    print("Inner Product correctness is: {}".format(torch.allclose(ref_result, buf2, atol=1e-3, rtol=1e-3)), flush=True)

    buf3 = torch.zeros((m, n), dtype=torch.float32)
    cpp_fused__softmax_0_outer_product(qarg0_1, qarg1_1, buf3, b_scale, b_zp)
    print("----- ref_result is: {}".format(ref_result), flush=True)
    print("----- buf3 is: {}".format(buf3), flush=True)
    print("Outer Product correctness2 is: {}".format(torch.allclose(ref_result, buf3, atol=1e-3, rtol=1e-3)), flush=True)

    return (buf2, )

def test_ref_performance(args):
    arg0_1, arg1_1, a_scale, a_zp, b_scale, b_zp = args
    args.clear()    
    ref_result = torch.matmul(arg0_1.to(torch.float32), arg1_1.to(torch.float32))
    return (ref_result, )

def test_inner_product_performance(args):
    qarg0_1, qarg1_1, buf2 = args
    args.clear()    
    cpp_fused__softmax_0_inner_product(qarg0_1, qarg1_1, buf2)
    return (buf2, )

def test_outer_product_performance(args):
    qarg0_1, qarg1_1, buf2 = args
    args.clear()    
    cpp_fused__softmax_0_outer_product(qarg0_1, qarg1_1, buf2)
    return (buf2, )

def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    a = rand_strided((m, k), (k, 1), device='cpu', dtype=torch.float32)
    b = rand_strided((k, n), (n, 1), device='cpu', dtype=torch.float32)

    a_scale = 0.02
    a_zp = 10
    b_scales = []
    for _ in range(n):
        b_scales.append(0.04)
    b_scale = torch.tensor(b_scales, dtype=torch.float32)
    b_zp = torch.zeros((n, ), dtype=torch.int)
    arg0_1 = torch.quantize_per_tensor(a, a_scale, a_zp, torch.quint8).dequantize()
    arg0_2 = torch.quantize_per_channel(b, b_scale, b_zp, 1, torch.qint8).dequantize()


    fn = lambda: test_correctness([arg0_1, arg0_2, a_scale, a_zp, b_scale, b_zp])
    fn()

    # fn2 = lambda: test_ref_performance([arg0_1, arg0_2, a_scale, a_zp, b_scale, b_zp])
    # print("---- ref time is: ", flush=True)
    # print_performance(fn2, times=times, repeat=repeat)

    # qarg0_1 = torch.clamp(torch.round(arg0_1 / a_scale) + a_zp, 0, 255).to(torch.uint8)
    # qarg1_1 = torch.clamp(torch.round(arg0_2 / b_scale) + b_zp, -128, 127).to(torch.int8)
    # buf2 = torch.zeros((m, n), dtype=torch.float32)

    # fn3 = lambda: test_inner_product_performance([qarg0_1, qarg1_1, buf2])
    # print("---- inner product time is: ", flush=True)
    # print_performance(fn3, times=times, repeat=repeat)

    # buf2 = torch.zeros((m, n), dtype=torch.float32)
    # fn4 = lambda: test_outer_product_performance([qarg0_1, qarg1_1, buf2])
    # print("---- outer product time is: ", flush=True)
    # print_performance(fn4, times=times, repeat=repeat)
    return 0


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)

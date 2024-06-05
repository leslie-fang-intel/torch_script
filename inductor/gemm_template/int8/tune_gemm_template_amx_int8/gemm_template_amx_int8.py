
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

import random
import numpy as np

random.seed(2023)
torch.manual_seed(2023)
np.random.seed(2023)

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()
_frozen_param0 = None  # device(type='cpu') torch.float32 (64,) (1,) 7fb7608fa840
_frozen_param1 = None  # device(type='cpu') torch.int64 (64,) (1,) 7fb7608fa980
_frozen_param3 = None  # device(type='cpu') torch.int8 (128, 64) (1, 0) 7fb73cf9b1a0
x_scale = None  # device(type='cpu') torch.float32 () () 7fb7609befc0
x_zp = None  # device(type='cpu') torch.int32 () () 7fb72ff37d80
_frozen_param1_0 = None  # device(type='cpu') torch.int32 (64,) (1,) 7fb72ff98860
_frozen_param3BMatricCompo = None  # device(type='cpu') torch.float32 (64,) (1,) 7fb72b4f13a0
constant7 = None  # device(type='cpu') torch.int8 (2, 128, 32) (4096, 32, 1) 7fb6e775bd80

# Inner Product
cpp_fused__softmax_0_inner_product = async_compile.cpp_pybinding(['const unsigned char*', 'const signed char*', 'float*', 'const float*', 'const int*', "const float*", "const int*"], '''
#include "/localdisk/leslie/torch_inductor_community/pytorch/torch/_inductor/codegen/cpp_prefix.h"
#include <iostream>
extern "C" void kernel(const unsigned char* __restrict__  in_ptr0,
                       const signed char* __restrict__  in_ptr1,
                       float* __restrict__  out_ptr1,
                       const float* __restrict__ b_scale,
                       const int* __restrict__ b_zp,
                       const float* _a_scale,
                       const int* _a_zp)
{
    float a_scale = _a_scale[0];
    int a_zp = _a_zp[0];

    // std::cout<<" a_scale is: "<<a_scale<<std::endl;
    // std::cout<<" a_zp is: "<<a_zp<<std::endl;
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
            std::cout<<out_ptr1[m*N + n]<<" ";
        }
        std::cout<<std::endl<<"-------"<<std::endl;   
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
    for (int m=0; m<M; m++) {
        for (int n=0; n<N; n++) {
            out_ptr1[m*N + n] = a_scale * b_scale[n] * out_ptr1[m*N + n];
            out_ptr1[m*N + n] -= a_scale * b_scale[n] * a_zp * b_compensate[n];
            out_ptr1[m*N + n] -= a_scale * b_scale[n] * b_zp[n] * a_compensate[m];
            out_ptr1[m*N + n] += a_scale * b_scale[n] * a_zp * b_zp[n] * K;
        }
    }
}
''')

cpp_fused_quantize_per_tensor_0 = async_compile.cpp_pybinding(['const float*', 'uint8_t*'], '''
#include "/localdisk/leslie/torch_inductor_community/pytorch/torch/_inductor/codegen/cpp_prefix.h"
extern "C" void kernel(const float* in_ptr0,
                       uint8_t* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(16L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0), 16);
            auto tmp1 = static_cast<float>(35.93893303716638);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp4 = tmp3.round();
            auto tmp5 = static_cast<float>(135.0);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 + tmp6;
            auto tmp8 = static_cast<float>(0.0);
            auto tmp9 = at::vec::Vectorized<float>(tmp8);
            auto tmp10 = at::vec::maximum(tmp7, tmp9);
            auto tmp11 = static_cast<float>(255.0);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = at::vec::minimum(tmp10, tmp12);
            auto tmp14 = at::vec::convert<uint8_t>(tmp13);
            tmp14.store(out_ptr0 + static_cast<long>(x0), 16);
        }
    }
}
''')


cpp_fused_mm_quantize_per_tensor_1 = async_compile.cpp_pybinding(['const uint8_t*', 'const float*', 'const int32_t*', 'const int8_t*', 'const float*', 'const int32_t*', 'const float*', 'float*'], '''
#include "/localdisk/leslie/torch_inductor_community/pytorch/torch/_inductor/codegen/cpp_prefix.h"
#include <iostream>
#include "c10/util/Unroll.h"



template <bool accum>
inline void kernel_micro_gemm_amx_kernel_32_1(
    AMXState& amx_state,
    const uint8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    int32_t* __restrict__ C,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint8_t tilecfg_rows
) {
    // TODO(jgong5): add prefetch hint for A, B, C
    auto loadconfig = [](const amx_tilecfg& cfg) {
        _tile_loadconfig(&cfg);
    };
    const auto last_k_offset = K / 16 * 16;
    const auto tail_k_size = K - last_k_offset;
    if C10_LIKELY (last_k_offset > 0) {
        amx_state.configure(tilecfg_rows, 16, 32 / 16, 1, loadconfig);
    } else {
        amx_state.configure(tilecfg_rows, tail_k_size * sizeof(uint8_t), 32 / 16, 1, loadconfig);
    }
    auto load_c = [&]() {
        _tile_loadd(0, C + 0 * ldc + 0, ldc * sizeof(float));
        _tile_loadd(1, C + 16 * ldc + 0, ldc * sizeof(float));
    };
    auto zero_c = [&]() {
        _tile_zero(0);
        _tile_zero(1);
    };

    if constexpr (accum) {
        load_c();
    } else {
        zero_c();
    }

    auto compute = [&](int k) {
        _tile_loadd(2, A + 0 * lda + k, lda * sizeof(uint8_t));
        _tile_loadd(4, B + k * ldb + 0, ldb * 4 * sizeof(uint8_t));
        // _tile_dpbf16ps(0, 2, 4);
        _tile_dpbusd(0, 2, 4);
        _tile_loadd(3, A + 16 * lda + k, lda * sizeof(uint8_t));
        // _tile_dpbf16ps(1, 3, 4);
        _tile_dpbusd(1, 3, 4);
    };

    #pragma GCC unroll 4
    for (int k = 0; k < last_k_offset; k += 16) {
        compute(k);
    }

    auto store_c = [&]() {
    // store to C
        _tile_stored(0, C + 0 * ldc + 0, ldc * sizeof(float));
        _tile_stored(1, C + 16 * ldc + 0, ldc * sizeof(float));
    };

    // TODO(jgong5): move tail k computation to separate loopnest to save tile configuration overhead
    if C10_UNLIKELY (tail_k_size > 0) {
        if C10_LIKELY (last_k_offset > 0) {
            store_c();
            amx_state.configure(tilecfg_rows, tail_k_size * sizeof(uint8_t), 32 / 16, 1, loadconfig);
            load_c();
        }
        compute(last_k_offset);
    }

    store_c();
}
template <bool accum>
inline void kernel_micro_gemm_amx_kernel_16_1(
    AMXState& amx_state,
    const uint8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    int32_t* __restrict__ C,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint8_t tilecfg_rows
) {
    // TODO(jgong5): add prefetch hint for A, B, C
    auto loadconfig = [](const amx_tilecfg& cfg) {
        _tile_loadconfig(&cfg);
    };
    const auto last_k_offset = K / 16 * 16;
    const auto tail_k_size = K - last_k_offset;
    if C10_LIKELY (last_k_offset > 0) {
        amx_state.configure(tilecfg_rows, 16, 16 / 16, 1, loadconfig);
    } else {
        amx_state.configure(tilecfg_rows, tail_k_size * sizeof(uint8_t), 16 / 16, 1, loadconfig);
    }
    auto load_c = [&]() {
        _tile_loadd(0, C + 0 * ldc + 0, ldc * sizeof(float));
    };
    auto zero_c = [&]() {
        _tile_zero(0);
    };

    if constexpr (accum) {
        load_c();
    } else {
        zero_c();
    }

    auto compute = [&](int k) {
        _tile_loadd(1, A + 0 * lda + k, lda * sizeof(uint8_t));
        _tile_loadd(2, B + k * ldb + 0, ldb * 4 * sizeof(uint8_t));
        // _tile_dpbf16ps(0, 1, 2);
        _tile_dpbusd(0, 1, 2);
    };

    #pragma GCC unroll 4
    for (int k = 0; k < last_k_offset; k += 16) {
        compute(k);
    }

    auto store_c = [&]() {
    // store to C
        _tile_stored(0, C + 0 * ldc + 0, ldc * sizeof(float));
    };

    // TODO(jgong5): move tail k computation to separate loopnest to save tile configuration overhead
    if C10_UNLIKELY (tail_k_size > 0) {
        if C10_LIKELY (last_k_offset > 0) {
            store_c();
            amx_state.configure(tilecfg_rows, tail_k_size * sizeof(uint8_t), 16 / 16, 1, loadconfig);
            load_c();
        }
        compute(last_k_offset);
    }

    store_c();
}

template <bool accum>
inline void kernel_micro_gemm(
    AMXState& amx_state,
    const uint8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    int32_t* __restrict__ C,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc
) {
    TORCH_CHECK(N % 16 == 0, "N dimension must be multiple of 16");
    TORCH_CHECK(K % 2 == 0, "K dimension must be multiple of 2");
    // TODO(jgong5): loop unroll for M and N
    for (int64_t m = 0; m < M; m += 32) {
        int64_t block_m = std::min<int64_t>(M - m, 32);
        int64_t m_tail = m;
        for (int64_t n = 0; n < N; n += 16) {
            if (block_m >= 32) {
                kernel_micro_gemm_amx_kernel_32_1<accum>(
                    amx_state,
                    A + m * lda,
                    B + n,
                    C + m * ldc + n,
                    K,
                    lda,
                    ldb,
                    ldc,
                    16
                );
                block_m -= 32;
                m_tail += 32;
            }
            else
            if (block_m >= 16) {
                kernel_micro_gemm_amx_kernel_16_1<accum>(
                    amx_state,
                    A + m * lda,
                    B + n,
                    C + m * ldc + n,
                    K,
                    lda,
                    ldb,
                    ldc,
                    16
                );
                block_m -= 16;
                m_tail += 16;
            }
            if (block_m > 0) {
                kernel_micro_gemm_amx_kernel_16_1<accum>(
                    amx_state,
                    A + m_tail * lda,
                    B + n,
                    C + m_tail * ldc + n,
                    K,
                    lda,
                    ldb,
                    ldc,
                    block_m
                );
            }
        }
    }
}

extern "C"
void kernel(const uint8_t* X, const float* x_scale, const int32_t* x_zp, const int8_t* W, const float* w_scale, const int32_t* w_zp, const float* in_ptr6, float* Y)
{

    constexpr int64_t num_threads = 1;
    constexpr int64_t N = static_cast<long>(64L);
    constexpr int64_t K = static_cast<long>(128L);
    constexpr int64_t M0 = 32;
    constexpr int64_t N0 = 16;
    constexpr int64_t K0 = 16;
    constexpr int64_t N0_blocks = (N + N0 - 1) / N0;
    constexpr int64_t K0_blocks = (K + K0 - 1) / K0;
    static_assert(N % N0 == 0, "N dimension must be multiple of N0");
    // const float* b_compensate = BMatricCompo;
    // TODO(jgong5): improve cache blocking with CPU info (Mc, Kc)
    constexpr int64_t M = static_cast<long>(32L);
    constexpr int64_t M0_blocks = (M + M0 - 1) / M0;
    constexpr int64_t Mt_blocks = 1;
    constexpr int64_t Nt_blocks = 4;
    constexpr int64_t Kt_blocks = 8;
    constexpr int64_t Mc_blocks = 1;
    constexpr int64_t Kc_blocks = 8;
    // TODO(jgong5): support k-slicing
    TORCH_CHECK(Kt_blocks == K0_blocks, "Do not support k slicing yet.");
    // make sure all partitions are assigned
    TORCH_CHECK(
        Mt_blocks * Nt_blocks * Kt_blocks * 1 >= M0_blocks * N0_blocks * K0_blocks,
        "Not all partitions are assigned."
    );
    {
        int64_t m_block_start = 0;
        int64_t m_block_end = M0_blocks;
        int64_t n_block_start = 0;
        int64_t n_block_end = N0_blocks;
        int64_t k_block_start = 0;
        int64_t k_block_end = K0_blocks;
        AMXState amx_state;
        for (int64_t mc = m_block_start; mc < m_block_end; mc += Mc_blocks) {
            const int64_t m_start = mc * M0;
            const int64_t m_end = std::min((mc + Mc_blocks) * M0, M);
            const int64_t m_size = m_end - m_start;
            for (int64_t nc = n_block_start; nc < n_block_end; ++nc) {
                const int64_t n_start = nc * N0;
                const int64_t n_size = N0;
                auto _local_acc_buf = std::make_unique<int32_t[]>(static_cast<long>(N0*(m_end + ((-1L)*m_start)))); auto local_acc_buf = _local_acc_buf.get();
                for (int64_t kc = k_block_start; kc < k_block_end; kc += Kc_blocks) {
                    int64_t k_start = kc * K0;
                    int64_t k_end = std::min((kc + Kc_blocks) * K0, K);
                    if (kc == k_block_start) {
                        kernel_micro_gemm<static_cast<bool>(false)>(
                            amx_state,
                            &(X[static_cast<long>(k_start + (128L*m_start))]),
                            &(W[static_cast<long>((16L*k_start) + (2048L*nc))]),
                            &(local_acc_buf[static_cast<long>(0L)]),
                            static_cast<long>(m_end + ((-1L)*m_start)),
                            static_cast<long>(N0),
                            static_cast<long>(k_end + ((-1L)*k_start)),
                            static_cast<long>(128L),
                            static_cast<long>(16L),
                            static_cast<long>(N0)
                        );

                    } else {
                        TORCH_CHECK(false, "Shouldn't hit this branch");
                        kernel_micro_gemm<static_cast<bool>(true)>(
                            amx_state,
                            &(X[static_cast<long>(k_start + (128L*m_start))]),
                            &(W[static_cast<long>((16L*k_start) + (2048L*nc))]),
                            &(local_acc_buf[static_cast<long>(0L)]),
                            static_cast<long>(m_end + ((-1L)*m_start)),
                            static_cast<long>(N0),
                            static_cast<long>(k_end + ((-1L)*k_start)),
                            static_cast<long>(128L),
                            static_cast<long>(16L),
                            static_cast<long>(N0)
                        );

                    }
                }
                std::cout<<"--- start to print int32 result----"<<std::endl;
                for (int _m=m_start; _m<m_end ; _m++) {
                    for (int _n=n_start; _n < n_start + N0; _n ++ ) {
                        std::cout<<local_acc_buf[_n + (N0*_m)]<<" ";
                    }
                    std::cout<<std::endl;                                        
                }
                {
                    #pragma GCC ivdep
                    for(long x0=static_cast<long>(0L); x0<static_cast<long>(m_end + ((-1L)*m_start)); x0+=static_cast<long>(1L))
                    {
                        for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L*(c10::div_floor_integer(N0, 16L))); x1+=static_cast<long>(16L))
                        {
                            auto tmp0 = at::vec::Vectorized<int32_t>::loadu(local_acc_buf + static_cast<long>(x1 + (N0*x0)), 16);
                            auto tmp2 = x_scale[static_cast<long>(0L)];
                            auto tmp3 = x_zp[static_cast<long>(0L)];
                            auto tmp4 = at::vec::Vectorized<float>::loadu(w_scale + static_cast<long>(n_start + x1), 16);
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(n_start + x1), 16);
                            auto tmp1 = at::vec::convert<float>(tmp0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp2);
                            auto tmp7 = tmp1 * tmp6;
                            auto tmp8 = tmp7 * tmp4;
                            auto tmp9 = tmp6 * tmp4;
                            auto tmp10 = c10::convert<float>(tmp3);
                            auto tmp11 = at::vec::Vectorized<float>(tmp10);
                            auto tmp12 = tmp9 * tmp11;
                            auto tmp13 = tmp12 * tmp5;
                            auto tmp14 = tmp8 - tmp13;
                            tmp14.store(Y + static_cast<long>(n_start + x1 + (64L*m_start) + (64L*x0)));
                        }
                        #pragma omp simd simdlen(8) 
                        for(long x1=static_cast<long>(16L*(c10::div_floor_integer(N0, 16L))); x1<static_cast<long>(N0); x1+=static_cast<long>(1L))
                        {
                            auto tmp0 = local_acc_buf[static_cast<long>(x1 + (N0*x0))];
                            auto tmp2 = x_scale[static_cast<long>(0L)];
                            auto tmp3 = x_zp[static_cast<long>(0L)];
                            auto tmp4 = w_scale[static_cast<long>(n_start + x1)];
                            auto tmp5 = in_ptr6[static_cast<long>(n_start + x1)];
                            auto tmp1 = c10::convert<float>(tmp0);
                            auto tmp6 = decltype(tmp1)(tmp1 * tmp2);
                            auto tmp7 = decltype(tmp6)(tmp6 * tmp4);
                            auto tmp8 = decltype(tmp2)(tmp2 * tmp4);
                            auto tmp9 = decltype(tmp8)(tmp8 * tmp3);
                            auto tmp10 = decltype(tmp9)(tmp9 * tmp5);
                            auto tmp11 = decltype(tmp7)(tmp7 - tmp10);
                            Y[static_cast<long>(n_start + x1 + (64L*m_start) + (64L*x0))] = tmp11;
                        }
                    }
                }

            }
        }
        amx_state.release([]() { _tile_release(); });
    }
}
''')


async_compile.wait(globals())
del async_compile

# def call(args):
#     arg3_1, = args
#     args.clear()
#     assert_size_stride(arg3_1, (32, 128), (128, 1))
#     buf0 = empty_strided_cpu((32, 128), (128, 1), torch.uint8)
#     cpp_fused_quantize_per_tensor_0(arg3_1, buf0)
#     del arg3_1


#     # Calculate the reference result
#     ref_res = empty_strided_cpu((32, 64), (64, 1), torch.float32)

#     print("buf0 is: {}".format(buf0), flush=True)
#     print("constant7 is: {}".format(constant7), flush=True)
#     print("ref_res is: {}".format(ref_res), flush=True)
#     print("_frozen_param0 is: {}".format(_frozen_param0), flush=True)
#     print("_frozen_param1_0 is: {}".format(_frozen_param1_0), flush=True)

#     cpp_fused__softmax_0_inner_product(buf0, constant7, ref_res, _frozen_param0, _frozen_param1_0, x_scale, x_zp)
#     print("---- inner_product res is: {}".format(ref_res), flush=True)

#     buf1 = empty_strided_cpu((32, 64), (64, 1), torch.float32)
#     cpp_fused_mm_quantize_per_tensor_1(buf0, x_scale, x_zp, constant7, _frozen_param0, _frozen_param1_0, _frozen_param3BMatricCompo, buf1)
#     print("---- amx int res is: {}".format(buf1), flush=True)

#     print(torch.allclose(ref_res, buf1), flush=True)
    
#     return (buf1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    global _frozen_param0
    _frozen_param0 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    global _frozen_param1
    _frozen_param1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.int64)
    global _frozen_param3
    _frozen_param3 = rand_strided((128, 64), (1, 0), device='cpu', dtype=torch.int8)
    global x_scale
    x_scale = rand_strided((), (), device='cpu', dtype=torch.float32)
    global x_zp
    x_zp = rand_strided((), (), device='cpu', dtype=torch.int32)
    global _frozen_param1_0
    _frozen_param1_0 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.int32)
    global _frozen_param3BMatricCompo
    _frozen_param3BMatricCompo = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    global constant7
    # constant7 = rand_strided((2, 128, 32), (4096, 32, 1), device='cpu', dtype=torch.int8)
    
    b = rand_strided((128, 64), (64, 1), device='cpu', dtype=torch.float32)
    b = torch.quantize_per_channel(b, _frozen_param0, _frozen_param1_0, 1, torch.qint8).dequantize()

    arg3_1 = rand_strided((32, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg3_1 = torch.quantize_per_tensor(arg3_1, x_scale, x_zp, torch.quint8).dequantize()

    fp32_res = torch.matmul(arg3_1, b)
    print("---- fp32_res is: {}".format(fp32_res), flush=True)

    constant7 = torch.ops.quantized_decomposed.quantize_per_channel(
        b,
        _frozen_param0,
        _frozen_param1_0,
        1,
        -128,
        127,
        torch.int8,
    )
    # constant7 = rand_strided((128, 64), (64, 1), device='cpu', dtype=torch.int8)
    # Change to VNNI Layout

    # fn = lambda: call([arg3_1])
    # return print_performance(fn, times=times, repeat=repeat)
    # res = fn()


    assert_size_stride(arg3_1, (32, 128), (128, 1))
    buf0 = empty_strided_cpu((32, 128), (128, 1), torch.uint8)
    # cpp_fused_quantize_per_tensor_0(arg3_1, buf0)
    # print("----- arg3_1 is: {}".format(arg3_1), flush=True)
    buf0 = torch.clamp(torch.round(arg3_1 / x_scale) + x_zp, 0, 255).to(torch.uint8)
    del arg3_1


    # Calculate the reference result
    inner_product_res = empty_strided_cpu((32, 64), (64, 1), torch.float32)

    # print("----- start to print the inputs -----", flush=True)
    # print("buf0 is: {}".format(buf0), flush=True)
    # print("constant7 is: {}".format(constant7), flush=True)
    # print("inner_product_res is: {}".format(inner_product_res), flush=True)
    # print("_frozen_param0 is: {}".format(_frozen_param0), flush=True)
    # print("_frozen_param1_0 is: {}".format(_frozen_param1_0), flush=True)
    # print("x_scale is: {}".format(x_scale), flush=True)
    # print("x_zp is: {}".format(x_zp), flush=True)

    cpp_fused__softmax_0_inner_product(buf0, constant7, inner_product_res, _frozen_param0, _frozen_param1_0, x_scale, x_zp)
    print("---- inner_product res is: {}".format(inner_product_res), flush=True)


    buf1 = empty_strided_cpu((32, 64), (64, 1), torch.float32)
    # Pack to VNNI Layout
    n= 64
    k = 128
    vnni_size = 4
    block_n = 16
    constant7 = constant7.contiguous().reshape(k, n // block_n, block_n)
    constant7 = constant7.transpose(0, 1).contiguous()
    packed_constant7 = constant7.view(n // block_n, k // vnni_size, vnni_size, block_n)
    packed_constant7 = packed_constant7.transpose(-1, -2).contiguous().view(n // block_n, k, block_n)
    cpp_fused_mm_quantize_per_tensor_1(buf0, x_scale, x_zp, packed_constant7, _frozen_param0, _frozen_param1_0, _frozen_param3BMatricCompo, buf1)
    print("---- amx int res is: {}".format(buf1), flush=True)

    print(torch.allclose(fp32_res, inner_product_res, rtol=1e-3, atol=1e-3), flush=True)
    print(torch.allclose(fp32_res, buf1, rtol=1e-3, atol=1e-3), flush=True)

    return 1.0

if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)

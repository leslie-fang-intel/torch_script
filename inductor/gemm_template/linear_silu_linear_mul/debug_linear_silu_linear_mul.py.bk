# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
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
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
_frozen_param4 = None  # device(type='cpu') torch.bfloat16 (11008, 4096) (1, 0) 7f85c7ecd8f0
_frozen_param5 = None  # device(type='cpu') torch.bfloat16 (11008, 4096) (1, 0) 7f85c7eec770
constant2 = None  # device(type='cpu') torch.bfloat16 (344, 4096, 32) (131072, 32, 1) 7f85c939a890
constant3 = None  # device(type='cpu') torch.bfloat16 (344, 4096, 32) (131072, 32, 1) 7f85c6f6b920


cpp_fused_mul_0 = async_compile.cpp_pybinding(['const bfloat16*', 'const bfloat16*', 'const bfloat16*', 'bfloat16*'], '''
#include "/home/leslie/lz/pytorch/torch/_inductor/codegen/cpp_prefix.h"
#include <c10/util/Unroll.h>


template <bool has_gate_bias, bool has_up_bias>
inline void kernel_micro_gemm_silu_mul_epilogue_fusion(
    float* in_ptr0,
    float* in_ptr1,
    const bfloat16* inp0,
    const bfloat16* inp1,
    bfloat16* out_ptr,
    int64_t M,
    int64_t N,
    int64_t in_lda,
    int64_t out_lda) {
    int64_t n_scalar_start = 0;
#if defined(CPU_CAPABILITY_AVX512) && !defined(_MSC_VER)
    using Vectorized_fp32 = at::vec::Vectorized<float>;
    using Vectorized_bf16 = at::vec::Vectorized<bfloat16>;
    int64_t N1 = N / 32 * 32;
    int64_t N2 = N / 16 * 16;
    n_scalar_start = N2;
#endif
    for (int64_t m = 0; m < M; m += 1) {
#if defined(CPU_CAPABILITY_AVX512) && !defined(_MSC_VER)
        for (int64_t n = 0; n < N1; n += 32) {
            Vectorized_fp32 tmp0 = Vectorized_fp32::loadu(in_ptr0 + (m * in_lda + n));
            Vectorized_fp32 tmp0_1 = Vectorized_fp32::loadu(in_ptr0 + (m * in_lda + n + 16));
            Vectorized_fp32 tmp1 = Vectorized_fp32::loadu(in_ptr1 + (m * in_lda + n));
            Vectorized_fp32 tmp1_1 = Vectorized_fp32::loadu(in_ptr1 + (m * in_lda + n + 16));
            if constexpr (has_gate_bias) {
                const auto inp_bf16_vec = Vectorized_bf16::loadu(inp0 + n);
                at::vec::VectorizedN<float, 2> inp_fp32_vecs = at::vec::convert<float, 2, bfloat16, 1>(inp_bf16_vec);
                tmp0 = tmp0 + Vectorized_fp32(inp_fp32_vecs[0]);
                tmp0_1 = tmp0_1 + Vectorized_fp32(inp_fp32_vecs[1]);
            }
            if constexpr (has_up_bias) {
                const auto inp_bf16_vec = Vectorized_bf16::loadu(inp1 + n);
                at::vec::VectorizedN<float, 2> inp_fp32_vecs = at::vec::convert<float, 2, bfloat16, 1>(inp_bf16_vec);
                tmp1 = tmp1 + Vectorized_fp32(inp_fp32_vecs[0]);
                tmp1_1 = tmp1_1 + Vectorized_fp32(inp_fp32_vecs[1]);
            }
            tmp0 = tmp0 * (decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp()));
            tmp0_1 = tmp0_1 * (decltype(tmp0_1)(1)/(decltype(tmp0_1)(1) + tmp0_1.neg().exp()));
            tmp0 = tmp0 * tmp1;
            tmp0_1 = tmp0_1 * tmp1_1;
            at::vec::VectorizedN<float, 2> out_fp32_vec(tmp0, tmp0_1);
            Vectorized_bf16 out_bf16_vec = at::vec::convert<bfloat16, 1, float, 2>(out_fp32_vec);
            out_bf16_vec.store(out_ptr + (m * out_lda + n));
        }
        for (int64_t n = N1; n < N2; n += 16) {
            Vectorized_fp32 tmp0 = Vectorized_fp32::loadu(in_ptr0 + (m * in_lda + n));
            Vectorized_fp32 tmp1 = Vectorized_fp32::loadu(in_ptr1 + (m * in_lda + n));
            if constexpr (has_gate_bias) {
                const auto inp_bf16_vec = Vectorized_bf16::loadu(inp0 + n, 16);
                Vectorized_fp32 inp_fp32_vec = at::vec::convert<float, 1, bfloat16, 1>(inp_bf16_vec);
                tmp0 = tmp0 + inp_fp32_vec;
            }
            if constexpr (has_up_bias) {
                const auto inp_bf16_vec = Vectorized_bf16::loadu(inp1 + n, 16);
                Vectorized_fp32 inp_fp32_vec = at::vec::convert<float, 1, bfloat16, 1>(inp_bf16_vec);
                tmp1 = tmp1 + inp_fp32_vec;
            }
            tmp0 = tmp0 * (decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp()));
            tmp0 = tmp0 * tmp1;
            Vectorized_bf16 out_bf16_vec = at::vec::convert<bfloat16, 1, float, 1>(tmp0);
            out_bf16_vec.store(out_ptr + (m * out_lda + n), 16);
        }
#endif
        for (int64_t n = n_scalar_start; n < N; n += 1) {
            float tmp0 = in_ptr0[m * in_lda + n];
            float tmp1 = in_ptr1[m * in_lda + n];
            // Bias add
            if constexpr (has_gate_bias) {
                tmp0 = tmp0 + (float)inp0[n];
            }
            if constexpr (has_up_bias) {
                tmp1 = tmp1 + (float)inp1[n];
            }
            // Silu
            tmp0 = tmp0 * (1.0 / ( 1.0 + std::exp(-tmp0)));
            // Mul
            tmp0 = tmp0 * tmp1;
            // Store output
            out_ptr[m * out_lda + n] = (bfloat16)tmp0;
        }
    }
}


template <bool accum>
inline void kernel_micro_gemm_amx_kernel_32_2(
    AMXState& amx_state,
    const bfloat16* __restrict__ A,
    const bfloat16* __restrict__ B,
    float* __restrict__ C,
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
    const auto last_k_offset = K / 32 * 32;
    const auto tail_k_size = K - last_k_offset;
    if C10_LIKELY (last_k_offset > 0) {
        amx_state.configure(tilecfg_rows, 64, 32 / 16, 2, loadconfig);
    } else {
        amx_state.configure(tilecfg_rows, tail_k_size * sizeof(bfloat16), 32 / 16, 2, loadconfig);
    }
    auto load_c = [&]() {
        _tile_loadd(0, C + 0 * ldc + 0, ldc * sizeof(float));
        _tile_loadd(1, C + 0 * ldc + 16, ldc * sizeof(float));
        _tile_loadd(2, C + 16 * ldc + 0, ldc * sizeof(float));
        _tile_loadd(3, C + 16 * ldc + 16, ldc * sizeof(float));
    };
    auto zero_c = [&]() {
        _tile_zero(0);
        _tile_zero(1);
        _tile_zero(2);
        _tile_zero(3);
    };

    if constexpr (accum) {
        load_c();
    } else {
        zero_c();
    }

    auto compute = [&](int k) {
        _tile_stream_loadd(4, A + 0 * lda + k, lda * sizeof(bfloat16));
        _tile_loadd(6, B + k * ldb + 0, ldb * 2 * sizeof(bfloat16));
        _tile_dpbf16ps(0, 4, 6);
        _tile_loadd(7, B + k * ldb + 32, ldb * 2 * sizeof(bfloat16));
        _tile_dpbf16ps(1, 4, 7);
        _tile_stream_loadd(5, A + 16 * lda + k, lda * sizeof(bfloat16));
        _tile_dpbf16ps(2, 5, 6);
        _tile_dpbf16ps(3, 5, 7);
    };

    #pragma GCC unroll 4
    for (int k = 0; k < last_k_offset; k += 32) {
        compute(k);
    }

    auto store_c = [&]() {
    // store to C
        _tile_stored(0, C + 0 * ldc + 0, ldc * sizeof(float));
        _tile_stored(1, C + 0 * ldc + 16, ldc * sizeof(float));
        _tile_stored(2, C + 16 * ldc + 0, ldc * sizeof(float));
        _tile_stored(3, C + 16 * ldc + 16, ldc * sizeof(float));
    };

    // TODO(jgong5): move tail k computation to separate loopnest to save tile configuration overhead
    if C10_UNLIKELY (tail_k_size > 0) {
        if C10_LIKELY (last_k_offset > 0) {
            store_c();
            amx_state.configure(tilecfg_rows, tail_k_size * sizeof(bfloat16), 32 / 16, 2, loadconfig);
            load_c();
        }
        compute(last_k_offset);
    }

    store_c();
}
template <bool accum>
inline void kernel_micro_gemm_amx_kernel_16_2(
    AMXState& amx_state,
    const bfloat16* __restrict__ A,
    const bfloat16* __restrict__ B,
    float* __restrict__ C,
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
    const auto last_k_offset = K / 32 * 32;
    const auto tail_k_size = K - last_k_offset;
    if C10_LIKELY (last_k_offset > 0) {
        amx_state.configure(tilecfg_rows, 64, 16 / 16, 2, loadconfig);
    } else {
        amx_state.configure(tilecfg_rows, tail_k_size * sizeof(bfloat16), 16 / 16, 2, loadconfig);
    }
    auto load_c = [&]() {
        _tile_loadd(0, C + 0 * ldc + 0, ldc * sizeof(float));
        _tile_loadd(1, C + 0 * ldc + 16, ldc * sizeof(float));
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
        _tile_stream_loadd(2, A + 0 * lda + k, lda * sizeof(bfloat16));
        _tile_loadd(3, B + k * ldb + 0, ldb * 2 * sizeof(bfloat16));
        _tile_dpbf16ps(0, 2, 3);
        _tile_loadd(4, B + k * ldb + 32, ldb * 2 * sizeof(bfloat16));
        _tile_dpbf16ps(1, 2, 4);
    };

    #pragma GCC unroll 4
    for (int k = 0; k < last_k_offset; k += 32) {
        compute(k);
    }

    auto store_c = [&]() {
    // store to C
        _tile_stored(0, C + 0 * ldc + 0, ldc * sizeof(float));
        _tile_stored(1, C + 0 * ldc + 16, ldc * sizeof(float));
    };

    // TODO(jgong5): move tail k computation to separate loopnest to save tile configuration overhead
    if C10_UNLIKELY (tail_k_size > 0) {
        if C10_LIKELY (last_k_offset > 0) {
            store_c();
            amx_state.configure(tilecfg_rows, tail_k_size * sizeof(bfloat16), 16 / 16, 2, loadconfig);
            load_c();
        }
        compute(last_k_offset);
    }

    store_c();
}

template <bool accum>
inline void kernel_micro_gemm(
    AMXState& amx_state,
    const bfloat16* __restrict__ A,
    const bfloat16* __restrict__ B,
    float* __restrict__ C,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc
) {
    AOTI_TORCH_CHECK(N % 32 == 0, "N dimension must be multiple of 32");
    AOTI_TORCH_CHECK(K % 2 == 0, "K dimension must be multiple of 2");
    // TODO(jgong5): loop unroll for M and N
    for (int64_t m = 0; m < M; m += 32) {
        int64_t block_m = std::min<int64_t>(M - m, 32);
        int64_t m_tail = m;
        for (int64_t n = 0; n < N; n += 32) {
            if (block_m >= 32) {
                kernel_micro_gemm_amx_kernel_32_2<accum>(
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
                kernel_micro_gemm_amx_kernel_16_2<accum>(
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
                kernel_micro_gemm_amx_kernel_16_2<accum>(
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
void kernel(const bfloat16* X, const bfloat16* W, const bfloat16* W1, bfloat16* Y)
{

    constexpr int64_t num_threads = 32;
    constexpr int64_t N = 11008;
    constexpr int64_t K = 4096;
    constexpr int64_t Mr = 32;
    constexpr int64_t Nr = 32;
    constexpr int64_t Kr = 32;
    constexpr int64_t Nr_blocks = (N + Nr - 1) / Nr;
    constexpr int64_t Kr_blocks = (K + Kr - 1) / Kr;
    constexpr int64_t M = static_cast<int64_t>(32256L);
    constexpr int64_t Mr_blocks = (M + Mr - 1) / Mr;
    constexpr int64_t Mt_blocks = 126;
    constexpr int64_t Nt_blocks = 86;
    constexpr int64_t Kt_blocks = 128;
    constexpr int64_t Mc_blocks = 4;
    constexpr int64_t Nc_blocks = 1;
    constexpr int64_t Kc_blocks = 19;
    constexpr int64_t num_Mc_blocks = (Mr_blocks + Mc_blocks - 1) / Mc_blocks;
    constexpr int64_t num_Nc_blocks = (Nr_blocks + Nc_blocks - 1) / Nc_blocks;
    constexpr int64_t num_Mt_blocks = (Mr_blocks + Mt_blocks - 1) / Mt_blocks;
    constexpr int64_t num_Nt_blocks = (Nr_blocks + Nt_blocks - 1) / Nt_blocks;
    constexpr int64_t num_Kt_blocks = (Kr_blocks + Kt_blocks - 1) / Kt_blocks;

    // make sure all partitions are assigned
    AOTI_TORCH_CHECK(
        Mt_blocks * Nt_blocks * Kt_blocks * 32 >= Mr_blocks * Nr_blocks * Kr_blocks,
        "Not all partitions are assigned."
    );
    #pragma omp parallel num_threads(32)
    {
        const int tid = omp_get_thread_num();
        const int64_t k_group_id = tid / num_Kt_blocks;
        const int64_t k_slice_id = tid % num_Kt_blocks;
        const int64_t n_group_id = k_group_id / num_Nt_blocks;
        const int64_t n_slice_id = k_group_id % num_Nt_blocks;
        const int64_t k_block_start = k_slice_id * Kt_blocks;
        const int64_t k_block_end = std::min(k_block_start + Kt_blocks, Kr_blocks);
        const int64_t n_block_start = n_slice_id * Nt_blocks;
        const int64_t n_block_end = std::min(n_block_start + Nt_blocks, Nr_blocks);
        const int64_t m_block_start = std::min(n_group_id * Mt_blocks, Mr_blocks);
        const int64_t m_block_end = std::min(m_block_start + Mt_blocks, Mr_blocks);
        const int64_t num_Mc_blocks_per_thread = (m_block_end - m_block_start + Mc_blocks - 1) / Mc_blocks;
        AMXState amx_state;
        auto _local_acc_buf = std::make_unique<float[]>(static_cast<int64_t>(Mc_blocks*Mr*Nc_blocks*Nr)); auto local_acc_buf = _local_acc_buf.get();
        auto _local_acc_buf2 = std::make_unique<float[]>(static_cast<int64_t>(Mc_blocks*Mr*Nc_blocks*Nr)); auto local_acc_buf2 = _local_acc_buf2.get();
        for (int64_t mc_block_id = 0; mc_block_id < num_Mc_blocks_per_thread; mc_block_id++) {
            const int64_t my_mc_block_id = (mc_block_id + n_slice_id) % num_Mc_blocks_per_thread;
            const int64_t mc = m_block_start + my_mc_block_id * Mc_blocks;
            const int64_t m_start = mc * Mr;
            const int64_t m_end = std::min(std::min(mc + Mc_blocks, m_block_end) * Mr, M);
            const int64_t m_size = m_end - m_start;
            for (int64_t nc = n_block_start; nc < n_block_end; nc += Nc_blocks) {
                const int64_t n_start = nc * Nr;
                const int64_t n_end = std::min(std::min(nc + Nc_blocks, n_block_end) * Nr, N);
                const int64_t n_size = n_end - n_start;
                // NB: assume we pad N, nc_block_end won't exceed padded N here.
                const int64_t nc_block_end = std::min(nc + Nc_blocks, n_block_end);
                if (_local_acc_buf == nullptr) { _local_acc_buf = std::make_unique<float[]>(static_cast<int64_t>(Mc_blocks*Mr*Nc_blocks*Nr)); local_acc_buf = _local_acc_buf.get(); }
                if (_local_acc_buf2 == nullptr) { _local_acc_buf2 = std::make_unique<float[]>(static_cast<int64_t>(Mc_blocks*Mr*Nc_blocks*Nr)); local_acc_buf2 = _local_acc_buf2.get(); }
                for (int64_t kc = k_block_start; kc < k_block_end; kc += Kc_blocks) {
                    int64_t k_start = kc * Kr;
                    int64_t k_end = std::min(std::min(kc + Kc_blocks, k_block_end) * Kr, K);
                    for (int64_t nci = nc; nci < nc_block_end; nci++) {
                        if (kc == k_block_start) {

                            kernel_micro_gemm<static_cast<bool>(false)>(
                                amx_state,
                                &(X[static_cast<int64_t>(k_start + (4096L*m_start))]),
                                &(W[static_cast<int64_t>((32L*k_start) + (131072L*nci))]),
                                &(local_acc_buf[static_cast<int64_t>((Nr*nci) + ((-1L)*Nr*nc))]),
                                static_cast<int64_t>(m_end + ((-1L)*m_start)),
                                static_cast<int64_t>(Nr),
                                static_cast<int64_t>(k_end + ((-1L)*k_start)),
                                static_cast<int64_t>(4096L),
                                static_cast<int64_t>(32L),
                                static_cast<int64_t>(Nc_blocks*Nr)
                            );

                                              
                            kernel_micro_gemm<static_cast<bool>(false)>(
                                amx_state,
                                &(X[static_cast<int64_t>(k_start + (4096L*m_start))]),
                                &(W1[static_cast<int64_t>((32L*k_start) + (131072L*nci))]),
                                &(local_acc_buf2[static_cast<int64_t>((Nr*nci) + ((-1L)*Nr*nc))]),
                                static_cast<int64_t>(m_end + ((-1L)*m_start)),
                                static_cast<int64_t>(Nr),
                                static_cast<int64_t>(k_end + ((-1L)*k_start)),
                                static_cast<int64_t>(4096L),
                                static_cast<int64_t>(32L),
                                static_cast<int64_t>(Nc_blocks*Nr)
                            );


                        } else {

                            kernel_micro_gemm<static_cast<bool>(true)>(
                                amx_state,
                                &(X[static_cast<int64_t>(k_start + (4096L*m_start))]),
                                &(W[static_cast<int64_t>((32L*k_start) + (131072L*nci))]),
                                &(local_acc_buf[static_cast<int64_t>((Nr*nci) + ((-1L)*Nr*nc))]),
                                static_cast<int64_t>(m_end + ((-1L)*m_start)),
                                static_cast<int64_t>(Nr),
                                static_cast<int64_t>(k_end + ((-1L)*k_start)),
                                static_cast<int64_t>(4096L),
                                static_cast<int64_t>(32L),
                                static_cast<int64_t>(Nc_blocks*Nr)
                            );


                            kernel_micro_gemm<static_cast<bool>(true)>(
                                amx_state,
                                &(X[static_cast<int64_t>(k_start + (4096L*m_start))]),
                                &(W1[static_cast<int64_t>((32L*k_start) + (131072L*nci))]),
                                &(local_acc_buf2[static_cast<int64_t>((Nr*nci) + ((-1L)*Nr*nc))]),
                                static_cast<int64_t>(m_end + ((-1L)*m_start)),
                                static_cast<int64_t>(Nr),
                                static_cast<int64_t>(k_end + ((-1L)*k_start)),
                                static_cast<int64_t>(4096L),
                                static_cast<int64_t>(32L),
                                static_cast<int64_t>(Nc_blocks*Nr)
                            );


                        }
                    }
                }

                {
                    // silu-mul epilogues

  kernel_micro_gemm_silu_mul_epilogue_fusion<static_cast<bool>(false), static_cast<bool>(false)>(&(local_acc_buf[static_cast<int64_t>(0L)]), &(local_acc_buf2[static_cast<int64_t>(0L)]), &(Y[static_cast<int64_t>(n_start + (11008L*m_start))]), &(Y[static_cast<int64_t>(n_start + (11008L*m_start))]), &(Y[static_cast<int64_t>(n_start + (11008L*m_start))]), static_cast<int64_t>(m_end + ((-1L)*m_start)), static_cast<int64_t>(n_end + ((-1L)*n_start)), static_cast<int64_t>(Nc_blocks*Nr), static_cast<int64_t>(11008L));

                }
            }
        }
        amx_state.release([]() { _tile_release(); });
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg2_1, = args
    args.clear()
    assert_size_stride(arg2_1, (32256, 4096), (4096, 1))
    buf0 = empty_strided_cpu((32256, 11008), (11008, 1), torch.bfloat16)
    cpp_fused_mul_0(arg2_1, constant2, constant3, buf0)
    del arg2_1
    return (buf0, )


def get_ref_res(x, W, W1):
    x1 = torch.nn.functional.linear(x, W.transpose(0, 1).contiguous())
    x1 = torch.nn.functional.silu(x1)
    x2 = torch.nn.functional.linear(x, W1.transpose(0, 1).contiguous())
    return x1 * x2


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    global _frozen_param4
    _frozen_param4 = rand_strided((11008, 4096), (1, 0), device='cpu', dtype=torch.bfloat16)
    global _frozen_param5
    _frozen_param5 = rand_strided((11008, 4096), (1, 0), device='cpu', dtype=torch.bfloat16)
    global constant2
    # constant2 = rand_strided((344, 4096, 32), (131072, 32, 1), device='cpu', dtype=torch.bfloat16)
    W = rand_strided((4096, 11008), (11008, 1), device='cpu', dtype=torch.bfloat16) # K * N
    block_n = 32
    vnni_size = 2
    constant2 = W.reshape(4096, 11008 // block_n, block_n).transpose(0, 1).contiguous()
    constant2 = constant2.view(11008 // block_n, 4096 // vnni_size, vnni_size, block_n).transpose(-1, -2).contiguous().view(11008 // block_n, 4096, block_n)

    global constant3
    # constant3 = rand_strided((344, 4096, 32), (131072, 32, 1), device='cpu', dtype=torch.bfloat16)
    W1 = rand_strided((4096, 11008), (11008, 1), device='cpu', dtype=torch.bfloat16) # K * N
    constant3 = W1.reshape(4096, 11008 // block_n, block_n).transpose(0, 1).contiguous()
    constant3 = constant3.view(11008 // block_n, 4096 // vnni_size, vnni_size, block_n).transpose(-1, -2).contiguous().view(11008 // block_n, 4096, block_n)

    arg2_1 = rand_strided((32256, 4096), (4096, 1), device='cpu', dtype=torch.bfloat16)
    
    fn = lambda: call([arg2_1])
    
    ref_res = get_ref_res(arg2_1, W, W1)
    res = fn()[0]

    # print("ref_res is: {}".format(ref_res), flush=True)
    # print("res is: {}".format(res), flush=True)

    print(torch.allclose(ref_res, res, atol=5e-2, rtol=5e-2), flush=True)
    
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)

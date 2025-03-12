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
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p
_frozen_param2 = None  # device(type='cpu') torch.bfloat16 (1024, 512) (1, 0) 7fd049709fd0
constant1 = None  # device(type='cpu') torch.bfloat16 (64, 512, 16) (8192, 16, 1) 7fd048d5e110


cpp_fused_mm_0 = async_compile.cpp_pybinding(['const bfloat16*', 'const bfloat16*', 'bfloat16*'], '''
#include "/home/leslie/lz/pytorch/torch/_inductor/codegen/cpp_prefix.h"
#include <c10/util/Unroll.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>




template <bool accum>
inline void kernel_micro_gemm_amx_kernel_48_1(
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
        amx_state.configure(tilecfg_rows, 64, 48 / 16, 1, loadconfig);
    } else {
        amx_state.configure(tilecfg_rows, tail_k_size * sizeof(bfloat16), 48 / 16, 1, loadconfig);
    }
    auto load_c = [&]() {
        _tile_loadd(0, C + 0 * ldc + 0, ldc * sizeof(float));
        _tile_loadd(1, C + 16 * ldc + 0, ldc * sizeof(float));
        _tile_loadd(2, C + 32 * ldc + 0, ldc * sizeof(float));
    };
    auto zero_c = [&]() {
        _tile_zero(0);
        _tile_zero(1);
        _tile_zero(2);
    };

    if constexpr (accum) {
        load_c();
    } else {
        zero_c();
    }

    auto compute = [&](int k) {
                                             std::cout<<"start to calculation 1"<<std::endl;
        _tile_stream_loadd(3, A + 0 * lda + k, lda * sizeof(bfloat16));
        _tile_loadd(6, B + k * ldb + 0, ldb * 2 * sizeof(bfloat16));
        _tile_dpbf16ps(0, 3, 6);
        _tile_stream_loadd(4, A + 16 * lda + k, lda * sizeof(bfloat16));
        _tile_dpbf16ps(1, 4, 6);
        _tile_stream_loadd(5, A + 32 * lda + k, lda * sizeof(bfloat16));
        _tile_dpbf16ps(2, 5, 6);
    };

    #pragma GCC unroll 4
    for (int k = 0; k < last_k_offset; k += 32) {
        compute(k);
    }

    auto store_c = [&]() {
    // store to C
        _tile_stored(0, C + 0 * ldc + 0, ldc * sizeof(float));
        _tile_stored(1, C + 16 * ldc + 0, ldc * sizeof(float));
        _tile_stored(2, C + 32 * ldc + 0, ldc * sizeof(float));
    };

    // TODO(jgong5): move tail k computation to separate loopnest to save tile configuration overhead
    if C10_UNLIKELY (tail_k_size > 0) {
        if C10_LIKELY (last_k_offset > 0) {
            store_c();
            amx_state.configure(tilecfg_rows, tail_k_size * sizeof(bfloat16), 48 / 16, 1, loadconfig);
            load_c();
        }
        compute(last_k_offset);
    }

    store_c();
}

template <bool accum>
inline void kernel_micro_gemm_amx_kernel_32_1(
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
        amx_state.configure(tilecfg_rows, 64, 32 / 16, 1, loadconfig);
    } else {
        amx_state.configure(tilecfg_rows, tail_k_size * sizeof(bfloat16), 32 / 16, 1, loadconfig);
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
        _tile_stream_loadd(2, A + 0 * lda + k, lda * sizeof(bfloat16));
        _tile_loadd(4, B + k * ldb + 0, ldb * 2 * sizeof(bfloat16));

        // _tile_dpbf16ps(0, 2, 4);
        std::cout<<"start to calculation"<<std::endl;
        _tile_dpfp16ps(0, 2, 4);

        _tile_stream_loadd(3, A + 16 * lda + k, lda * sizeof(bfloat16));
        _tile_dpbf16ps(1, 3, 4);
    };

    #pragma GCC unroll 4
    for (int k = 0; k < last_k_offset; k += 32) {
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
            amx_state.configure(tilecfg_rows, tail_k_size * sizeof(bfloat16), 32 / 16, 1, loadconfig);
            load_c();
        }
        compute(last_k_offset);
    }

    store_c();
}

template <bool accum>
inline void kernel_micro_gemm_amx_kernel_16_1(
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
        amx_state.configure(tilecfg_rows, 64, 16 / 16, 1, loadconfig);
    } else {
        amx_state.configure(tilecfg_rows, tail_k_size * sizeof(bfloat16), 16 / 16, 1, loadconfig);
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
        
        _tile_stream_loadd(1, A + 0 * lda + k, lda * sizeof(bfloat16));
        _tile_loadd(2, B + k * ldb + 0, ldb * 2 * sizeof(bfloat16));
        // std::cout<<"start to calculation2"<<std::endl;
        _tile_dpfp16ps(0, 1, 2);
    };

    #pragma GCC unroll 4
    for (int k = 0; k < last_k_offset; k += 32) {
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
            amx_state.configure(tilecfg_rows, tail_k_size * sizeof(bfloat16), 16 / 16, 1, loadconfig);
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
    AOTI_TORCH_CHECK(N % 16 == 0, "N dimension must be multiple of 16");
    AOTI_TORCH_CHECK(K % 2 == 0, "K dimension must be multiple of 2");
// The ldb would not be block_n if N != block_n
    const int64_t updated_ldb = ldb;
    // TODO(jgong5): loop unroll for M and N
    for (int64_t n = 0; n < N; n += 16) {
        for (int64_t m = 0; m < M; m += 48) {
            int64_t block_m = std::min<int64_t>(M - m, 48);
            int64_t m_tail = m;
            if (block_m >= 48) {
                kernel_micro_gemm_amx_kernel_48_1<accum>(
                    amx_state,
                    A + m * lda,
                    B + n,
                    C + m * ldc + n,
                    K,
                    lda,
                    updated_ldb,
                    ldc,
                    16
                );
                block_m -= 48;
                m_tail += 48;
            }
            else
            if (block_m >= 32) {
                kernel_micro_gemm_amx_kernel_32_1<accum>(
                    amx_state,
                    A + m * lda,
                    B + n,
                    C + m * ldc + n,
                    K,
                    lda,
                    updated_ldb,
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
                    updated_ldb,
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
                    updated_ldb,
                    ldc,
                    block_m
                );
            }
        }
    }
}

extern "C" 
void kernel(const bfloat16* X, const bfloat16* W, bfloat16* Y)
{


    constexpr int64_t num_threads = 56;
    constexpr int64_t N = 1024;
    constexpr int64_t K = 512;
    constexpr int64_t Mr = 48;
    constexpr int64_t Nr = 16;
    constexpr int64_t Kr = 32;
    constexpr int64_t Nr_blocks = (N + Nr - 1) / Nr;
    constexpr int64_t Kr_blocks = (K + Kr - 1) / Kr;
    constexpr int64_t M = static_cast<int64_t>(4L);
    constexpr int64_t Mr_blocks = (M + Mr - 1) / Mr;
    constexpr int64_t Mt_blocks = 1;
    constexpr int64_t Nt_blocks = 2;
    constexpr int64_t Kt_blocks = 16;
    constexpr int64_t Mc_blocks = 1;
    constexpr int64_t Nc_blocks = 1;
    constexpr int64_t Kc_blocks = 16;
    constexpr int64_t num_Mc_blocks = (Mr_blocks + Mc_blocks - 1) / Mc_blocks;
    constexpr int64_t num_Nc_blocks = (Nr_blocks + Nc_blocks - 1) / Nc_blocks;
    constexpr int64_t num_Mt_blocks = (Mr_blocks + Mt_blocks - 1) / Mt_blocks;
    constexpr int64_t num_Nt_blocks = (Nr_blocks + Nt_blocks - 1) / Nt_blocks;
    constexpr int64_t num_Kt_blocks = (Kr_blocks + Kt_blocks - 1) / Kt_blocks;

    // make sure all partitions are assigned
    AOTI_TORCH_CHECK(
        Mt_blocks * Nt_blocks * Kt_blocks * 56 >= Mr_blocks * Nr_blocks * Kr_blocks,
        "Not all partitions are assigned."
    );
    #pragma omp parallel num_threads(56)
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
                for (int64_t kc = k_block_start; kc < k_block_end; kc += Kc_blocks) {
                    int64_t k_start = kc * Kr;
                    int64_t k_end = std::min(std::min(kc + Kc_blocks, k_block_end) * Kr, K);
                    for (int64_t nci = nc; nci < nc_block_end; nci++) {
                        if (kc == k_block_start) {
                            kernel_micro_gemm<static_cast<bool>(false)>(
                                amx_state,
                                &(X[static_cast<int64_t>(k_start + 512L*m_start)]),
                                &(W[static_cast<int64_t>(16L*k_start + 8192L*nci)]),
                                &(local_acc_buf[static_cast<int64_t>(Nr*nci + ((-1L)*Nr*nc))]),
                                static_cast<int64_t>(m_end + ((-1L)*m_start)),
                                static_cast<int64_t>(Nr),
                                static_cast<int64_t>(k_end + ((-1L)*k_start)),
                                static_cast<int64_t>(512L),
                                static_cast<int64_t>(16L),
                                static_cast<int64_t>(Nc_blocks*Nr)
                            );

                        } else {
                            kernel_micro_gemm<static_cast<bool>(true)>(
                                amx_state,
                                &(X[static_cast<int64_t>(k_start + 512L*m_start)]),
                                &(W[static_cast<int64_t>(16L*k_start + 8192L*nci)]),
                                &(local_acc_buf[static_cast<int64_t>(Nr*nci + ((-1L)*Nr*nc))]),
                                static_cast<int64_t>(m_end + ((-1L)*m_start)),
                                static_cast<int64_t>(Nr),
                                static_cast<int64_t>(k_end + ((-1L)*k_start)),
                                static_cast<int64_t>(512L),
                                static_cast<int64_t>(16L),
                                static_cast<int64_t>(Nc_blocks*Nr)
                            );

                        }
                    }
                }
                {
                    {
                        #pragma GCC ivdep
                        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(m_end + ((-1L)*m_start)); x0+=static_cast<int64_t>(1L))
                        {
                            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(n_end + ((-1L)*n_start)); x1+=static_cast<int64_t>(16L))
                            {
                                {
                                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(16L*(c10::div_floor_integer(static_cast<int64_t>(n_end + ((-1L)*n_start)), static_cast<int64_t>(16L))))))
                                    {
                                        auto tmp0 = at::vec::Vectorized<float>::loadu(local_acc_buf + static_cast<int64_t>(x1 + Nc_blocks*Nr*x0), static_cast<int64_t>(16));
                                        auto tmp1 = at::vec::convert<bfloat16>(tmp0);
                                        tmp1.store(Y + static_cast<int64_t>(n_start + x1 + 1024L*m_start + 1024L*x0), static_cast<int64_t>(16));
                                    }
                                    if(C10_UNLIKELY(x1 >= static_cast<int64_t>(16L*(c10::div_floor_integer(static_cast<int64_t>(n_end + ((-1L)*n_start)), static_cast<int64_t>(16L)))) && x1 < static_cast<int64_t>(n_end + ((-1L)*n_start))))
                                    {
                                        auto tmp0 = at::vec::Vectorized<float>::loadu(local_acc_buf + static_cast<int64_t>(x1 + Nc_blocks*Nr*x0), static_cast<int64_t>(n_end + ((-1L)*n_start) + ((-16L)*(c10::div_floor_integer(static_cast<int64_t>(n_end + ((-1L)*n_start)), static_cast<int64_t>(16L))))));
                                        auto tmp1 = at::vec::convert<bfloat16>(tmp0);
                                        tmp1.store(Y + static_cast<int64_t>(n_start + x1 + 1024L*m_start + 1024L*x0), static_cast<int64_t>(n_end + ((-1L)*n_start) + ((-16L)*(c10::div_floor_integer(static_cast<int64_t>(n_end + ((-1L)*n_start)), static_cast<int64_t>(16L))))));
                                    }
                                }
                            }
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

def call(args):
    arg1_1, = args
    args.clear()
    assert_size_stride(arg1_1, (4, 512), (512, 1))
    buf0 = empty_strided_cpu((4, 1024), (1024, 1), torch.bfloat16)
    cpp_fused_mm_0(arg1_1, constant1, buf0)
    del arg1_1
    return (buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    global _frozen_param2
    _frozen_param2 = rand_strided((1024, 512), (1, 0), device='cpu', dtype=torch.bfloat16)
    global constant1
    constant1 = rand_strided((64, 512, 16), (8192, 16, 1), device='cpu', dtype=torch.bfloat16)
    arg1_1 = rand_strided((4, 512), (512, 1), device='cpu', dtype=torch.bfloat16)
    fn = lambda: call([arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)

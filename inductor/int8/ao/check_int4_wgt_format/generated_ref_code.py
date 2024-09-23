
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
_frozen_param0 = None  # device(type='cpu') torch.int32 (2, 2, 32, 4) (256, 128, 4, 1) 7f31c0a3d0d0
_frozen_param1 = None  # device(type='cpu') torch.bfloat16 (1, 16, 2) (32, 2, 1) 7f31e5cae020
qGroupSize = None  # device(type='cpu') torch.int64 () () 7f31bd71b970
constant3 = None  # device(type='cpu') torch.int32 (16, 32, 1) (32, 1, 1) 7f31bd48dcb0


cpp_fused__to_copy_0 = async_compile.cpp_pybinding(['const float*', 'bfloat16*'], '''
#include "/tmp/torchinductor_leslie/ws/cwsvzzfeuboyadztlqjgeyrffcfenmft2dgohqyyv2epni6zrmkq.h"
extern "C"  void kernel(const float* in_ptr0,
                       bfloat16* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(16L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0), 16);
            auto tmp1 = at::vec::convert<bfloat16>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0), 16);
        }
    }
}
''')


cpp_fused__to_copy__weight_int4pack_mm_1 = async_compile.cpp_pybinding(['const bfloat16*', 'const int32_t*', 'const int64_t*', 'const bfloat16*', 'float*'], '''
#include "/tmp/torchinductor_leslie/ws/cwsvzzfeuboyadztlqjgeyrffcfenmft2dgohqyyv2epni6zrmkq.h"

#include "c10/util/Unroll.h"

const bfloat16* dequant(
    const int* in_ptr,
    const bfloat16*  ScaleAndZeros,
    bfloat16* out_ptr,
    long M,
    long N,
    long K){

    // bfloat16* out_ptr = std::make_unique<bfloat16[]>(12288).get();

    static constexpr float lut[16] = {
        -8.0f, -7.0f, -6.0f, -5.0f,
        -4.0f, -3.0f, -2.0f, -1.0f,
        0.0f, 1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f, 7.0f
    };

    // long K = 256;
    // long N = 16;
    int K_int = K / 8;

    int N_loop = 1;   

    std::cout<<"in_ptr 2 is: "<<in_ptr[2]<<std::endl;
    std::cout<<"in_ptr 7 is: "<<in_ptr[7]<<std::endl;
    std::cout<<"in_ptr 8 is: "<<in_ptr[8]<<std::endl;


    const unsigned char* in_ptr_cast = reinterpret_cast<const unsigned char*>(in_ptr);
    for (int n = 0; n < N_loop; n += 1) {
        for (int k = 0; k < K; k += 1) {
            int kb = k / 256;  

            const auto scale = static_cast<float>(ScaleAndZeros[0]);
            const auto zero = static_cast<float>(ScaleAndZeros[1]);

            long idx = (n * K + k) / 2;
            long offset = 1 - (n * K + k) % 2;
            unsigned char val = in_ptr_cast[idx];
            int index = ((val & (0xF << (offset * 4))) >> (offset * 4));
            const bfloat16 b_val = static_cast<bfloat16>(lut[index] * scale + zero);
            out_ptr[n * K + k] = b_val;

            // std::cout<<"---- n is: "<<n<<" k is: "<<k<<" val is: "<<static_cast<int>(val)<<" deqaunt_val is: "<<b_val<<std::endl;
        }
    }
    return out_ptr;
}




template <bool accum>
inline void kernel_micro_gemm(
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
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            float result = accum ? C[m * ldc + n] : 0;
            for (int64_t k = 0; k < K; ++k) {
                // std::cout<<"---- m: "<<m<<", n: "<<n<<", k: "<<k<<std::endl;
                // std::cout<<"---- A[m * lda + k]: "<<A[m * lda + k]<<std::endl;
                // std::cout<<"---- B[k * ldb + n]: "<<B[k * ldb + n]<<std::endl;

                result += (float)A[m * lda + k] * (float)B[k * ldb + n] * 1;

                // std::cout<<"---- result: "<<result<<std::endl;

            }
            C[m * ldc + n] = result;
        }
    }
}

extern "C" 
void kernel(const bfloat16* X, const int32_t* W, const int64_t* qGroupSize, const bfloat16* ScaleZP, float* Y)
{
    std::cout<<"---- start a gemm run ----"<<std::endl;

    constexpr int64_t num_threads = 1;
    constexpr int64_t N = 16;
    constexpr int64_t K = 256;
    constexpr int64_t M0 = 1;
    constexpr int64_t N0 = 1;
    constexpr int64_t K0 = 1;
    constexpr int64_t N0_blocks = (N + N0 - 1) / N0;
    constexpr int64_t K0_blocks = (K + K0 - 1) / K0;
    constexpr int64_t M = static_cast<long>(2L);
    constexpr int64_t M0_blocks = (M + M0 - 1) / M0;
    constexpr int64_t Mt_blocks = 2;
    constexpr int64_t Nt_blocks = 16;
    constexpr int64_t Kt_blocks = 256;
    constexpr int64_t Mc_blocks = 2;
    constexpr int64_t Kc_blocks = 256;
    constexpr int64_t num_Mc_blocks = (M0_blocks + Mc_blocks - 1) / Mc_blocks;
    constexpr int64_t num_Nc_blocks = N0_blocks;
    constexpr int64_t num_k_slices = (K0_blocks + Kt_blocks - 1) / Kt_blocks;

    // make sure all partitions are assigned
    TORCH_CHECK(
        Mt_blocks * Nt_blocks * Kt_blocks * 1 >= M0_blocks * N0_blocks * K0_blocks,
        "Not all partitions are assigned."
    );
    {
        const int tid = 0;
        const int64_t m_block_start = 0;
        const int64_t m_block_end = M0_blocks;
        const int64_t n_block_start = 0;
        const int64_t n_block_end = N0_blocks;
        const int64_t k_block_start = 0;
        const int64_t k_block_end = K0_blocks;


        std::cout<<"m_block_start is: "<<m_block_start<<std::endl;
        std::cout<<"m_block_end is: "<<m_block_end<<std::endl;
        std::cout<<"Mc_blocks is: "<<Mc_blocks<<std::endl;
        std::cout<<"M0 is: "<<M0<<std::endl;


        std::cout<<"n_block_end is: "<<n_block_end<<std::endl;
        std::cout<<"n_block_start is: "<<n_block_start<<std::endl;
        // std::cout<<"micro_gemm.register_blocking.block_n is: "<<micro_gemm.register_blocking.block_n<<std::endl;
        std::cout<<"N0 is: "<<N0<<std::endl;


        std::cout<<"k_block_end is: "<<k_block_end<<std::endl;
        std::cout<<"k_block_start is: "<<k_block_start<<std::endl;
        std::cout<<"Kc_blocks is: "<<Kc_blocks<<std::endl;
        std::cout<<"K0 is: "<<K0<<std::endl;


        for (int64_t mc = m_block_start; mc < m_block_end; mc += Mc_blocks) {
            const int64_t m_start = mc * M0;
            const int64_t m_end = std::min(std::min(mc + Mc_blocks, m_block_end) * M0, M);
            const int64_t m_size = m_end - m_start;
            auto _local_acc_buf = std::make_unique<float[]>(static_cast<long>((N0*m_end) + ((-1L)*N0*m_start))); auto local_acc_buf = _local_acc_buf.get();
            for (int64_t nc = n_block_start; nc < n_block_end; ++nc) {
                const int64_t n_start = nc * N0;
                const int64_t n_end = std::min((nc + 1) * N0, N);
                const int64_t n_size = n_end - n_start;
                if (_local_acc_buf == nullptr) { _local_acc_buf = std::make_unique<float[]>(static_cast<long>((N0*m_end) + ((-1L)*N0*m_start))); local_acc_buf = _local_acc_buf.get(); }
                for (int64_t kc = k_block_start; kc < k_block_end; kc += Kc_blocks) {
                    int64_t k_start = kc * K0;
                    int64_t k_end = std::min(std::min(kc + Kc_blocks, k_block_end) * K0, K);
                    auto _int4_dequant_buf = std::make_unique<bfloat16[]>(static_cast<long>(k_end + ((-1L)*k_start))); auto int4_dequant_buf = _int4_dequant_buf.get();

                    std::cout<<"run micro gemm n_start is: "<<n_start<<std::endl;

                    dequant(&(W[static_cast<long>(k_start + (32L*nc))]), &(ScaleZP[static_cast<long>((2L*nc) + (32L*k_start))]), &(int4_dequant_buf[static_cast<long>(0L)]), static_cast<long>(2L), static_cast<long>(16L), static_cast<long>(256L)); // hhh
                    if (kc == k_block_start) {
                        kernel_micro_gemm<static_cast<bool>(false)>(
                            &(X[static_cast<long>(k_start + (256L*m_start))]),
                            &(int4_dequant_buf[static_cast<long>(0L)]),
                            &(local_acc_buf[static_cast<long>(0L)]),
                            static_cast<long>(m_end + ((-1L)*m_start)),
                            static_cast<long>(N0),
                            static_cast<long>(k_end + ((-1L)*k_start)),
                            static_cast<long>(256L),
                            static_cast<long>(1L),
                            static_cast<long>(N0)
                        );

                    } else {
                        kernel_micro_gemm<static_cast<bool>(true)>(
                            &(X[static_cast<long>(k_start + (256L*m_start))]),
                            &(int4_dequant_buf[static_cast<long>(0L)]),
                            &(local_acc_buf[static_cast<long>(0L)]),
                            static_cast<long>(m_end + ((-1L)*m_start)),
                            static_cast<long>(N0),
                            static_cast<long>(k_end + ((-1L)*k_start)),
                            static_cast<long>(256L),
                            static_cast<long>(1L),
                            static_cast<long>(N0)
                        );

                    }

                }
                {
                    {
                        #pragma GCC ivdep
                        for(long x0=static_cast<long>(0L); x0<static_cast<long>(m_end + ((-1L)*m_start)); x0+=static_cast<long>(1L))
                        {
                            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L*(c10::div_floor_integer(N0, 16L))); x1+=static_cast<long>(16L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(local_acc_buf + static_cast<long>(x1 + (N0*x0)), 16);
                                auto tmp1 = (tmp0);
                                tmp1.store(Y + static_cast<long>(n_start + x1 + (16L*m_start) + (16L*x0)));
                            }
                            #pragma omp simd simdlen(8) 
                            for(long x1=static_cast<long>(16L*(c10::div_floor_integer(N0, 16L))); x1<static_cast<long>(N0); x1+=static_cast<long>(1L))
                            {
                                auto tmp0 = local_acc_buf[static_cast<long>(x1 + (N0*x0))];
                                auto tmp1 = c10::convert<float>(tmp0);
                                Y[static_cast<long>(n_start + x1 + (16L*m_start) + (16L*x0))] = tmp1;
                            }
                        }
                    }

                }
            }
        }

    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg2_1, = args
    args.clear()
    assert_size_stride(arg2_1, (2, 256), (256, 1))
    buf0 = empty_strided_cpu((2, 256), (256, 1), torch.bfloat16)
    cpp_fused__to_copy_0(arg2_1, buf0)
    del arg2_1
    buf2 = empty_strided_cpu((2, 16), (16, 1), torch.float32)
    cpp_fused__to_copy__weight_int4pack_mm_1(buf0, constant3, qGroupSize, _frozen_param1, buf2)
    return (buf2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    global _frozen_param0
    _frozen_param0 = rand_strided((2, 2, 32, 4), (256, 128, 4, 1), device='cpu', dtype=torch.int32)
    global _frozen_param1
    _frozen_param1 = rand_strided((1, 16, 2), (32, 2, 1), device='cpu', dtype=torch.bfloat16)
    global qGroupSize
    qGroupSize = rand_strided((), (), device='cpu', dtype=torch.int64)
    global constant3
    constant3 = rand_strided((16, 32, 1), (32, 1, 1), device='cpu', dtype=torch.int32)
    arg2_1 = rand_strided((2, 256), (256, 1), device='cpu', dtype=torch.float32)
    fn = lambda: call([arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)

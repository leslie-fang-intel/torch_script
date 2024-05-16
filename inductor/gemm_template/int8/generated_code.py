
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

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()
_frozen_param0 = None  # device(type='cpu') torch.float32 (64,) (1,) 7facaaad18f0
_frozen_param1 = None  # device(type='cpu') torch.int64 (64,) (1,) 7facaaad1e40
_frozen_param3 = None  # device(type='cpu') torch.int8 (128, 64) (1, 0) 7faca4eb2c00
x_scale = None  # device(type='cpu') torch.float32 () () 7faca4d685e0
x_zp = None  # device(type='cpu') torch.int32 () () 7faca5020bd0
_frozen_param1_0 = None  # device(type='cpu') torch.int32 (64,) (1,) 7faca545dd50
constant6 = None  # device(type='cpu') torch.int8 (128, 64, 1) (64, 1, 1) 7faca4bce020


local_seed = 2024
torch.manual_seed(local_seed) # Set PyTorch seed
np.random.seed(seed=local_seed) # Set Numpy seed
random.seed(local_seed) # Set the Python seed

cpp_fused_quantize_per_tensor_0 = async_compile.cpp_pybinding(['const float*', 'unsigned char*'], '''
#include "/localdisk/leslie/torch_inductor_community/pytorch/torch/_inductor/codegen/cpp_prefix.h"
extern "C" void kernel(const float* in_ptr0,
                       unsigned char* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(16L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0), 16);
            auto tmp1 = static_cast<float>(33.556986525599264);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp4 = tmp3.round();
            auto tmp5 = static_cast<float>(132.0);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 + tmp6;
            auto tmp8 = static_cast<float>(0.0);
            auto tmp9 = at::vec::Vectorized<float>(tmp8);
            auto tmp10 = at::vec::maximum(tmp7, tmp9);
            auto tmp11 = static_cast<float>(255.0);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = at::vec::minimum(tmp10, tmp12);
            auto tmp14 = at::vec::convert<unsigned char>(tmp13);
            tmp14.store(out_ptr0 + static_cast<long>(x0), 16);
        }
    }
}
''')


cpp_fused_mm_quantize_per_tensor_1 = async_compile.cpp_pybinding(['const unsigned char*', 'const float*', 'const int*', 'const signed char*', 'const float*', 'const int*', 'float*'], '''
#include "/localdisk/leslie/torch_inductor_community/pytorch/torch/_inductor/codegen/cpp_prefix.h"

#include "c10/util/Unroll.h"




template <bool accum>
inline void kernel_micro_gemm(
    const unsigned char* __restrict__ A,
    const signed char* __restrict__ B,
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
            int result = accum ? C[m * ldc + n] : 0;
            for (int64_t k = 0; k < K; ++k) {
                result += (int)A[m * lda + k] * (int)B[k * ldb + n] * 1;
            }
            C[m * ldc + n] = result;
        }
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

extern "C"
void kernel(const unsigned char* X, const float* x_scale, const int* x_zp, const signed char* W, const float* w_scale, const int* w_zp, float* Y)
{

    constexpr int64_t num_threads = 1;
    constexpr int64_t N = static_cast<long>(64L);
    constexpr int64_t K = static_cast<long>(128L);
    constexpr int64_t M0 = 1;
    constexpr int64_t N0 = 1;
    constexpr int64_t K0 = 1;
    constexpr int64_t N0_blocks = (N + N0 - 1) / N0;
    constexpr int64_t K0_blocks = (K + K0 - 1) / K0;

    static_assert(N % N0 == 0, "N dimension must be multiple of N0");

    float b_compensate[N];                               
    _sum_b_contiguous_kernel<signed char>(
        W,
        b_compensate,
        K,
        N,
        N);

    // TODO(jgong5): improve cache blocking with CPU info (Mc, Kc)
    constexpr int64_t M = static_cast<long>(32L);
    constexpr int64_t M0_blocks = (M + M0 - 1) / M0;
    constexpr int64_t Mt_blocks = 32;
    constexpr int64_t Nt_blocks = 64;
    constexpr int64_t Kt_blocks = 128;
    constexpr int64_t Mc_blocks = 32;
    constexpr int64_t Kc_blocks = 128;

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

        std::cout<<"---- m_block_start is: "<<m_block_start<<std::endl;
        std::cout<<"---- m_block_end is: "<<m_block_end<<std::endl;
        std::cout<<"---- Mc_blocks is: "<<Mc_blocks<<std::endl;

        std::cout<<"---- n_block_start is: "<<n_block_start<<std::endl;
        std::cout<<"---- n_block_end is: "<<n_block_end<<std::endl;

        std::cout<<"---- k_block_start is: "<<k_block_start<<std::endl;
        std::cout<<"---- k_block_end is: "<<k_block_end<<std::endl;
        std::cout<<"---- Kc_blocks is: "<<Kc_blocks<<std::endl;

        for (int64_t mc = m_block_start; mc < m_block_end; mc += Mc_blocks) {
            const int64_t m_start = mc * M0;
            const int64_t m_end = std::min((mc + Mc_blocks) * M0, M);
            std::cout<<"m_start is: "<<m_start<<std::endl;
            std::cout<<"m_end is: "<<m_end<<std::endl;
            const int64_t m_size = m_end - m_start;
            for (int64_t nc = n_block_start; nc < n_block_end; ++nc) {
                const int64_t n_start = nc * N0;
                const int64_t n_size = N0;
                for (int64_t kc = k_block_start; kc < k_block_end; kc += Kc_blocks) {
                    int64_t k_start = kc * K0;
                    int64_t k_end = std::min((kc + Kc_blocks) * K0, K);

                    // std::cout<<"k_start is: "<<k_start<<std::endl;
                    // std::cout<<"k_end is: "<<k_end<<std::endl;
                                                                 
                    if (kc == k_block_start) {
                        kernel_micro_gemm<static_cast<bool>(false)>(
                            &(X[static_cast<long>(k_start + (128L*m_start))]),
                            // &(W[static_cast<long>(k_start + (64L*nc))]),
                            &(W[static_cast<long>(n_start + (64L*kc))]),
                            &(Y[static_cast<long>(n_start + (64L*m_start))]),
                            static_cast<long>(m_end + ((-1L)*m_start)),
                            static_cast<long>(N0),
                            static_cast<long>(k_end + ((-1L)*k_start)),
                            static_cast<long>(128L),
                            static_cast<long>(64L),
                            static_cast<long>(64L)
                        );

                    } else {
                        std::cout<<"----- shouldn't hit this branch ----"<<std::endl;
                        kernel_micro_gemm<static_cast<bool>(true)>(
                            &(X[static_cast<long>(k_start + (128L*m_start))]),
                            &(W[static_cast<long>(k_start + (64L*nc))]),
                            &(Y[static_cast<long>(n_start + (64L*m_start))]),
                            static_cast<long>(m_end + ((-1L)*m_start)),
                            static_cast<long>(N0),
                            static_cast<long>(k_end + ((-1L)*k_start)),
                            static_cast<long>(128L),
                            static_cast<long>(1L),
                            static_cast<long>(64L)
                        );

                    }
                }
                // compensation s32 to f32
                // std::cout<<"---- kernel template is int8_gemm ----"<<std::endl;

            }
        }
    }
    // for (int m=0; m<32;m++) {
    //    for (int n=0;n<64;n++) {
    //        float cur_y = Y[m*64 + n];

    //        if (m == 0 && n == 0) {
    //            std::cout<<"---- m is: "<<m<<" n is: "<<n<<" cur_y is: "<<cur_y<<std::endl;
    //        }

    //        cur_y = x_scale[0] * w_scale[n] * cur_y;
    //        cur_y -= x_scale[0] * w_scale[n] * x_zp[0] * b_compensate[n];
    //        Y[m*64 + n] = cur_y;  
    //    }
    //}
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg3_1, = args
    args.clear()
    assert_size_stride(arg3_1, (32, 128), (128, 1))
    buf0 = empty_strided_cpu((32, 128), (128, 1), torch.uint8)
    cpp_fused_quantize_per_tensor_0(arg3_1, buf0)
    del arg3_1
    buf2 = empty_strided_cpu((32, 64), (64, 1), torch.float32)

    print("buf0 is: {}".format(buf0), flush=True)
    print("constant6 is: {}".format(constant6), flush=True)


    ref_res = torch.matmul(buf0.to(torch.int32), constant6.squeeze(2).to(torch.int32))
    print("---- Eager ref_res is: {}".format(ref_res), flush=True)

    cpp_fused_mm_quantize_per_tensor_1(buf0, x_scale, x_zp, constant6, _frozen_param0, _frozen_param1_0, buf2)
    print("---- Inductor buf2 is: {}".format(buf2), flush=True)

    print(torch.allclose(ref_res.to(torch.float), buf2, atol=1e-3, rtol=1e-3), flush=True)

    return (buf2, )


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
    global constant6

    weight_f = rand_strided((128, 64, 1), (64, 1, 1), device='cpu', dtype=torch.float32)
    constant6 = weight_f.to(torch.int8)
    arg3_1 = rand_strided((32, 128), (128, 1), device='cpu', dtype=torch.float32)
    fn = lambda: call([arg3_1])
    # return print_performance(fn, times=times, repeat=repeat)

    res = fn()

    return 0

if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)

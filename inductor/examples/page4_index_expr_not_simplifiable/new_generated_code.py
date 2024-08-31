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
_frozen_param3 = None  # device(type='cpu') torch.float32 (1536,) (1,) 7f665d925210
_frozen_param6 = None  # device(type='cpu') torch.float32 (512, 1, 1) (1, 1, 1) 7f6618515c10
_frozen_param7 = None  # device(type='cpu') torch.float32 (512, 1, 1) (1, 1, 1) 7f66185167a0
_frozen_param8 = None  # device(type='cpu') torch.float32 (512, 1, 1) (1, 1, 1) 7f6618515940
_frozen_param9 = None  # device(type='cpu') torch.float32 (512, 1, 1) (1, 1, 1) 7f66185162f0
_frozen_param10 = None  # device(type='cpu') torch.float32 (1536, 512, 1, 1) (1, 0, 0, 0) 7f6618541120


cpp_fused__native_batch_norm_legit_no_training_silu_0 = async_compile.cpp_pybinding(['float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*'], '''
#include "/tmp/torchinductor_leslie/z4/cz4j2mmotlx3z2b7u4fbjtdt4x6plhd67ljwzg5bk7ekv4xz6y7q.h"
extern "C"  void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(56)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(64L); x0+=static_cast<int64_t>(1L))
            {
                #pragma GCC ivdep
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(64L); x1+=static_cast<int64_t>(1L))
                {
                    for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(512L); x2+=static_cast<int64_t>(16L))
                    {
                        auto tmp0 =
                        [&]
                        {
                            __at_align__ std::array<float, 16> tmpbuf;
                            #pragma GCC unroll 16
                            for (long x2_inner = 0; x2_inner < 16; x2_inner++)
                            {
                                tmpbuf[x2_inner] = in_ptr0[static_cast<int64_t>((128L*x1) + (8192L*(c10::div_floor_integer(static_cast<int64_t>((x1 + (64L*x2) + (64L*x2_inner))), static_cast<int64_t>(8192L)))) + (32768L*x0) + (static_cast<int64_t>((x2 + x2_inner)) % static_cast<int64_t>(128L)))];
                            }
                            return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 16);
                        }
                        ()
                        ;
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x2), 16);
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x2), 16);
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x2), 16);
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<int64_t>(x2), 16);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 + tmp7;
                        auto tmp9 = decltype(tmp8)(1)/(decltype(tmp8)(1) + tmp8.neg().exp());
                        auto tmp10 = tmp8 * tmp9;
                        tmp10.store(in_out_ptr0 + static_cast<int64_t>(x2 + (512L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_1 = async_compile.cpp_pybinding(['const float*', 'float*'], '''
#include "/tmp/torchinductor_leslie/z4/cz4j2mmotlx3z2b7u4fbjtdt4x6plhd67ljwzg5bk7ekv4xz6y7q.h"
extern "C"  void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(56)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(64L); x0+=static_cast<int64_t>(1L))
            {
                #pragma GCC ivdep
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(1536L); x1+=static_cast<int64_t>(16L))
                {
                    #pragma GCC ivdep
                    for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(64L); x2+=static_cast<int64_t>(16L))
                    {
                        alignas(16) float tmp0[16*16];
                        at::vec::transpose_mxn<float,16,16>(in_ptr0 + static_cast<int64_t>(x1 + (1536L*x2) + (98304L*x0)), static_cast<int64_t>(1536L), tmp0, 16);
                        for (long x1_inner = 0; x1_inner < 16; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<int64_t>(16L*x1_inner), 16);
                            tmp1.store(out_ptr0 + static_cast<int64_t>(x2 + (64L*x1) + (64L*x1_inner) + (98304L*x0)));
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
    arg6_1, = args
    args.clear()
    assert_size_stride(arg6_1, (256, 64, 128), (8192, 128, 1))
    buf0 = empty_strided_cpu((64, 512, 8, 8), (32768, 1, 4096, 512), torch.float32)
    buf1 = buf0; del buf0  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_0(buf1, arg6_1, _frozen_param6, _frozen_param7, _frozen_param8, _frozen_param9)
    del arg6_1
    buf2 = torch.ops.mkldnn._convolution_pointwise.default(buf1, _frozen_param10, _frozen_param3, [0, 0], [1, 1], [1, 1], 1, 'none', [None], '')
    assert_size_stride(buf2, (64, 1536, 8, 8), (98304, 1, 12288, 1536))
    del buf1
    buf3 = empty_strided_cpu((64, 1536, 8, 8), (98304, 64, 8, 1), torch.float32)
    cpp_fused_1(buf2, buf3)
    return (buf3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    global _frozen_param3
    _frozen_param3 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    global _frozen_param6
    _frozen_param6 = rand_strided((512, 1, 1), (1, 1, 1), device='cpu', dtype=torch.float32)
    global _frozen_param7
    _frozen_param7 = rand_strided((512, 1, 1), (1, 1, 1), device='cpu', dtype=torch.float32)
    global _frozen_param8
    _frozen_param8 = rand_strided((512, 1, 1), (1, 1, 1), device='cpu', dtype=torch.float32)
    global _frozen_param9
    _frozen_param9 = rand_strided((512, 1, 1), (1, 1, 1), device='cpu', dtype=torch.float32)
    global _frozen_param10
    _frozen_param10 = rand_strided((1536, 512, 1, 1), (1, 0, 0, 0), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((256, 64, 128), (8192, 128, 1), device='cpu', dtype=torch.float32)
    fn = lambda: call([arg6_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)

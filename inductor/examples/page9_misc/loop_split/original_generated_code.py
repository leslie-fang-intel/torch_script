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


cpp_fused_native_group_norm_0 = async_compile.cpp_pybinding(['const bfloat16*', 'const float*', 'const float*', 'float*', 'float*', 'bfloat16*'], '''
#include "/tmp/torchinductor_leslie/z4/cz4j2mmotlx3z2b7u4fbjtdt4x6plhd67ljwzg5bk7ekv4xz6y7q.h"
extern "C"  void kernel(const bfloat16* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       bfloat16* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(2L); x0+=static_cast<int64_t>(1L))
        {
            #pragma GCC ivdep
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(32L); x1+=static_cast<int64_t>(1L))
            {
                {
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    Welford<at::vec::Vectorized<float>> masked_tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    static WeightRecp<at::vec::Vectorized<float>> wrecps0(static_cast<int64_t>(17280L));
                    for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(9216L); x2+=static_cast<int64_t>(1L))
                    {
                        for(int64_t x3=static_cast<int64_t>(0L); x3<static_cast<int64_t>(16L); x3+=static_cast<int64_t>(16L))
                        {
                            auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr0 + static_cast<int64_t>(x3 + (30L*x1) + (960L*x2) + (8847360L*x0)), 16);
                            auto tmp1 = at::vec::convert<float>(tmp0);
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1, &wrecps0);
                        }
                        for(int64_t x3=static_cast<int64_t>(16L); x3<static_cast<int64_t>(30L); x3+=static_cast<int64_t>(14L))
                        {
                            auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr0 + static_cast<int64_t>(x3 + (30L*x1) + (960L*x2) + (8847360L*x0)), 14);
                            auto tmp1 = at::vec::convert<float>(tmp0);
                            masked_tmp_acc0_vec = welford_combine(masked_tmp_acc0_vec, tmp1, 14, &wrecps0);
                        }
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(masked_tmp_acc0_vec));
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<int64_t>(x1 + (32L*x0))] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<int64_t>(x1 + (32L*x0))] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(2L); x0+=static_cast<int64_t>(1L))
        {
            #pragma GCC ivdep
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(9216L); x1+=static_cast<int64_t>(1L))
            {
                for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(960L); x2+=static_cast<int64_t>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr0 + static_cast<int64_t>(x2 + (960L*x1) + (8847360L*x0)), 16);
                    auto tmp2 =
                    [&]
                    {
                        __at_align__ std::array<float, 16> tmpbuf;
                        #pragma GCC unroll 16
                        for (long x2_inner = 0; x2_inner < 16; x2_inner++)
                        {
                            tmpbuf[x2_inner] = out_ptr0[static_cast<int64_t>((32L*x0) + (c10::div_floor_integer(static_cast<int64_t>((x2 + x2_inner)), static_cast<int64_t>(30L))))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 16);
                    }
                    ()
                    ;
                    auto tmp4 =
                    [&]
                    {
                        __at_align__ std::array<float, 16> tmpbuf;
                        #pragma GCC unroll 16
                        for (long x2_inner = 0; x2_inner < 16; x2_inner++)
                        {
                            tmpbuf[x2_inner] = out_ptr1[static_cast<int64_t>((32L*x0) + (c10::div_floor_integer(static_cast<int64_t>((x2 + x2_inner)), static_cast<int64_t>(30L))))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 16);
                    }
                    ()
                    ;
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x2), 16);
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x2), 16);
                    auto tmp1 = at::vec::convert<float>(tmp0);
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(276480.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::convert<bfloat16>(tmp16);
                    tmp17.store(out_ptr2 + static_cast<int64_t>(x2 + (960L*x1) + (8847360L*x0)), 16);
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (960, ), (1, ))
    assert_size_stride(arg1_1, (960, ), (1, ))
    assert_size_stride(arg2_1, (2, 960, 96, 96), (8847360, 1, 92160, 960))
    buf0 = empty_strided_cpu((2, 32, 1, 1), (32, 1, 64, 64), torch.float32)
    buf1 = empty_strided_cpu((2, 32, 1, 1), (32, 1, 64, 64), torch.float32)
    buf3 = empty_strided_cpu((2, 960, 96, 96), (8847360, 1, 92160, 960), torch.bfloat16)
    cpp_fused_native_group_norm_0(arg2_1, arg0_1, arg1_1, buf0, buf1, buf3)
    del arg0_1
    del arg1_1
    del arg2_1
    return (buf3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((2, 960, 96, 96), (8847360, 1, 92160, 960), device='cpu', dtype=torch.bfloat16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)

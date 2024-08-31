
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

from torch import device, empty, empty_strided
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


cpp_fused_native_layer_norm_0 = async_compile.cpp('''
#include "/tmp/torchinductor_leslie/gb/cgbau5vlj6cetmcjbjbtw6x4rrivaln6f45s5d72gy2bfx5foz3k.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(28L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(16L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 =
                        [&]
                        {
                            __at_align__ std::array<float, 16> tmpbuf;
                            #pragma GCC unroll 16
                            for (long x1_inner = 0; x1_inner < 16; x1_inner++)
                            {
                                tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*(c10::div_floor_integer(x2, 256L))) + (256L*x1) + (256L*x1_inner) + (7168L*(static_cast<long>(c10::div_floor_integer(x2, 128L)) % static_cast<long>(2L))) + (14336L*x0) + (static_cast<long>(x2) % static_cast<long>(128L)))];
                            }
                            return at::vec::Vectorized<float>::loadu(tmpbuf.data());
                        }
                        ()
                        ;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (28L*x0)));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (28L*x0)));
                }
            }
            #pragma omp simd simdlen(8) 
            for(long x1=static_cast<long>(16L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>((128L*(c10::div_floor_integer(x2, 256L))) + (256L*x1) + (7168L*(static_cast<long>(c10::div_floor_integer(x2, 128L)) % static_cast<long>(2L))) + (14336L*x0) + (static_cast<long>(x2) % static_cast<long>(128L)))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x1 + (28L*x0))] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x1 + (28L*x0))] = tmp_acc0.m2;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(28L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(16L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 =
                    [&]
                    {
                        __at_align__ std::array<float, 16> tmpbuf;
                        #pragma GCC unroll 16
                        for (long x1_inner = 0; x1_inner < 16; x1_inner++)
                        {
                            tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*(c10::div_floor_integer(x2, 256L))) + (256L*x1) + (256L*x1_inner) + (7168L*(static_cast<long>(c10::div_floor_integer(x2, 128L)) % static_cast<long>(2L))) + (14336L*x0) + (static_cast<long>(x2) % static_cast<long>(128L)))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data());
                    }
                    ()
                    ;
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (28L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (28L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(512.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    { __at_align__ float tmpbuf[16*sizeof(float)/sizeof(float)]; tmp11.store(tmpbuf); for (long x1_inner = 0; x1_inner < 16; x1_inner++) out_ptr2[static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (14336L*x0))] = tmpbuf[x1_inner]; }
                }
            }
            #pragma omp simd simdlen(8) 
            for(long x1=static_cast<long>(16L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>((128L*(c10::div_floor_integer(x2, 256L))) + (256L*x1) + (7168L*(static_cast<long>(c10::div_floor_integer(x2, 128L)) % static_cast<long>(2L))) + (14336L*x0) + (static_cast<long>(x2) % static_cast<long>(128L)))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1 + (28L*x0))];
                    auto tmp3 = out_ptr1[static_cast<long>(x1 + (28L*x0))];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(512.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    out_ptr2[static_cast<long>(x2 + (512L*x1) + (14336L*x0))] = tmp9;
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (1, 56, 56, 128), (401408, 7168, 128, 1))
    buf0 = empty_strided((1, 28, 28, 1), (784, 28, 1, 784), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((1, 28, 28, 1), (784, 28, 1, 784), device='cpu', dtype=torch.float32)
    buf3 = empty((1, 28, 28, 512), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_0(c_void_p(arg0_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf3.data_ptr()))
    del arg0_1
    return (buf3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 56, 56, 128), (401408, 7168, 128, 1), device='cpu', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)

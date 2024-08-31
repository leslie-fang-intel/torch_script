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


cpp_fused_native_layer_norm_0 = async_compile.cpp_pybinding(['const float*', 'float*', 'float*', 'float*'], '''
#include "/tmp/torchinductor_leslie/z4/cz4j2mmotlx3z2b7u4fbjtdt4x6plhd67ljwzg5bk7ekv4xz6y7q.h"
extern "C"  void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(28L); x0+=static_cast<int64_t>(1L))
        {
            #pragma GCC ivdep
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(28L); x1+=static_cast<int64_t>(1L))
            {
                {
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    Welford<at::vec::Vectorized<float>> masked_tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    static WeightRecp<at::vec::Vectorized<float>> wrecps0(static_cast<int64_t>(32L));
                    for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(512L); x2+=static_cast<int64_t>(16L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(
                            in_ptr0
                            + static_cast<int64_t>((128L*(c10::div_floor_integer(static_cast<int64_t>(x2), static_cast<int64_t>(256L)))) + (256L*x1)
                            + (7168L*(static_cast<int64_t>(c10::div_floor_integer(static_cast<int64_t>(x2), static_cast<int64_t>(128L))) % static_cast<int64_t>(2L)))
                            + (14336L*x0) + (static_cast<int64_t>(x2) % static_cast<int64_t>(128L))), 16);
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0, &wrecps0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(masked_tmp_acc0_vec));
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<int64_t>(x1 + (28L*x0))] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<int64_t>(x1 + (28L*x0))] = static_cast<float>(tmp_acc0.m2);
                }
                for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(512L); x2+=static_cast<int64_t>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(
                        in_ptr0
                        + static_cast<int64_t>((128L*(c10::div_floor_integer(static_cast<int64_t>(x2), static_cast<int64_t>(256L)))) + (256L*x1)
                        + (7168L*(static_cast<int64_t>(c10::div_floor_integer(static_cast<int64_t>(x2), static_cast<int64_t>(128L))) % static_cast<int64_t>(2L)))
                        + (14336L*x0) + (static_cast<int64_t>(x2) % static_cast<int64_t>(128L))), 16);
                    auto tmp1 = out_ptr0[static_cast<int64_t>(x1 + (28L*x0))];
                    auto tmp4 = out_ptr1[static_cast<int64_t>(x1 + (28L*x0))];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(512.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    tmp11.store(out_ptr2 + static_cast<int64_t>(x2 + (512L*x1) + (14336L*x0)));
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
    buf0 = empty_strided_cpu((1, 28, 28, 1), (784, 28, 1, 784), torch.float32)
    buf1 = empty_strided_cpu((1, 28, 28, 1), (784, 28, 1, 784), torch.float32)
    buf3 = empty_strided_cpu((1, 28, 28, 512), (401408, 14336, 512, 1), torch.float32)
    cpp_fused_native_layer_norm_0(arg0_1, buf0, buf1, buf3)
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


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


cpp_fused_constant_pad_nd_relu_sum_0 = async_compile.cpp('''
#include "/tmp/torchinductor_leslie/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(272L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(256);
                    auto tmp2 = tmp0 < tmp1;
                    auto tmp3 = [&]
                    {
                        auto tmp4 = in_ptr0[static_cast<long>(x1 + (256L*x0))];
                        return tmp4;
                    }
                    ;
                    auto tmp5 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                    auto tmp6 = tmp5 * (tmp5>0);
                    tmp_acc0 = tmp_acc0 + tmp6;
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0;
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
    assert_size_stride(arg0_1, (512, 256), (256, 1))
    buf0 = empty((512, ), device='cpu', dtype=torch.float32)
    cpp_fused_constant_pad_nd_relu_sum_0(c_void_p(arg0_1.data_ptr()), c_void_p(buf0.data_ptr()))
    del arg0_1
    return (buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)


from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile

from torch import empty_strided, as_strided, device
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
async_compile = AsyncCompile()


kernel_cpp_0 = async_compile.cpp('''
#include "/tmp/torchinductor_root/5b/c5bcubr6yrbvnx73gevjlm24khhax3e2tzjnnvb47oxio6qm462z.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(3L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(81L); i1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(i1 + (81L*i0))];
                out_ptr0[static_cast<long>(i0 + (3L*i1))] = tmp0;
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1 = args
    args.clear()
    buf0 = empty_strided((1, 3, 9, 9), (243, 1, 27, 3), device='cpu', dtype=torch.float32)
    kernel_cpp_0(c_void_p(arg7_1.data_ptr()), c_void_p(buf0.data_ptr()))
    del arg7_1
    buf1 = torch.ops.mkldnn._convolution_pointwise(buf0, arg0_1, arg1_1, (0, 0), (2, 2), (1, 1), 1, 'none', [], '')
    assert_size_stride(buf1, (1, 9, 4, 4), (144, 1, 36, 9))
    del arg0_1
    del arg1_1
    return (buf1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((9, 3, 3, 3), (1, 0, 0, 0), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((9, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((9, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((9, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((9, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((9, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg7_1 = rand_strided((1, 3, 9, 9), (243, 81, 9, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.utils import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)

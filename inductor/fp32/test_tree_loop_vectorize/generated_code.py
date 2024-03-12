
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

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


cpp_fused_add_dot_maximum_0 = async_compile.cpp_pybinding(['const unsigned char*', 'const unsigned char*', 'const unsigned char*', 'unsigned char*', 'long*'], '''
#include "/localdisk/leslie/torch_inductor_community/pytorch/torch/_inductor/codegen/cpp_prefix.h"
#include <iostream> 
extern "C" void kernel(const unsigned char* in_ptr0,
                       const unsigned char* in_ptr1,
                       const unsigned char* in_ptr2,
                       unsigned char* out_ptr0,
                       long* out_ptr1)
{
    {
        {
            long tmp_acc0 = 0;
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33024L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr1[static_cast<long>(0L)];
                auto tmp4 = in_ptr2[static_cast<long>(0L)];
                auto tmp2 = max_propagate_nan(tmp0, tmp1);
                auto tmp3 = decltype(tmp2)(tmp2 + tmp1);
                auto tmp5 = max_propagate_nan(tmp2, tmp4);
                auto tmp6 = decltype(tmp5)(tmp5 * tmp5);
                auto tmp7 = c10::convert<long>(tmp6);
                out_ptr0[static_cast<long>(x0)] = tmp3;
                tmp_acc0 = tmp_acc0 + tmp7;
            }
            out_ptr1[static_cast<long>(0L)] = tmp_acc0;
        }
    }
}
''')


cpp_fused_dot_1 = async_compile.cpp_pybinding(['const long*', 'unsigned char*'], '''
#include "/localdisk/leslie/torch_inductor_community/pytorch/torch/_inductor/codegen/cpp_prefix.h"
extern "C" void kernel(const long* in_ptr0,
                       unsigned char* out_ptr0)
{
    {
        auto tmp0 = in_ptr0[static_cast<long>(0L)];
        auto tmp1 = c10::convert<unsigned char>(tmp0);
        out_ptr0[static_cast<long>(0L)] = tmp1;
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (43, 1, 32, 24), (768, 768, 24, 1))
    assert_size_stride(arg1_1, (1, ), (1, ))
    assert_size_stride(arg2_1, (1, 1, 1), (1, 1, 1))
    buf0 = empty_strided_cpu((43, 1, 32, 24), (768, 768, 24, 1), torch.uint8)
    buf3 = empty_strided_cpu((), (), torch.int64)
    cpp_fused_add_dot_maximum_0(arg0_1, arg1_1, arg2_1, buf0, buf3)
    del arg0_1
    del arg1_1
    # Source Nodes: [cat_1], Original ATen: [aten.cat]
    buf1 = aten.cat.default([arg2_1, arg2_1], 0)
    del arg2_1
    buf2 = buf1
    del buf1
    buf4 = empty_strided_cpu((), (), torch.uint8)
    print("buf3 is: {}".format(buf3), flush=True)
    cpp_fused_dot_1(buf3, buf4)
    return (buf0, buf2, buf4, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance

    arg0_1 = torch.ones(size=(43, 1, 32, 24), device='cpu', dtype=torch.uint8) * 6
    arg1_1 = torch.ones(size=(1,), device='cpu', dtype=torch.uint8) * 6
    arg2_1 = torch.ones(size=(1, 1, 1), device='cpu', dtype=torch.uint8) * 6

    (buf0, buf2, buf4, ) = call([arg0_1, arg1_1, arg2_1])
    print("buf0 is: {}".format(buf0), flush=True) # [43, 1, 32, 24]
    print("buf2 is: {}".format(buf2), flush=True) # [2, 1, 1]
    print("buf4 is: {}".format(buf4), flush=True) # []
    return 0.09

if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)

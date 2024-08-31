
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
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()


cpp_fused__to_copy_add_0 = async_compile.cpp_pybinding(['const bfloat16*', 'const bfloat16*', 'float*'], '''
#include "/tmp/torchinductor_leslie/vu/cvuckxaygqfovv2zu2byqhcmiejbke7mdhf2rpgpr5mlscdev2hg.h"
extern "C"  void kernel(const bfloat16* in_ptr0,
                       const bfloat16* in_ptr1,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(17694720L); x0+=static_cast<long>(32L))
        {
            auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr0 + static_cast<long>(x0), 32);
            auto tmp2 = at::vec::Vectorized<bfloat16>::loadu(in_ptr1 + static_cast<long>(x0), 32);
            auto tmp1 = at::vec::convert<float,2,bfloat16,1>(tmp0);
            auto tmp3 = at::vec::convert<float,2,bfloat16,1>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = (tmp4);
            tmp5.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (2, 960, 96, 96), (8847360, 9216, 96, 1))
    assert_size_stride(arg1_1, (2, 960, 96, 96), (8847360, 9216, 96, 1))
    buf0 = empty_strided_cpu((2, 960, 96, 96), (8847360, 9216, 96, 1), torch.float32)
    cpp_fused__to_copy_add_0(arg0_1, arg1_1, buf0)
    del arg0_1
    del arg1_1
    return (buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((2, 960, 96, 96), (8847360, 9216, 96, 1), device='cpu', dtype=torch.bfloat16)
    arg1_1 = rand_strided((2, 960, 96, 96), (8847360, 9216, 96, 1), device='cpu', dtype=torch.bfloat16)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)

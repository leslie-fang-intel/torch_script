
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


cpp_fused_0 = async_compile.cpp_pybinding(['const int64_t*', 'float*'], '''
#include "/localdisk/leslie/torch_inductor_community/pytorch/torch/_inductor/codegen/cpp_prefix.h"
extern "C"  void kernel(const int64_t* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_ptr0[static_cast<long>(0L)];
            auto tmp1 = x0;
            auto tmp2 = c10::convert<int32_t>(tmp1);
            auto tmp3 = randn_cpu(tmp0, tmp2);
            out_ptr0[static_cast<long>(x0)] = tmp3;
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    buf0 = empty_strided_cpu((1, ), (1, ), torch.int64)
    # Source Nodes: [], Original ATen: []
    aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf0)
    print("---- buf0 is: {}".format(buf0), flush=True)
    buf1 = empty_strided_cpu((64, ), (1, ), torch.float32)
    cpp_fused_0(buf0, buf1)
    print("---- buf1 is: {}".format(buf1), flush=True)
    return (buf1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    fn = lambda: call([])
    # return print_performance(fn, times=times, repeat=repeat)
    fn()
    return 0.1

if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)

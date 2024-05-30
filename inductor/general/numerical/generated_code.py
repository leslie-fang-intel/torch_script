
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

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


cpp_fused_mul_0 = async_compile.cpp_pybinding(['const float*', 'float*'], '''
#include "/localdisk/leslie/torch_inductor_community/pytorch/torch/_inductor/codegen/cpp_prefix.h"
#include <iostream>
static constexpr unsigned int DENORMALS_ZERO = 0x0040;
static constexpr unsigned int FLUSH_ZERO = 0x8000;
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        _mm_lfence();
        unsigned int csr = _mm_getcsr();
        // csr &= DENORMALS_ZERO;
        // csr &= FLUSH_ZERO;
        std::cout<<"csr is: "<<csr<<std::endl;                           
        auto tmp0 = in_ptr0[static_cast<long>(0L)];
        auto tmp1 = static_cast<float>(4.70197740328915e-38);
        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
        out_ptr0[static_cast<long>(0L)] = tmp2;
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (), ())
    buf0 = empty_strided_cpu((), (), torch.float32)
    cpp_fused_mul_0(arg0_1, buf0)
    print("---- buf0 is: {}".format(buf0), flush=True)
    del arg0_1
    return (buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    # arg0_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg0_1 = torch.tensor(0.1875)
    fn = lambda: call([arg0_1])
    
    # return print_performance(fn, times=times, repeat=repeat)
    fn()
    return 0


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)


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


cpp_fused_sub_0 = async_compile.cpp_pybinding(['const bfloat16*', 'bfloat16*', 'bfloat16*'], '''
#include "/tmp/torchinductor_root/bk/cbkbu6ez5ilmfsvvwwbmj75kyi36qlkbdc7p2pehck5chxcrqdgq.h"
extern "C" void kernel(const bfloat16* in_ptr0,
                       bfloat16* out_ptr0,
                       bfloat16* out_ptr1)
{
    #pragma omp parallel num_threads(56)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(49152L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = c10::convert<float>(tmp0);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp1);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp2 = out_ptr0[static_cast<long>(x0)];
                    auto tmp1 = c10::convert<float>(tmp0);
                    auto tmp3 = c10::convert<float>(tmp2);
                    auto tmp4 = decltype(tmp1)(tmp1 - tmp3);
                    auto tmp5 = c10::convert<bfloat16>(tmp4);
                    out_ptr1[static_cast<long>(x1 + (1024L*x0))] = tmp5;
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
    assert_size_stride(arg0_1, (4, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    buf0 = empty_strided_cpu((4, 12, 1024, 1), (12288, 1024, 1, 49152), torch.bfloat16)
    buf1 = empty_strided_cpu((4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), torch.bfloat16)
    cpp_fused_sub_0(arg0_1, buf0, buf1)
    del arg0_1
    return (buf1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cpu', dtype=torch.bfloat16)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)

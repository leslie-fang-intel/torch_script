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


cpp_fused__to_copy_gt_where_0 = async_compile.cpp_pybinding(['const float*', 'const float*', 'int32_t*'], '''
#include "/tmp/torchinductor_leslie/z4/cz4j2mmotlx3z2b7u4fbjtdt4x6plhd67ljwzg5bk7ekv4xz6y7q.h"
extern "C"  void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       int32_t* out_ptr0)
{
    #pragma omp parallel num_threads(56)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(17694720L); x0+=static_cast<int64_t>(16L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), 16);
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x0), 16);
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = at::vec::VecMask<float,1>(tmp0 > tmp2);
                auto tmp5 = decltype(tmp0)::blendv(tmp4, tmp0, tmp3.template cast<float,1>());
                auto tmp6 = at::vec::convert<int32_t>(tmp5);
                auto tmp7 = static_cast<int32_t>(0);
                auto tmp8 = at::vec::Vectorized<int32_t>(tmp7);
                auto tmp9 = at::vec::VecMask<int32_t,1>(tmp6 > tmp8);
                auto tmp10 = at::vec::convert<int32_t>(tmp0);
                auto tmp11 = decltype(tmp6)::blendv(tmp10, tmp6, tmp9.template cast<int32_t,1>());
                tmp11.store(out_ptr0 + static_cast<int64_t>(x0), 16);
            }
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
    buf0 = empty_strided_cpu((2, 960, 96, 96), (8847360, 9216, 96, 1), torch.int32)
    cpp_fused__to_copy_gt_where_0(arg0_1, arg1_1, buf0)
    del arg0_1
    del arg1_1
    return (buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((2, 960, 96, 96), (8847360, 9216, 96, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((2, 960, 96, 96), (8847360, 9216, 96, 1), device='cpu', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)

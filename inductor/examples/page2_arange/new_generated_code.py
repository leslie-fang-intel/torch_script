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


cpp_fused_constant_pad_nd_relu_sum_0 = async_compile.cpp_pybinding(['const float*', 'float*'], '''
#include "/tmp/torchinductor_leslie/z4/cz4j2mmotlx3z2b7u4fbjtdt4x6plhd67ljwzg5bk7ekv4xz6y7q.h"
extern "C"  void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(512L); x0+=static_cast<int64_t>(1L))
        {
            {
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(272L); x1+=static_cast<int64_t>(16L))
                {
                    auto tmp0 = x1;
                    auto tmp1 = c10::convert<int64_t>(tmp0);
                    auto tmp2 = at::vec::VectorizedN<int64_t,2>::arange(tmp1, 1);
                    auto tmp3 = static_cast<int64_t>(256);
                    auto tmp4 = at::vec::VectorizedN<int64_t,2>(tmp3);
                    auto tmp5 = at::vec::VecMask<int64_t,2>(tmp2 < tmp4);
                    auto tmp6 = [&]
                    {
                        auto tmp7 = tmp5.template cast<float,1>().template loadu<float,1>(in_ptr0 + static_cast<int64_t>(x1 + (256L*x0)));
                        return tmp7;
                    }
                    ;
                    auto tmp10 =
                    [&]
                    {
                        if (tmp5.all_zero())
                        {
                            return at::vec::Vectorized<float>(static_cast<float>(0.0));
                        }
                        else
                        {
                            auto tmp8 = tmp6();
                            auto tmp9 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                            return decltype(tmp8)::blendv(tmp9, tmp8, tmp5.template cast<float,1>());
                        }
                    }
                    ()
                    ;
                    auto tmp11 = at::vec::clamp_min(tmp10, decltype(tmp10)(0));
                    tmp_acc0_vec = tmp_acc0_vec + tmp11;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0);
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
    buf0 = empty_strided_cpu((512, ), (1, ), torch.float32)
    cpp_fused_constant_pad_nd_relu_sum_0(arg0_1, buf0)
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

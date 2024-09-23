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


cpp_fused_native_layer_norm_0 = async_compile.cpp_pybinding(['const bfloat16*', 'float*', 'float*', 'bfloat16*'], '''
#include "/tmp/torchinductor_leslie/z4/cz4j2mmotlx3z2b7u4fbjtdt4x6plhd67ljwzg5bk7ekv4xz6y7q.h"
extern "C"  void kernel(const bfloat16* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       bfloat16* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(21504L); x0+=static_cast<int64_t>(1L))
        {
            {
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                Welford<at::vec::Vectorized<float>> masked_tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                static WeightRecp<at::vec::Vectorized<float>> wrecps0(static_cast<int64_t>(64L));
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(1024L); x1+=static_cast<int64_t>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr0 + static_cast<int64_t>(x1 + (1039L*x0)), 16);
                    auto tmp1 = at::vec::convert<float>(tmp0);
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1, &wrecps0);
                }
                for(int64_t x1=static_cast<int64_t>(1024L); x1<static_cast<int64_t>(1039L); x1+=static_cast<int64_t>(15L))
                {
                    auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr0 + static_cast<int64_t>(x1 + (1039L*x0)), 15);
                    auto tmp1 = at::vec::convert<float>(tmp0);
                    masked_tmp_acc0_vec = welford_combine(masked_tmp_acc0_vec, tmp1, 15, &wrecps0);
                }
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(masked_tmp_acc0_vec));
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr0[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr1[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(1024L); x1+=static_cast<int64_t>(16L))
            {
                auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr0 + static_cast<int64_t>(x1 + (1039L*x0)), 16);
                auto tmp2 = out_ptr0[static_cast<int64_t>(x0)];
                auto tmp5 = out_ptr1[static_cast<int64_t>(x0)];
                auto tmp1 = at::vec::convert<float>(tmp0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 - tmp3;
                auto tmp6 = static_cast<float>(1039.0);
                auto tmp7 = tmp5 / tmp6;
                auto tmp8 = static_cast<float>(1e-05);
                auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                auto tmp10 = 1 / std::sqrt(tmp9);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp4 * tmp11;
                auto tmp13 = at::vec::convert<bfloat16>(tmp12);
                tmp13.store(out_ptr2 + static_cast<int64_t>(x1 + (1039L*x0)), 16);
            }
            for(int64_t x1=static_cast<int64_t>(1024L); x1<static_cast<int64_t>(1039L); x1+=static_cast<int64_t>(15L))
            {
                auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr0 + static_cast<int64_t>(x1 + (1039L*x0)), 15);
                auto tmp2 = out_ptr0[static_cast<int64_t>(x0)];
                auto tmp5 = out_ptr1[static_cast<int64_t>(x0)];
                auto tmp1 = at::vec::convert<float>(tmp0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 - tmp3;
                auto tmp6 = static_cast<float>(1039.0);
                auto tmp7 = tmp5 / tmp6;
                auto tmp8 = static_cast<float>(1e-05);
                auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                auto tmp10 = 1 / std::sqrt(tmp9);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp4 * tmp11;
                auto tmp13 = at::vec::convert<bfloat16>(tmp12);
                tmp13.store(out_ptr2 + static_cast<int64_t>(x1 + (1039L*x0)), 15);
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
    assert_size_stride(arg0_1, (56, 384, 1039), (398976, 1039, 1))
    buf0 = empty_strided_cpu((56, 384, 1), (384, 1, 21504), torch.float32)
    buf1 = empty_strided_cpu((56, 384, 1), (384, 1, 21504), torch.float32)
    buf3 = empty_strided_cpu((56, 384, 1039), (398976, 1039, 1), torch.bfloat16)
    cpp_fused_native_layer_norm_0(arg0_1, buf0, buf1, buf3)
    del arg0_1
    return (buf3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((56, 384, 1039), (398976, 1039, 1), device='cpu', dtype=torch.bfloat16)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)

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


cpp_fused_add_embedding_native_layer_norm_0 = async_compile.cpp_pybinding(['const int64_t*', 'const float*', 'const float*', 'float*', 'float*', 'float*'], '''
#include "/tmp/torchinductor_leslie/z4/cz4j2mmotlx3z2b7u4fbjtdt4x6plhd67ljwzg5bk7ekv4xz6y7q.h"
extern "C"  void kernel(const int64_t* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        {
            Welford<float> tmp_acc0 = Welford<float>();
            Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
            Welford<at::vec::Vectorized<float>> masked_tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
            static WeightRecp<at::vec::Vectorized<float>> wrecps0(static_cast<int64_t>(65536L));
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(128L); x0+=static_cast<int64_t>(1L))
            {
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(8192L); x1+=static_cast<int64_t>(16L))
                {
                    auto tmp0 = in_ptr0[static_cast<int64_t>(x0)];
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1 + (8192L*x0)), 16);
                    auto tmp1 = 64L;
                    auto tmp2 = c10::convert<int64_t>(tmp1);
                    auto tmp3 = decltype(tmp0)(tmp0 + tmp2);
                    auto tmp4 = tmp0 < 0;
                    auto tmp5 = tmp4 ? tmp3 : tmp0;
                    auto tmp6 = tmp5;
                    auto tmp7 = c10::convert<int64_t>(tmp6);
                    TORCH_CHECK((0 <= tmp7) & (tmp7 < 64L), "index out of bounds: 0 <= tmp7 < 64L");
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + (8192L*tmp5)), 16);
                    auto tmp11 = tmp9 + tmp10;
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp11, &wrecps0);
                }
            }
            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(masked_tmp_acc0_vec));
            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
            out_ptr0[static_cast<int64_t>(0L)] = static_cast<float>(tmp_acc0.mean);
            out_ptr1[static_cast<int64_t>(0L)] = static_cast<float>(tmp_acc0.m2);
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(128L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(8192L); x1+=static_cast<int64_t>(16L))
            {
                auto tmp0 = in_ptr0[static_cast<int64_t>(x0)];
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1 + (8192L*x0)), 16);
                auto tmp12 = out_ptr0[static_cast<int64_t>(0L)];
                auto tmp15 = out_ptr1[static_cast<int64_t>(0L)];
                auto tmp1 = 64L;
                auto tmp2 = c10::convert<int64_t>(tmp1);
                auto tmp3 = decltype(tmp0)(tmp0 + tmp2);
                auto tmp4 = tmp0 < 0;
                auto tmp5 = tmp4 ? tmp3 : tmp0;
                auto tmp6 = tmp5;
                auto tmp7 = c10::convert<int64_t>(tmp6);
                TORCH_CHECK((0 <= tmp7) & (tmp7 < 64L), "index out of bounds: 0 <= tmp7 < 64L");
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + (8192L*tmp5)), 16);
                auto tmp11 = tmp9 + tmp10;
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 - tmp13;
                auto tmp16 = static_cast<float>(1048576.0);
                auto tmp17 = tmp15 / tmp16;
                auto tmp18 = static_cast<float>(1e-05);
                auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                auto tmp20 = 1 / std::sqrt(tmp19);
                auto tmp21 = at::vec::Vectorized<float>(tmp20);
                auto tmp22 = tmp14 * tmp21;
                tmp22.store(out_ptr2 + static_cast<int64_t>(x1 + (8192L*x0)));
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, 8192), (8192, 1))
    assert_size_stride(arg1_1, (4, 32), (32, 1))
    assert_size_stride(arg2_1, (4, 32, 8192), (262144, 8192, 1))
    buf0 = empty_strided_cpu((1, 1, 1), (1, 1, 1), torch.float32)
    buf1 = empty_strided_cpu((1, 1, 1), (1, 1, 1), torch.float32)
    buf3 = empty_strided_cpu((4, 32, 8192), (262144, 8192, 1), torch.float32)
    cpp_fused_add_embedding_native_layer_norm_0(arg1_1, arg0_1, arg2_1, buf0, buf1, buf3)
    del arg0_1
    del arg1_1
    del arg2_1
    return (buf3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((4, 32), (32, 1), device='cpu', dtype=torch.int64)
    arg2_1 = rand_strided((4, 32, 8192), (262144, 8192, 1), device='cpu', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)


# AOT ID: ['8_inference']
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

cpp_fused__softmax_0 = async_compile.cpp_pybinding(['int64_t*', 'bfloat16*', 'bfloat16*', 'bfloat16*'], '''
#include "/localdisk/leslie/torch_inductor_community/pytorch/torch/_inductor/codegen/cpp_prefix.h"
extern "C" void kernel(int64_t* out_ptr12,
                       bfloat16* in_ptr13,
                       bfloat16* out_ptr19,
                       bfloat16* out_ptr20)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr12[static_cast<long>((33L*x0) + (33L*(c10::div_floor_integer((x2 + (64L*x1)), 135168L))) + (c10::div_floor_integer((x2 + (64L*x1)), 4096L)))];
                    auto tmp13 = out_ptr12[static_cast<long>(30L + (33L*x0) + (33L*(c10::div_floor_integer((122880L + x2 + (64L*x1)), 135168L))) + (c10::div_floor_integer((x2 + (64L*x1)), 4096L)))];
                    auto tmp1 = (13L*x0) + (26L*(c10::div_floor_integer((x2 + (64L*x1)), 135168L)));
                    auto tmp2 = c10::convert<int64_t>(tmp1);
                    auto tmp3 = decltype(tmp0)(tmp0 + tmp2);
                    auto tmp4 = 156L;
                    auto tmp5 = c10::convert<int64_t>(tmp4);
                    auto tmp6 = decltype(tmp3)(tmp3 + tmp5);
                    auto tmp7 = tmp3 < 0;
                    auto tmp8 = tmp7 ? tmp6 : tmp3;
                    auto tmp9 = tmp8;
                    auto tmp10 = c10::convert<int64_t>(tmp9);
                    TORCH_CHECK((0 <= tmp10) & (tmp10 < 156L), "index out of bounds: 0 <= tmp10 < 156L");
                    auto tmp12 = in_ptr13[static_cast<long>(x2 + (64L*(static_cast<long>(c10::div_floor_integer(tmp8, 13L)) % static_cast<long>(12L))) + (768L*(static_cast<long>(x1) % static_cast<long>(64L))) + (49152L*(static_cast<long>(tmp8) % static_cast<long>(13L))))];
                    auto tmp14 = (13L*x0) + (13L*(c10::div_floor_integer((30L + (c10::div_floor_integer((x2 + (64L*x1)), 4096L))), 33L))) + (13L*(c10::div_floor_integer((122880L + x2 + (64L*x1)), 135168L)));
                    auto tmp15 = c10::convert<int64_t>(tmp14);
                    auto tmp16 = decltype(tmp13)(tmp13 + tmp15);
                    auto tmp17 = decltype(tmp16)(tmp16 + tmp5);
                    auto tmp18 = tmp16 < 0;
                    auto tmp19 = tmp18 ? tmp17 : tmp16;
                    auto tmp20 = tmp19;
                    auto tmp21 = c10::convert<int64_t>(tmp20);
                    TORCH_CHECK((0 <= tmp21) & (tmp21 < 156L), "index out of bounds: 0 <= tmp21 < 156L");
                    auto tmp23 = in_ptr13[static_cast<long>(x2 + (64L*(static_cast<long>(c10::div_floor_integer(tmp19, 13L)) % static_cast<long>(12L))) + (768L*(static_cast<long>(x1) % static_cast<long>(64L))) + (49152L*(static_cast<long>(tmp19) % static_cast<long>(13L))))];
                    out_ptr19[static_cast<long>(x2 + (64L*x1) + (28672L*x0))] = tmp12;
                    out_ptr20[static_cast<long>(x2 + (64L*x1) + (28672L*x0))] = tmp23;
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
    buf0 = empty_strided_cpu((1, 12, 192, 64), (344064, 28672, 64, 1), torch.int64)
    buf1 = empty_strided_cpu((1, 12, 192, 64), (344064, 28672, 64, 1), torch.bfloat16)
    buf2 = empty_strided_cpu((1, 12, 192, 64), (344064, 28672, 64, 1), torch.bfloat16)
    buf3 = empty_strided_cpu((1, 12, 192, 64), (344064, 28672, 64, 1), torch.bfloat16)
    cpp_fused__softmax_0(buf0, buf1, buf2, buf3)
    del arg0_1
    return (buf3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cpu', dtype=torch.bfloat16)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)

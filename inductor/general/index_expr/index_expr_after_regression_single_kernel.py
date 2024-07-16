
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
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(16L))
                {
                    auto tmp0 =
                    [&]
                    {
                        __at_align__ std::array<int64_t, 16> tmpbuf;
                        #pragma GCC unroll 16
                        for (long x2_inner = 0; x2_inner < 16; x2_inner++)
                        {
                            tmpbuf[x2_inner] = out_ptr12[static_cast<long>((33L*x0) + (33L*(c10::div_floor_integer((x2 + x2_inner + (64L*x1)), 135168L))) + (c10::div_floor_integer((x2 + x2_inner + (64L*x1)), 4096L)))];
                        }
                        return at::vec::VectorizedN<int64_t,2>::loadu(tmpbuf.data(), 16);
                    }
                    ()
                    ;
                    auto tmp15 =
                    [&]
                    {
                        __at_align__ std::array<int64_t, 16> tmpbuf;
                        #pragma GCC unroll 16
                        for (long x2_inner = 0; x2_inner < 16; x2_inner++)
                        {
                            tmpbuf[x2_inner] = out_ptr12[static_cast<long>(30L + (33L*x0) + (33L*(c10::div_floor_integer((122880L + x2 + x2_inner + (64L*x1)), 135168L))) + (c10::div_floor_integer((x2 + x2_inner + (64L*x1)), 4096L)))];
                        }
                        return at::vec::VectorizedN<int64_t,2>::loadu(tmpbuf.data(), 16);
                    }
                    ()
                    ;
                    auto tmp1 =
                    [&]
                    {
                        __at_align__ std::array<int64_t, 16> tmpbuf;
                        #pragma GCC unroll 16
                        for (long x2_inner = 0; x2_inner < 16; x2_inner++)
                        {
                            tmpbuf[x2_inner] = static_cast<long>((13L*x0) + (26L*(c10::div_floor_integer((x2 + x2_inner + (64L*x1)), 135168L))));
                        }
                        return at::vec::VectorizedN<int64_t,2>::loadu(tmpbuf.data(), 16);
                    }
                    ()
                    ;
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp3 = 156L;
                    auto tmp4 = c10::convert<int64_t>(tmp3);
                    auto tmp5 = at::vec::VectorizedN<int64_t,2>(tmp4);
                    auto tmp6 = tmp2 + tmp5;
                    auto tmp7 = static_cast<int64_t>(0);
                    auto tmp8 = at::vec::VectorizedN<int64_t,2>(tmp7);
                    auto tmp9 = at::vec::VecMask<int64_t,2>(tmp2 < tmp8);
                    auto tmp10 = decltype(tmp6)::blendv(tmp2, tmp6, tmp9.template cast<int64_t,2>());
                    auto tmp11 =
                    [&]
                    {
                        __at_align__ std::array<int64_t, 16> tmpbuf;
                        tmp10.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp12 =
                    [&]
                    {
                        __at_align__ std::array<int32_t, 16> tmpbuf;
                        #pragma GCC unroll 16
                        for (long x2_inner = 0; x2_inner < 16; x2_inner++)
                        {
                            tmpbuf[x2_inner] = static_cast<long>(tmp11[x2_inner]);
                        }
                        return at::vec::Vectorized<int32_t>::loadu(tmpbuf.data(), 16);
                    }
                    ()
                    ;
                    TORCH_CHECK((at::vec::VecMask<int32_t,1>((at::vec::Vectorized<int32_t>(0) <= tmp12) & (tmp12 < at::vec::Vectorized<int32_t>(156L)))).all_masked(), "index out of bounds: 0 <= tmp12 < 156L");
                    auto tmp14 =
                    [&]
                    {
                        __at_align__ std::array<bfloat16, 32> tmpbuf;
                        #pragma GCC unroll 16
                        for (long x2_inner = 0; x2_inner < 16; x2_inner++)
                        {
                            tmpbuf[x2_inner] = in_ptr13[static_cast<long>(x2 + x2_inner + (64L*(static_cast<long>(c10::div_floor_integer(tmp11[x2_inner], 13L)) % static_cast<long>(12L))) + (768L*(static_cast<long>(x1) % static_cast<long>(64L))) + (49152L*(static_cast<long>(tmp11[x2_inner]) % static_cast<long>(13L))))];
                        }
                        return at::vec::Vectorized<bfloat16>::loadu(tmpbuf.data(), 16);
                    }
                    ()
                    ;
                    auto tmp16 =
                    [&]
                    {
                        __at_align__ std::array<int64_t, 16> tmpbuf;
                        #pragma GCC unroll 16
                        for (long x2_inner = 0; x2_inner < 16; x2_inner++)
                        {
                            tmpbuf[x2_inner] = static_cast<long>((13L*x0) + (13L*(c10::div_floor_integer((30L + (c10::div_floor_integer((x2 + x2_inner + (64L*x1)), 4096L))), 33L))) + (13L*(c10::div_floor_integer((122880L + x2 + x2_inner + (64L*x1)), 135168L))));
                        }
                        return at::vec::VectorizedN<int64_t,2>::loadu(tmpbuf.data(), 16);
                    }
                    ()
                    ;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp18 = tmp17 + tmp5;
                    auto tmp19 = at::vec::VecMask<int64_t,2>(tmp17 < tmp8);
                    auto tmp20 = decltype(tmp18)::blendv(tmp17, tmp18, tmp19.template cast<int64_t,2>());
                    auto tmp21 =
                    [&]
                    {
                        __at_align__ std::array<int64_t, 16> tmpbuf;
                        tmp20.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp22 =
                    [&]
                    {
                        __at_align__ std::array<int32_t, 16> tmpbuf;
                        #pragma GCC unroll 16
                        for (long x2_inner = 0; x2_inner < 16; x2_inner++)
                        {
                            tmpbuf[x2_inner] = static_cast<long>(tmp21[x2_inner]);
                        }
                        return at::vec::Vectorized<int32_t>::loadu(tmpbuf.data(), 16);
                    }
                    ()
                    ;
                    TORCH_CHECK((at::vec::VecMask<int32_t,1>((at::vec::Vectorized<int32_t>(0) <= tmp22) & (tmp22 < at::vec::Vectorized<int32_t>(156L)))).all_masked(), "index out of bounds: 0 <= tmp22 < 156L");
                    auto tmp24 =
                    [&]
                    {
                        __at_align__ std::array<bfloat16, 32> tmpbuf;
                        #pragma GCC unroll 16
                        for (long x2_inner = 0; x2_inner < 16; x2_inner++)
                        {
                            tmpbuf[x2_inner] = in_ptr13[static_cast<long>(x2 + x2_inner + (64L*(static_cast<long>(c10::div_floor_integer(tmp21[x2_inner], 13L)) % static_cast<long>(12L))) + (768L*(static_cast<long>(x1) % static_cast<long>(64L))) + (49152L*(static_cast<long>(tmp21[x2_inner]) % static_cast<long>(13L))))];
                        }
                        return at::vec::Vectorized<bfloat16>::loadu(tmpbuf.data(), 16);
                    }
                    ()
                    ;
                    tmp14.store(out_ptr19 + static_cast<long>(x2 + (64L*x1) + (28672L*x0)), 16);
                    tmp24.store(out_ptr20 + static_cast<long>(x2 + (64L*x1) + (28672L*x0)), 16);
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

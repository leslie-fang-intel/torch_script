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
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


cpp_fused_index_put_0 = async_compile.cpp_pybinding(['const int32_t*', 'const int32_t*', 'const int64_t*', 'const bfloat16*', 'const bfloat16*', 'bfloat16*', 'bfloat16*'], '''
#include <ATen/record_function.h>
#include "/home/leslie/lz/pytorch/torch/_inductor/codegen/cpp_prefix.h"
extern "C"  void kernel(const int32_t* in_ptr0,
                       const int32_t* in_ptr1,
                       const int64_t* in_ptr2,
                       const bfloat16* in_ptr3,
                       const bfloat16* in_ptr4,
                       bfloat16* out_ptr0,
                       bfloat16* out_ptr1)
{
    RECORD_FUNCTION("graph_0_cpp_fused_index_put_0", c10::ArrayRef<c10::IValue>({}));
    #pragma omp parallel num_threads(32)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(32L); x0+=static_cast<int64_t>(1L))
            {
                #pragma GCC ivdep
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(138240L); x1+=static_cast<int64_t>(1L))
                {
                    for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(128L); x2+=static_cast<int64_t>(32L))
                    {
                        auto tmp0 = in_ptr0[static_cast<int64_t>(static_cast<int64_t>(x1) % static_cast<int64_t>(1152L))];
                        auto tmp11 = in_ptr1[static_cast<int64_t>(c10::div_floor_integer(static_cast<int64_t>(x1), static_cast<int64_t>(1152L)))];
                        auto tmp42 = at::vec::Vectorized<bfloat16>::loadu(in_ptr3 + static_cast<int64_t>(x2 + (128L*(static_cast<int64_t>(x1) % static_cast<int64_t>(1152L))) + (147456L*x0) + (4718592L*(c10::div_floor_integer(static_cast<int64_t>(x1), static_cast<int64_t>(1152L))))), static_cast<int64_t>(32));
                        auto tmp43 = at::vec::Vectorized<bfloat16>::loadu(in_ptr4 + static_cast<int64_t>(x2 + (128L*(static_cast<int64_t>(x1) % static_cast<int64_t>(1152L))) + (147456L*x0) + (4718592L*(c10::div_floor_integer(static_cast<int64_t>(x1), static_cast<int64_t>(1152L))))), static_cast<int64_t>(32));
                        auto tmp1 = static_cast<int32_t>(128);
                        auto tmp2 = ((tmp0 < 0) != (tmp1 < 0) ? (tmp0 % tmp1 != 0 ? tmp0 / tmp1 - 1 : tmp0 / tmp1) : tmp0 / tmp1);
                        auto tmp3 = 1107L;
                        auto tmp4 = c10::convert<int64_t>(tmp3);
                        auto tmp5 = decltype(tmp2)(tmp2 + tmp4);
                        auto tmp6 = tmp2 < 0;
                        auto tmp7 = tmp6 ? tmp5 : tmp2;
                        auto tmp8 = tmp7;
                        auto tmp9 = c10::convert<int64_t>(tmp8);
                        AOTI_TORCH_CHECK((0 <= tmp9) & (tmp9 < 1107L), "index out of bounds: 0 <= tmp9 < 1107L");
                        auto tmp12 = 123L;
                        auto tmp13 = c10::convert<int64_t>(tmp12);
                        auto tmp14 = decltype(tmp11)(tmp11 + tmp13);
                        auto tmp15 = tmp11 < 0;
                        auto tmp16 = tmp15 ? tmp14 : tmp11;
                        auto tmp17 = tmp16;
                        auto tmp18 = c10::convert<int64_t>(tmp17);
                        AOTI_TORCH_CHECK((0 <= tmp18) & (tmp18 < 123L), "index out of bounds: 0 <= tmp18 < 123L");
                        auto tmp20 = in_ptr2[static_cast<int64_t>(tmp7 + (1107L*tmp16))];
                        auto tmp21 = static_cast<int64_t>(128);
                        auto tmp22 = decltype(tmp20)(tmp20 * tmp21);
                        auto tmp23 = mod(tmp0, tmp1);
                        auto tmp24 = static_cast<int32_t>(0);
                        auto tmp25 = tmp23 != tmp24;
                        auto tmp26 = std::signbit(tmp23);
                        auto tmp27 = std::signbit(tmp1);
                        auto tmp28 = tmp26 != tmp27;
                        auto tmp29 = tmp25 & tmp28;
                        auto tmp30 = decltype(tmp23)(tmp23 + tmp1);
                        auto tmp31 = tmp29 ? tmp30 : tmp23;
                        auto tmp32 = c10::convert<int64_t>(tmp31);
                        auto tmp33 = decltype(tmp22)(tmp22 + tmp32);
                        auto tmp34 = 141696L;
                        auto tmp35 = c10::convert<int64_t>(tmp34);
                        auto tmp36 = decltype(tmp33)(tmp33 + tmp35);
                        auto tmp37 = tmp33 < 0;
                        auto tmp38 = tmp37 ? tmp36 : tmp33;
                        auto tmp39 = tmp38;
                        auto tmp40 = c10::convert<int64_t>(tmp39);
                        AOTI_TORCH_CHECK((0 <= tmp40) & (tmp40 < 141696L), "index out of bounds: 0 <= tmp40 < 141696L");
                        tmp42.store(out_ptr0 + static_cast<int64_t>(x2 + (128L*tmp38) + (18137088L*x0)), static_cast<int64_t>(32));
                        tmp43.store(out_ptr1 + static_cast<int64_t>(x2 + (128L*tmp38) + (18137088L*x0)), static_cast<int64_t>(32));
                    }
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1 = args
    args.clear()
    assert_size_stride(arg0_1, (120, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg1_1, (120, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg2_1, (120, ), (1, ))
    assert_size_stride(arg3_1, (1, 32, 141696, 128), (580386816, 18137088, 128, 1))
    assert_size_stride(arg4_1, (1152, ), (1, ))
    assert_size_stride(arg5_1, (1, 32, 141696, 128), (580386816, 18137088, 128, 1))
    assert_size_stride(arg6_1, (123, 1107), (1107, 1))
    from torch.profiler import record_function
    with record_function('graph_0_inductor_wrapper_call'):
        cpp_fused_index_put_0(arg4_1, arg2_1, arg6_1, arg0_1, arg1_1, arg3_1, arg5_1)
        del arg0_1
        del arg1_1
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        del arg6_1
        return ()


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((120, 32, 1152, 128), (4718592, 147456, 128, 1), device='cpu', dtype=torch.bfloat16)
    arg1_1 = rand_strided((120, 32, 1152, 128), (4718592, 147456, 128, 1), device='cpu', dtype=torch.bfloat16)
    arg2_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.int32)
    arg3_1 = rand_strided((1, 32, 141696, 128), (580386816, 18137088, 128, 1), device='cpu', dtype=torch.bfloat16)
    arg4_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.int32)
    arg5_1 = rand_strided((1, 32, 141696, 128), (580386816, 18137088, 128, 1), device='cpu', dtype=torch.bfloat16)
    arg6_1 = rand_strided((123, 1107), (1107, 1), device='cpu', dtype=torch.int64)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)

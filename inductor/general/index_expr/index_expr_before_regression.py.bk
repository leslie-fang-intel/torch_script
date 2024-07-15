
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


cpp_fused__softmax_add_mul_rsub_0 = async_compile.cpp_pybinding(['const bfloat16*', 'const float*', 'float*', 'float*', 'float*', 'bfloat16*'], '''
#include <ATen/record_function.h>
#include "/localdisk/leslie/torch_inductor_community/pytorch/torch/_inductor/codegen/cpp_prefix.h"
extern "C"  void kernel(const bfloat16* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       bfloat16* out_ptr3)
{
    RECORD_FUNCTION("graph_8_cpp_fused__softmax_add_mul_rsub_0", c10::ArrayRef<c10::IValue>({}));
    #pragma omp parallel num_threads(56)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(832L); x1+=static_cast<long>(16L))
                    {
                        auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr0 + static_cast<long>(x1 + (832L*x0)), 16);
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1), 16);
                        auto tmp1 = at::vec::convert<float>(tmp0);
                        auto tmp2 = static_cast<float>(0.125);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp5 = (tmp4);
                        auto tmp7 = static_cast<float>(1.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp8 - tmp6;
                        auto tmp10 = static_cast<float>(-10000.0);
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 * tmp11;
                        auto tmp13 = tmp5 + tmp12;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
                {
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(832L); x1+=static_cast<long>(16L))
                    {
                        auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr0 + static_cast<long>(x1 + (832L*x0)), 16);
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1), 16);
                        auto tmp14 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = at::vec::convert<float>(tmp0);
                        auto tmp2 = static_cast<float>(0.125);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp5 = (tmp4);
                        auto tmp7 = static_cast<float>(1.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp8 - tmp6;
                        auto tmp10 = static_cast<float>(-10000.0);
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 * tmp11;
                        auto tmp13 = tmp5 + tmp12;
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp13 - tmp15;
                        auto tmp17 = tmp16.exp();
                        tmp17.store(out_ptr1 + static_cast<long>(x1 + (832L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp17;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(832L); x1+=static_cast<long>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (832L*x0)), 16);
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = at::vec::convert<bfloat16>(tmp3);
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (832L*x0)), 16);
                }
            }
        }
    }
}
''')


cpp_fused__to_copy_cat_stack_1 = async_compile.cpp_pybinding(['const int32_t*', 'const int32_t*', 'const int32_t*', 'const int32_t*', 'const int32_t*', 'const int32_t*', 'const int32_t*', 'const int32_t*', 'const int32_t*', 'const int32_t*', 'const int32_t*', 'const int32_t*', 'const int32_t*', 'const bfloat16*', 'int32_t*', 'int32_t*', 'int32_t*', 'int32_t*', 'int32_t*', 'int32_t*', 'int32_t*', 'int32_t*', 'int32_t*', 'int32_t*', 'int32_t*', 'int32_t*', 'int64_t*', 'bfloat16*', 'bfloat16*', 'bfloat16*', 'bfloat16*', 'bfloat16*', 'bfloat16*', 'bfloat16*', 'bfloat16*'], '''
#include <ATen/record_function.h>
#include "/localdisk/leslie/torch_inductor_community/pytorch/torch/_inductor/codegen/cpp_prefix.h"
extern "C"  void kernel(const int32_t* in_ptr0,
                       const int32_t* in_ptr1,
                       const int32_t* in_ptr2,
                       const int32_t* in_ptr3,
                       const int32_t* in_ptr4,
                       const int32_t* in_ptr5,
                       const int32_t* in_ptr6,
                       const int32_t* in_ptr7,
                       const int32_t* in_ptr8,
                       const int32_t* in_ptr9,
                       const int32_t* in_ptr10,
                       const int32_t* in_ptr11,
                       const int32_t* in_ptr12,
                       const bfloat16* in_ptr13,
                       int32_t* out_ptr0,
                       int32_t* out_ptr1,
                       int32_t* out_ptr2,
                       int32_t* out_ptr3,
                       int32_t* out_ptr4,
                       int32_t* out_ptr5,
                       int32_t* out_ptr6,
                       int32_t* out_ptr7,
                       int32_t* out_ptr8,
                       int32_t* out_ptr9,
                       int32_t* out_ptr10,
                       int32_t* out_ptr11,
                       int64_t* out_ptr12,
                       bfloat16* out_ptr13,
                       bfloat16* out_ptr14,
                       bfloat16* out_ptr15,
                       bfloat16* out_ptr16,
                       bfloat16* out_ptr17,
                       bfloat16* out_ptr18,
                       bfloat16* out_ptr19,
                       bfloat16* out_ptr20)
{
    RECORD_FUNCTION("graph_8_cpp_fused__to_copy_cat_stack_1", c10::ArrayRef<c10::IValue>({}));
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(16L))
        {
            auto tmp0 = at::vec::Vectorized<int32_t>::loadu(in_ptr0 + static_cast<long>(x0), 16);
            tmp0.store(out_ptr0 + static_cast<long>(x0), 16);
        }
        #pragma omp simd simdlen(8) 
        for(long x0=static_cast<long>(32L); x0<static_cast<long>(33L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_ptr0[static_cast<long>(x0)];
            out_ptr0[static_cast<long>(x0)] = tmp0;
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(16L))
        {
            auto tmp0 = at::vec::Vectorized<int32_t>::loadu(in_ptr1 + static_cast<long>(x0), 16);
            tmp0.store(out_ptr1 + static_cast<long>(x0), 16);
        }
        #pragma omp simd simdlen(8) 
        for(long x0=static_cast<long>(32L); x0<static_cast<long>(33L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_ptr1[static_cast<long>(x0)];
            out_ptr1[static_cast<long>(x0)] = tmp0;
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(16L))
        {
            auto tmp0 = at::vec::Vectorized<int32_t>::loadu(in_ptr2 + static_cast<long>(x0), 16);
            tmp0.store(out_ptr2 + static_cast<long>(x0), 16);
        }
        #pragma omp simd simdlen(8) 
        for(long x0=static_cast<long>(32L); x0<static_cast<long>(33L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_ptr2[static_cast<long>(x0)];
            out_ptr2[static_cast<long>(x0)] = tmp0;
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(16L))
        {
            auto tmp0 = at::vec::Vectorized<int32_t>::loadu(in_ptr3 + static_cast<long>(x0), 16);
            tmp0.store(out_ptr3 + static_cast<long>(x0), 16);
        }
        #pragma omp simd simdlen(8) 
        for(long x0=static_cast<long>(32L); x0<static_cast<long>(33L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_ptr3[static_cast<long>(x0)];
            out_ptr3[static_cast<long>(x0)] = tmp0;
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(16L))
        {
            auto tmp0 = at::vec::Vectorized<int32_t>::loadu(in_ptr4 + static_cast<long>(x0), 16);
            tmp0.store(out_ptr4 + static_cast<long>(x0), 16);
        }
        #pragma omp simd simdlen(8) 
        for(long x0=static_cast<long>(32L); x0<static_cast<long>(33L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_ptr4[static_cast<long>(x0)];
            out_ptr4[static_cast<long>(x0)] = tmp0;
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(16L))
        {
            auto tmp0 = at::vec::Vectorized<int32_t>::loadu(in_ptr5 + static_cast<long>(x0), 16);
            tmp0.store(out_ptr5 + static_cast<long>(x0), 16);
        }
        #pragma omp simd simdlen(8) 
        for(long x0=static_cast<long>(32L); x0<static_cast<long>(33L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_ptr5[static_cast<long>(x0)];
            out_ptr5[static_cast<long>(x0)] = tmp0;
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(16L))
        {
            auto tmp0 = at::vec::Vectorized<int32_t>::loadu(in_ptr6 + static_cast<long>(x0), 16);
            tmp0.store(out_ptr6 + static_cast<long>(x0), 16);
        }
        #pragma omp simd simdlen(8) 
        for(long x0=static_cast<long>(32L); x0<static_cast<long>(33L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_ptr6[static_cast<long>(x0)];
            out_ptr6[static_cast<long>(x0)] = tmp0;
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(16L))
        {
            auto tmp0 = at::vec::Vectorized<int32_t>::loadu(in_ptr7 + static_cast<long>(x0), 16);
            tmp0.store(out_ptr7 + static_cast<long>(x0), 16);
        }
        #pragma omp simd simdlen(8) 
        for(long x0=static_cast<long>(32L); x0<static_cast<long>(33L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_ptr7[static_cast<long>(x0)];
            out_ptr7[static_cast<long>(x0)] = tmp0;
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(16L))
        {
            auto tmp0 = at::vec::Vectorized<int32_t>::loadu(in_ptr8 + static_cast<long>(x0), 16);
            tmp0.store(out_ptr8 + static_cast<long>(x0), 16);
        }
        #pragma omp simd simdlen(8) 
        for(long x0=static_cast<long>(32L); x0<static_cast<long>(33L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_ptr8[static_cast<long>(x0)];
            out_ptr8[static_cast<long>(x0)] = tmp0;
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(16L))
        {
            auto tmp0 = at::vec::Vectorized<int32_t>::loadu(in_ptr9 + static_cast<long>(x0), 16);
            tmp0.store(out_ptr9 + static_cast<long>(x0), 16);
        }
        #pragma omp simd simdlen(8) 
        for(long x0=static_cast<long>(32L); x0<static_cast<long>(33L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_ptr9[static_cast<long>(x0)];
            out_ptr9[static_cast<long>(x0)] = tmp0;
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(16L))
        {
            auto tmp0 = at::vec::Vectorized<int32_t>::loadu(in_ptr10 + static_cast<long>(x0), 16);
            tmp0.store(out_ptr10 + static_cast<long>(x0), 16);
        }
        #pragma omp simd simdlen(8) 
        for(long x0=static_cast<long>(32L); x0<static_cast<long>(33L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_ptr10[static_cast<long>(x0)];
            out_ptr10[static_cast<long>(x0)] = tmp0;
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(16L))
        {
            auto tmp0 = at::vec::Vectorized<int32_t>::loadu(in_ptr11 + static_cast<long>(x0), 16);
            tmp0.store(out_ptr11 + static_cast<long>(x0), 16);
        }
        #pragma omp simd simdlen(8) 
        for(long x0=static_cast<long>(32L); x0<static_cast<long>(33L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_ptr11[static_cast<long>(x0)];
            out_ptr11[static_cast<long>(x0)] = tmp0;
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(16L))
        {
            auto tmp0 = at::vec::Vectorized<int32_t>::loadu(in_ptr12 + static_cast<long>(x0), 16);
            auto tmp1 = at::vec::convert<int64_t,2,int32_t,1>(tmp0);
            tmp1.store(out_ptr12 + static_cast<long>(x0), 16);
        }
        #pragma omp simd simdlen(8) 
        for(long x0=static_cast<long>(384L); x0<static_cast<long>(396L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_ptr12[static_cast<long>(x0)];
            auto tmp1 = c10::convert<int64_t>(tmp0);
            out_ptr12[static_cast<long>(x0)] = tmp1;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(32L))
                {
                    auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr13 + static_cast<long>(x2 + (64L*x0) + (768L*x1)), 32);
                    tmp0.store(out_ptr13 + static_cast<long>(x2 + (64L*x1) + (28672L*x0)), 32);
                    tmp0.store(out_ptr14 + static_cast<long>(x2 + (64L*x1) + (28672L*x0)), 32);
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(32L))
                {
                    auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr13 + static_cast<long>(49152L + x2 + (64L*x0) + (768L*x1)), 32);
                    tmp0.store(out_ptr15 + static_cast<long>(x2 + (64L*x1) + (28672L*x0)), 32);
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(32L))
                {
                    auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr13 + static_cast<long>(98304L + x2 + (64L*x0) + (768L*x1)), 32);
                    tmp0.store(out_ptr16 + static_cast<long>(x2 + (64L*x1) + (28672L*x0)), 32);
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(32L))
                {
                    auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr13 + static_cast<long>(589824L + x2 + (64L*x0) + (768L*x1)), 32);
                    tmp0.store(out_ptr17 + static_cast<long>(x2 + (64L*x1) + (28672L*x0)), 32);
                    tmp0.store(out_ptr18 + static_cast<long>(x2 + (64L*x1) + (28672L*x0)), 32);
                }
            }
        }
    }
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


cpp_fused__softmax_add_cat_minimum_mul_new_ones_rsub_2 = async_compile.cpp_pybinding(['const float*', 'const float*', 'const int64_t*', 'const bfloat16*', 'const float*', 'const float*', 'const bfloat16*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'bfloat16*', 'bfloat16*', 'bfloat16*', 'bfloat16*', 'bfloat16*', 'bfloat16*', 'bfloat16*', 'bfloat16*', 'bfloat16*'], '''
#include <ATen/record_function.h>
#include "/localdisk/leslie/torch_inductor_community/pytorch/torch/_inductor/codegen/cpp_prefix.h"
extern "C"  void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const int64_t* in_ptr2,
                       const bfloat16* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const bfloat16* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8,
                       bfloat16* out_ptr9,
                       bfloat16* out_ptr10,
                       bfloat16* out_ptr11,
                       bfloat16* out_ptr12,
                       bfloat16* out_ptr13,
                       bfloat16* out_ptr14,
                       bfloat16* out_ptr15,
                       bfloat16* out_ptr16,
                       bfloat16* out_ptr17)
{
    RECORD_FUNCTION("graph_8_cpp_fused__softmax_add_cat_minimum_mul_new_ones_rsub_2", c10::ArrayRef<c10::IValue>({}));
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(16L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0), 16);
            tmp0.store(out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(16L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x0), 16);
            tmp0.store(out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(16L))
        {
            auto tmp0 = static_cast<float>(1.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(16L))
            {
                auto tmp0 = static_cast<float>(1.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr3 + static_cast<long>(x1 + (448L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(16L))
                {
                    auto tmp0 = in_ptr1[static_cast<long>(64L + x1)];
                    auto tmp1 = in_ptr2[static_cast<long>((33L*x0) + (c10::div_floor_integer(x2, 64L)))];
                    auto tmp12 = in_ptr1[static_cast<long>(704L + x1)];
                    auto tmp13 = in_ptr2[static_cast<long>(30L + (33L*x0) + (c10::div_floor_integer(x2, 64L)))];
                    auto tmp2 = 13L;
                    auto tmp3 = c10::convert<int64_t>(tmp2);
                    auto tmp4 = decltype(tmp1)(tmp1 + tmp3);
                    auto tmp5 = tmp1 < 0;
                    auto tmp6 = tmp5 ? tmp4 : tmp1;
                    auto tmp7 =
                    [&]
                    {
                        __at_align__ std::array<int64_t, 16> tmpbuf;
                        #pragma GCC unroll 16
                        for (long x2_inner = 0; x2_inner < 16; x2_inner++)
                        {
                            tmpbuf[x2_inner] = static_cast<long>(tmp6);
                        }
                        return at::vec::VectorizedN<int64_t,2>::loadu(tmpbuf.data(), 16);
                    }
                    ()
                    ;
                    TORCH_CHECK((at::vec::VecMask<int64_t,2>((at::vec::VectorizedN<int64_t,2>(0) <= tmp7) & (tmp7 < at::vec::VectorizedN<int64_t,2>(13L)))).all_masked(), "index out of bounds: 0 <= tmp7 < 13L");
                    auto tmp9 =
                    [&]
                    {
                        __at_align__ std::array<float, 16> tmpbuf;
                        #pragma GCC unroll 16
                        for (long x2_inner = 0; x2_inner < 16; x2_inner++)
                        {
                            tmpbuf[x2_inner] = in_ptr1[static_cast<long>((64L*tmp6) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(64L)))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 16);
                    }
                    ()
                    ;
                    auto tmp10 = at::vec::Vectorized<float>(tmp0);
                    auto tmp11 = tmp10 * tmp9;
                    auto tmp14 = decltype(tmp13)(tmp13 + tmp3);
                    auto tmp15 = tmp13 < 0;
                    auto tmp16 = tmp15 ? tmp14 : tmp13;
                    auto tmp17 =
                    [&]
                    {
                        __at_align__ std::array<int64_t, 16> tmpbuf;
                        #pragma GCC unroll 16
                        for (long x2_inner = 0; x2_inner < 16; x2_inner++)
                        {
                            tmpbuf[x2_inner] = static_cast<long>(tmp16);
                        }
                        return at::vec::VectorizedN<int64_t,2>::loadu(tmpbuf.data(), 16);
                    }
                    ()
                    ;
                    TORCH_CHECK((at::vec::VecMask<int64_t,2>((at::vec::VectorizedN<int64_t,2>(0) <= tmp17) & (tmp17 < at::vec::VectorizedN<int64_t,2>(13L)))).all_masked(), "index out of bounds: 0 <= tmp17 < 13L");
                    auto tmp19 =
                    [&]
                    {
                        __at_align__ std::array<float, 16> tmpbuf;
                        #pragma GCC unroll 16
                        for (long x2_inner = 0; x2_inner < 16; x2_inner++)
                        {
                            tmpbuf[x2_inner] = in_ptr1[static_cast<long>((64L*tmp16) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(64L)))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 16);
                    }
                    ()
                    ;
                    auto tmp20 = at::vec::Vectorized<float>(tmp12);
                    auto tmp21 = tmp20 * tmp19;
                    tmp11.store(out_ptr4 + static_cast<long>(x2 + (448L*x1) + (28672L*x0)));
                    tmp21.store(out_ptr5 + static_cast<long>(x2 + (448L*x1) + (28672L*x0)));
                }
            }
        }
    }
    #pragma omp parallel num_threads(56)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(16L))
                    {
                        auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr3 + static_cast<long>(x1 + (448L*x0)), 16);
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1), 16);
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (448L*x0)), 16);
                        auto tmp1 = at::vec::convert<float>(tmp0);
                        auto tmp2 = static_cast<float>(0.125);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp5 = (tmp4);
                        auto tmp8 = at::vec::minimum(tmp6, tmp7);
                        auto tmp9 = static_cast<float>(1.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp10 - tmp8;
                        auto tmp12 = static_cast<float>(-10000.0);
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp5 + tmp14;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp15);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr6[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
                {
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(16L))
                    {
                        auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr3 + static_cast<long>(x1 + (448L*x0)), 16);
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1), 16);
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (448L*x0)), 16);
                        auto tmp16 = out_ptr6[static_cast<long>(x0)];
                        auto tmp1 = at::vec::convert<float>(tmp0);
                        auto tmp2 = static_cast<float>(0.125);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp5 = (tmp4);
                        auto tmp8 = at::vec::minimum(tmp6, tmp7);
                        auto tmp9 = static_cast<float>(1.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp10 - tmp8;
                        auto tmp12 = static_cast<float>(-10000.0);
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp5 + tmp14;
                        auto tmp17 = at::vec::Vectorized<float>(tmp16);
                        auto tmp18 = tmp15 - tmp17;
                        auto tmp19 = tmp18.exp();
                        tmp19.store(out_ptr7 + static_cast<long>(x1 + (448L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp19;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr8[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr7 + static_cast<long>(x1 + (448L*x0)), 16);
                    auto tmp1 = out_ptr8[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = at::vec::convert<bfloat16>(tmp3);
                    tmp4.store(out_ptr9 + static_cast<long>(x1 + (448L*x0)), 16);
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(32L))
                        {
                            auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr6 + static_cast<long>(x2 + (64L*x0) + (768L*x1)), 32);
                            tmp0.store(out_ptr10 + static_cast<long>(x2 + (64L*x1) + (28672L*x0)), 32);
                            tmp0.store(out_ptr11 + static_cast<long>(x2 + (64L*x1) + (28672L*x0)), 32);
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(32L))
                        {
                            auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr6 + static_cast<long>(49152L + x2 + (64L*x0) + (768L*x1)), 32);
                            tmp0.store(out_ptr12 + static_cast<long>(x2 + (64L*x1) + (28672L*x0)), 32);
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(32L))
                        {
                            auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr6 + static_cast<long>(98304L + x2 + (64L*x0) + (768L*x1)), 32);
                            tmp0.store(out_ptr13 + static_cast<long>(x2 + (64L*x1) + (28672L*x0)), 32);
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(32L))
                        {
                            auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr6 + static_cast<long>(589824L + x2 + (64L*x0) + (768L*x1)), 32);
                            tmp0.store(out_ptr14 + static_cast<long>(x2 + (64L*x1) + (28672L*x0)), 32);
                            tmp0.store(out_ptr15 + static_cast<long>(x2 + (64L*x1) + (28672L*x0)), 32);
                        }
                    }
                }
            }
        }
        #pragma omp single
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
                            auto tmp0 = in_ptr2[static_cast<long>((33L*x0) + (33L*(c10::div_floor_integer((x2 + (64L*x1)), 135168L))) + (c10::div_floor_integer((x2 + (64L*x1)), 4096L)))];
                            auto tmp13 = in_ptr2[static_cast<long>(30L + (33L*x0) + (33L*(c10::div_floor_integer((122880L + x2 + (64L*x1)), 135168L))) + (c10::div_floor_integer((x2 + (64L*x1)), 4096L)))];
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
                            auto tmp12 = in_ptr6[static_cast<long>(x2 + (64L*(static_cast<long>(c10::div_floor_integer(tmp8, 13L)) % static_cast<long>(12L))) + (768L*(static_cast<long>(x1) % static_cast<long>(64L))) + (49152L*(static_cast<long>(tmp8) % static_cast<long>(13L))))];
                            auto tmp14 = (13L*x0) + (13L*(c10::div_floor_integer((30L + (c10::div_floor_integer((x2 + (64L*x1)), 4096L))), 33L))) + (13L*(c10::div_floor_integer((122880L + x2 + (64L*x1)), 135168L)));
                            auto tmp15 = c10::convert<int64_t>(tmp14);
                            auto tmp16 = decltype(tmp13)(tmp13 + tmp15);
                            auto tmp17 = decltype(tmp16)(tmp16 + tmp5);
                            auto tmp18 = tmp16 < 0;
                            auto tmp19 = tmp18 ? tmp17 : tmp16;
                            auto tmp20 = tmp19;
                            auto tmp21 = c10::convert<int64_t>(tmp20);
                            TORCH_CHECK((0 <= tmp21) & (tmp21 < 156L), "index out of bounds: 0 <= tmp21 < 156L");
                            auto tmp23 = in_ptr6[static_cast<long>(x2 + (64L*(static_cast<long>(c10::div_floor_integer(tmp19, 13L)) % static_cast<long>(12L))) + (768L*(static_cast<long>(x1) % static_cast<long>(64L))) + (49152L*(static_cast<long>(tmp19) % static_cast<long>(13L))))];
                            out_ptr16[static_cast<long>(x2 + (64L*x1) + (28672L*x0))] = tmp12;
                            out_ptr17[static_cast<long>(x2 + (64L*x1) + (28672L*x0))] = tmp23;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_clone_3 = async_compile.cpp_pybinding(['const bfloat16*', 'const bfloat16*', 'bfloat16*', 'bfloat16*', 'bfloat16*', 'bfloat16*'], '''
#include <ATen/record_function.h>
#include "/localdisk/leslie/torch_inductor_community/pytorch/torch/_inductor/codegen/cpp_prefix.h"
extern "C"  void kernel(const bfloat16* in_ptr0,
                       const bfloat16* in_ptr1,
                       bfloat16* out_ptr0,
                       bfloat16* out_ptr1,
                       bfloat16* out_ptr2,
                       bfloat16* out_ptr3)
{
    RECORD_FUNCTION("graph_8_cpp_fused_cat_clone_3", c10::ArrayRef<c10::IValue>({}));
    #pragma omp parallel num_threads(56)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(32L))
                        {
                            auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr0 + static_cast<long>(49152L + x3 + (64L*x0) + (768L*x2) + (49152L*x1)), 32);
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12288L*x1) + (110592L*x0)), 32);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(32L))
                        {
                            auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr0 + static_cast<long>(98304L + x3 + (64L*x0) + (768L*x2) + (49152L*x1)), 32);
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (64L*x2) + (12288L*x1) + (110592L*x0)), 32);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(32L))
                        {
                            auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr0 + static_cast<long>(147456L + x3 + (64L*x0) + (768L*x2) + (49152L*x1)), 32);
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (64L*x2) + (12288L*x1) + (110592L*x0)), 32);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(32L))
                    {
                        auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr1 + static_cast<long>(98304L + x2 + (64L*x0) + (768L*x1)), 32);
                        tmp0.store(out_ptr3 + static_cast<long>(x2 + (64L*x1) + (36864L*x0)), 32);
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_4 = async_compile.cpp_pybinding(['const int64_t*', 'const bfloat16*', 'const bfloat16*', 'bfloat16*', 'bfloat16*'], '''
#include <ATen/record_function.h>
#include "/localdisk/leslie/torch_inductor_community/pytorch/torch/_inductor/codegen/cpp_prefix.h"
extern "C"  void kernel(const int64_t* in_ptr0,
                       const bfloat16* in_ptr1,
                       const bfloat16* in_ptr2,
                       bfloat16* out_ptr0,
                       bfloat16* out_ptr1)
{
    RECORD_FUNCTION("graph_8_cpp_fused_clone_4", c10::ArrayRef<c10::IValue>({}));
    #pragma omp parallel num_threads(56)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(3L + (3L*x1) + (33L*x0) + (33L*(c10::div_floor_integer((12288L + x3 + (64L*x2) + (12288L*x1)), 135168L))) + (c10::div_floor_integer((x3 + (64L*x2)), 4096L)))];
                            auto tmp1 = (13L*x0) + (13L*(c10::div_floor_integer((3L + (3L*x1) + (c10::div_floor_integer((x3 + (64L*x2)), 4096L))), 33L))) + (13L*(c10::div_floor_integer((12288L + x3 + (64L*x2) + (12288L*x1)), 135168L)));
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
                            auto tmp12 = in_ptr1[static_cast<long>(x3 + (64L*(static_cast<long>(c10::div_floor_integer(tmp8, 13L)) % static_cast<long>(12L))) + (768L*(static_cast<long>(x2) % static_cast<long>(64L))) + (49152L*(static_cast<long>(tmp8) % static_cast<long>(13L))))];
                            auto tmp13 = in_ptr2[static_cast<long>(x3 + (64L*(static_cast<long>(c10::div_floor_integer(tmp8, 13L)) % static_cast<long>(12L))) + (768L*(static_cast<long>(x2) % static_cast<long>(64L))) + (49152L*(static_cast<long>(tmp8) % static_cast<long>(13L))))];
                            out_ptr0[static_cast<long>(x3 + (64L*x2) + (12288L*x1) + (110592L*x0))] = tmp12;
                            out_ptr1[static_cast<long>(x3 + (64L*x2) + (12288L*x1) + (110592L*x0))] = tmp13;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax__to_copy_add_cat_mul_rsub_5 = async_compile.cpp_pybinding(['const bfloat16*', 'const float*', 'const bfloat16*', 'const float*', 'const bfloat16*', 'const float*', 'const int64_t*', 'const bfloat16*', 'const bfloat16*', 'const bfloat16*', 'bfloat16*', 'bfloat16*', 'bfloat16*', 'bfloat16*', 'float*', 'float*', 'float*', 'bfloat16*', 'bfloat16*', 'bfloat16*', 'bfloat16*'], '''
#include <ATen/record_function.h>
#include "/localdisk/leslie/torch_inductor_community/pytorch/torch/_inductor/codegen/cpp_prefix.h"
extern "C"  void kernel(const bfloat16* in_ptr0,
                       const float* in_ptr1,
                       const bfloat16* in_ptr2,
                       const float* in_ptr3,
                       const bfloat16* in_ptr4,
                       const float* in_ptr5,
                       const int64_t* in_ptr6,
                       const bfloat16* in_ptr7,
                       const bfloat16* in_ptr8,
                       const bfloat16* in_ptr9,
                       bfloat16* out_ptr0,
                       bfloat16* out_ptr1,
                       bfloat16* out_ptr2,
                       bfloat16* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       bfloat16* out_ptr7,
                       bfloat16* out_ptr8,
                       bfloat16* out_ptr9,
                       bfloat16* out_ptr10)
{
    RECORD_FUNCTION("graph_8_cpp_fused__softmax__to_copy_add_cat_mul_rsub_5", c10::ArrayRef<c10::IValue>({}));
    #pragma omp parallel num_threads(56)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6912L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)), 16);
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1), 16);
                    auto tmp1 = at::vec::convert<float>(tmp0);
                    auto tmp2 = static_cast<float>(0.125);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = (tmp4);
                    auto tmp7 = static_cast<float>(1.0);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp8 - tmp6;
                    auto tmp10 = static_cast<float>(-10000.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 + tmp12;
                    auto tmp14 = at::vec::convert<bfloat16>(tmp13);
                    tmp14.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)), 16);
                }
            }
        }
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(16L))
                    {
                        auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr2 + static_cast<long>(x2 + (192L*x1) + (110592L*x0)), 16);
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (192L*x1)), 16);
                        auto tmp1 = at::vec::convert<float>(tmp0);
                        auto tmp2 = static_cast<float>(0.125);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp5 = (tmp4);
                        auto tmp7 = static_cast<float>(1.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp8 - tmp6;
                        auto tmp10 = static_cast<float>(-10000.0);
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 * tmp11;
                        auto tmp13 = tmp5 + tmp12;
                        auto tmp14 = at::vec::convert<bfloat16>(tmp13);
                        tmp14.store(out_ptr1 + static_cast<long>(x2 + (512L*x1) + (294912L*x0)), 16);
                    }
                }
            }
        }
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(16L))
                        {
                            auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr4 + static_cast<long>(x3 + (192L*x2) + (12288L*x1) + (110592L*x0)), 16);
                            auto tmp6 = in_ptr5[static_cast<long>(128L + x2 + (64L*x1))];
                            auto tmp7 = in_ptr6[static_cast<long>(3L + (3L*x1) + (33L*x0) + (c10::div_floor_integer(x3, 64L)))];
                            auto tmp1 = at::vec::convert<float>(tmp0);
                            auto tmp2 = static_cast<float>(0.125);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = (tmp4);
                            auto tmp8 = 13L;
                            auto tmp9 = c10::convert<int64_t>(tmp8);
                            auto tmp10 = decltype(tmp7)(tmp7 + tmp9);
                            auto tmp11 = tmp7 < 0;
                            auto tmp12 = tmp11 ? tmp10 : tmp7;
                            auto tmp13 =
                            [&]
                            {
                                __at_align__ std::array<int64_t, 16> tmpbuf;
                                #pragma GCC unroll 16
                                for (long x3_inner = 0; x3_inner < 16; x3_inner++)
                                {
                                    tmpbuf[x3_inner] = static_cast<long>(tmp12);
                                }
                                return at::vec::VectorizedN<int64_t,2>::loadu(tmpbuf.data(), 16);
                            }
                            ()
                            ;
                            TORCH_CHECK((at::vec::VecMask<int64_t,2>((at::vec::VectorizedN<int64_t,2>(0) <= tmp13) & (tmp13 < at::vec::VectorizedN<int64_t,2>(13L)))).all_masked(), "index out of bounds: 0 <= tmp13 < 13L");
                            auto tmp15 =
                            [&]
                            {
                                __at_align__ std::array<float, 16> tmpbuf;
                                #pragma GCC unroll 16
                                for (long x3_inner = 0; x3_inner < 16; x3_inner++)
                                {
                                    tmpbuf[x3_inner] = in_ptr5[static_cast<long>((64L*tmp12) + (static_cast<long>((x3 + x3_inner)) % static_cast<long>(64L)))];
                                }
                                return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 16);
                            }
                            ()
                            ;
                            auto tmp16 = at::vec::Vectorized<float>(tmp6);
                            auto tmp17 = tmp16 * tmp15;
                            auto tmp18 = static_cast<float>(1.0);
                            auto tmp19 = at::vec::Vectorized<float>(tmp18);
                            auto tmp20 = tmp19 - tmp17;
                            auto tmp21 = static_cast<float>(-10000.0);
                            auto tmp22 = at::vec::Vectorized<float>(tmp21);
                            auto tmp23 = tmp20 * tmp22;
                            auto tmp24 = tmp5 + tmp23;
                            auto tmp25 = at::vec::convert<bfloat16>(tmp24);
                            tmp25.store(out_ptr2 + static_cast<long>(x3 + (512L*x2) + (32768L*x1) + (294912L*x0)), 16);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6912L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr7 + static_cast<long>(x1 + (64L*x0)), 16);
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(768L + x1), 16);
                    auto tmp1 = at::vec::convert<float>(tmp0);
                    auto tmp2 = static_cast<float>(0.125);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = (tmp4);
                    auto tmp7 = static_cast<float>(1.0);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp8 - tmp6;
                    auto tmp10 = static_cast<float>(-10000.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 + tmp12;
                    auto tmp14 = at::vec::convert<bfloat16>(tmp13);
                    tmp14.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)), 16);
                }
            }
        }
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6912L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(16L))
                    {
                        auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr8 + static_cast<long>(x1 + (512L*x0)), 16);
                        auto tmp1 = at::vec::convert<float>(tmp0);
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
                {
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(16L))
                    {
                        auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr8 + static_cast<long>(x1 + (512L*x0)), 16);
                        auto tmp2 = out_ptr4[static_cast<long>(x0)];
                        auto tmp1 = at::vec::convert<float>(tmp0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp5 = tmp4.exp();
                        tmp5.store(out_ptr5 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr6[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x1 + (512L*x0)), 16);
                    auto tmp1 = out_ptr6[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = at::vec::convert<bfloat16>(tmp3);
                    tmp4.store(out_ptr7 + static_cast<long>(x1 + (512L*x0)), 16);
                }
            }
        }
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(32L))
                        {
                            auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr9 + static_cast<long>(49152L + x3 + (64L*x0) + (768L*x2) + (49152L*x1)), 32);
                            tmp0.store(out_ptr8 + static_cast<long>(x3 + (64L*x2) + (12288L*x1) + (110592L*x0)), 32);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(32L))
                        {
                            auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr9 + static_cast<long>(98304L + x3 + (64L*x0) + (768L*x2) + (49152L*x1)), 32);
                            tmp0.store(out_ptr9 + static_cast<long>(x3 + (64L*x2) + (12288L*x1) + (110592L*x0)), 32);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(32L))
                        {
                            auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr9 + static_cast<long>(147456L + x3 + (64L*x0) + (768L*x2) + (49152L*x1)), 32);
                            tmp0.store(out_ptr10 + static_cast<long>(x3 + (64L*x2) + (12288L*x1) + (110592L*x0)), 32);
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_6 = async_compile.cpp_pybinding(['const bfloat16*', 'bfloat16*', 'bfloat16*'], '''
#include <ATen/record_function.h>
#include "/localdisk/leslie/torch_inductor_community/pytorch/torch/_inductor/codegen/cpp_prefix.h"
extern "C"  void kernel(const bfloat16* in_ptr0,
                       bfloat16* out_ptr0,
                       bfloat16* out_ptr1)
{
    RECORD_FUNCTION("graph_8_cpp_fused_cat_6", c10::ArrayRef<c10::IValue>({}));
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(32L))
                {
                    auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr0 + static_cast<long>(491520L + x2 + (64L*x0) + (768L*x1)), 32);
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (28672L*x0)), 32);
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(32L))
                {
                    auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr0 + static_cast<long>(540672L + x2 + (64L*x0) + (768L*x1)), 32);
                    tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (28672L*x0)), 32);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_cat_minimum_mul_rsub_7 = async_compile.cpp_pybinding(['const float*', 'const bfloat16*', 'const float*', 'const float*', 'const bfloat16*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'bfloat16*', 'bfloat16*', 'bfloat16*'], '''
#include <ATen/record_function.h>
#include "/localdisk/leslie/torch_inductor_community/pytorch/torch/_inductor/codegen/cpp_prefix.h"
extern "C"  void kernel(const float* in_ptr0,
                       const bfloat16* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const bfloat16* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       bfloat16* out_ptr7,
                       bfloat16* out_ptr8,
                       bfloat16* out_ptr9)
{
    RECORD_FUNCTION("graph_8_cpp_fused__softmax_add_cat_minimum_mul_rsub_7", c10::ArrayRef<c10::IValue>({}));
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(16L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0), 16);
            tmp0.store(out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(16L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(640L + x0), 16);
            tmp0.store(out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(16L))
        {
            auto tmp0 = static_cast<float>(1.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(16L))
            {
                auto tmp0 = static_cast<float>(1.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr3 + static_cast<long>(x1 + (448L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(56)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(16L))
                    {
                        auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr1 + static_cast<long>(x1 + (448L*x0)), 16);
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1), 16);
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (448L*x0)), 16);
                        auto tmp1 = at::vec::convert<float>(tmp0);
                        auto tmp2 = static_cast<float>(0.125);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp5 = (tmp4);
                        auto tmp8 = at::vec::minimum(tmp6, tmp7);
                        auto tmp9 = static_cast<float>(1.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp10 - tmp8;
                        auto tmp12 = static_cast<float>(-10000.0);
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp5 + tmp14;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp15);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
                {
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(16L))
                    {
                        auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr1 + static_cast<long>(x1 + (448L*x0)), 16);
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1), 16);
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (448L*x0)), 16);
                        auto tmp16 = out_ptr4[static_cast<long>(x0)];
                        auto tmp1 = at::vec::convert<float>(tmp0);
                        auto tmp2 = static_cast<float>(0.125);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp5 = (tmp4);
                        auto tmp8 = at::vec::minimum(tmp6, tmp7);
                        auto tmp9 = static_cast<float>(1.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp10 - tmp8;
                        auto tmp12 = static_cast<float>(-10000.0);
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp5 + tmp14;
                        auto tmp17 = at::vec::Vectorized<float>(tmp16);
                        auto tmp18 = tmp15 - tmp17;
                        auto tmp19 = tmp18.exp();
                        tmp19.store(out_ptr5 + static_cast<long>(x1 + (448L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp19;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr6[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x1 + (448L*x0)), 16);
                    auto tmp1 = out_ptr6[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = at::vec::convert<bfloat16>(tmp3);
                    tmp4.store(out_ptr7 + static_cast<long>(x1 + (448L*x0)), 16);
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(32L))
                        {
                            auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr4 + static_cast<long>(491520L + x2 + (64L*x0) + (768L*x1)), 32);
                            tmp0.store(out_ptr8 + static_cast<long>(x2 + (64L*x1) + (28672L*x0)), 32);
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(32L))
                        {
                            auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr4 + static_cast<long>(540672L + x2 + (64L*x0) + (768L*x1)), 32);
                            tmp0.store(out_ptr9 + static_cast<long>(x2 + (64L*x1) + (28672L*x0)), 32);
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_mul_rsub_8 = async_compile.cpp_pybinding(['const bfloat16*', 'const float*', 'float*', 'float*', 'float*', 'bfloat16*'], '''
#include <ATen/record_function.h>
#include "/localdisk/leslie/torch_inductor_community/pytorch/torch/_inductor/codegen/cpp_prefix.h"
extern "C"  void kernel(const bfloat16* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       bfloat16* out_ptr3)
{
    RECORD_FUNCTION("graph_8_cpp_fused__softmax_add_mul_rsub_8", c10::ArrayRef<c10::IValue>({}));
    #pragma omp parallel num_threads(56)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(832L); x1+=static_cast<long>(16L))
                    {
                        auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr0 + static_cast<long>(x1 + (832L*x0)), 16);
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1), 16);
                        auto tmp1 = at::vec::convert<float>(tmp0);
                        auto tmp2 = static_cast<float>(0.125);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp5 = (tmp4);
                        auto tmp7 = static_cast<float>(1.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp8 - tmp6;
                        auto tmp10 = static_cast<float>(-10000.0);
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 * tmp11;
                        auto tmp13 = tmp5 + tmp12;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
                {
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(832L); x1+=static_cast<long>(16L))
                    {
                        auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr0 + static_cast<long>(x1 + (832L*x0)), 16);
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1), 16);
                        auto tmp14 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = at::vec::convert<float>(tmp0);
                        auto tmp2 = static_cast<float>(0.125);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp5 = (tmp4);
                        auto tmp7 = static_cast<float>(1.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp8 - tmp6;
                        auto tmp10 = static_cast<float>(-10000.0);
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 * tmp11;
                        auto tmp13 = tmp5 + tmp12;
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp13 - tmp15;
                        auto tmp17 = tmp16.exp();
                        tmp17.store(out_ptr1 + static_cast<long>(x1 + (832L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp17;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(832L); x1+=static_cast<long>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (832L*x0)), 16);
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = at::vec::convert<bfloat16>(tmp3);
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (832L*x0)), 16);
                }
            }
        }
    }
}
''')


cpp_fused_cat_mul_9 = async_compile.cpp_pybinding(['const bfloat16*', 'const bfloat16*', 'const bfloat16*', 'const bfloat16*', 'const bfloat16*', 'const bfloat16*', 'const bfloat16*', 'const bfloat16*', 'const bfloat16*', 'const float*', 'bfloat16*', 'bfloat16*', 'bfloat16*', 'bfloat16*', 'bfloat16*', 'float*'], '''
#include <ATen/record_function.h>
#include "/localdisk/leslie/torch_inductor_community/pytorch/torch/_inductor/codegen/cpp_prefix.h"
extern "C"  void kernel(const bfloat16* in_ptr0,
                       const bfloat16* in_ptr1,
                       const bfloat16* in_ptr2,
                       const bfloat16* in_ptr3,
                       const bfloat16* in_ptr4,
                       const bfloat16* in_ptr5,
                       const bfloat16* in_ptr6,
                       const bfloat16* in_ptr7,
                       const bfloat16* in_ptr8,
                       const float* in_ptr9,
                       bfloat16* out_ptr0,
                       bfloat16* out_ptr1,
                       bfloat16* out_ptr2,
                       bfloat16* out_ptr3,
                       bfloat16* out_ptr4,
                       float* out_ptr5)
{
    RECORD_FUNCTION("graph_8_cpp_fused_cat_mul_9", c10::ArrayRef<c10::IValue>({}));
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(32L))
            {
                auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)), 32);
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (53248L*x0)), 32);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(32L))
            {
                auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)), 32);
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (53248L*x0)), 32);
            }
        }
    }
    #pragma omp parallel num_threads(56)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(36864L); x1+=static_cast<long>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr2 + static_cast<long>(x1 + (36864L*x0)), 16);
                    auto tmp2 = at::vec::Vectorized<bfloat16>::loadu(in_ptr3 + static_cast<long>(x1 + (36864L*x0)), 16);
                    auto tmp5 = at::vec::Vectorized<bfloat16>::loadu(in_ptr4 + static_cast<long>(x1 + (36864L*x0)), 16);
                    auto tmp8 = at::vec::Vectorized<bfloat16>::loadu(in_ptr5 + static_cast<long>(x1 + (36864L*x0)), 16);
                    auto tmp1 = at::vec::convert<float>(tmp0);
                    auto tmp3 = at::vec::convert<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp6 = at::vec::convert<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp9 = at::vec::convert<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = at::vec::convert<bfloat16>(tmp10);
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (53248L*x0)), 16);
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(32L))
                    {
                        auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr6 + static_cast<long>(x1 + (4096L*x0)), 32);
                        tmp0.store(out_ptr3 + static_cast<long>(x1 + (53248L*x0)), 32);
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(32L))
                    {
                        auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr7 + static_cast<long>(x1 + (4096L*x0)), 32);
                        tmp0.store(out_ptr4 + static_cast<long>(x1 + (53248L*x0)), 32);
                    }
                }
            }
        }
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(832L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(16L))
                    {
                        auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr8 + static_cast<long>(x2 + (64L*x1) + (53248L*x0)), 16);
                        auto tmp2 = in_ptr9[static_cast<long>(x1)];
                        auto tmp1 = at::vec::convert<float>(tmp0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        tmp4.store(out_ptr5 + static_cast<long>(x2 + (64L*x1) + (53248L*x0)));
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1 = args
    args.clear()
    assert_size_stride(arg0_1, (11, 3), (3, 1))
    assert_size_stride(arg1_1, (11, 3), (3, 1))
    assert_size_stride(arg2_1, (11, 3), (3, 1))
    assert_size_stride(arg3_1, (11, 3), (3, 1))
    assert_size_stride(arg4_1, (11, 3), (3, 1))
    assert_size_stride(arg5_1, (11, 3), (3, 1))
    assert_size_stride(arg6_1, (11, 3), (3, 1))
    assert_size_stride(arg7_1, (11, 3), (3, 1))
    assert_size_stride(arg8_1, (11, 3), (3, 1))
    assert_size_stride(arg9_1, (11, 3), (3, 1))
    assert_size_stride(arg10_1, (11, 3), (3, 1))
    assert_size_stride(arg11_1, (11, 3), (3, 1))
    assert_size_stride(arg12_1, (1, 12, 832, 64), (638976, 64, 768, 1))
    assert_size_stride(arg13_1, (1, 13, 64), (832, 64, 1))
    assert_size_stride(arg14_1, (1, 12, 832, 64), (638976, 64, 768, 1))
    assert_size_stride(arg15_1, (1, 12, 832, 64), (638976, 64, 768, 1))
    assert_size_stride(arg16_1, (1, 1, 1, 832), (832, 832, 832, 1))
    assert_size_stride(arg17_1, (1, 1, 9, 64, 192), (110592, 110592, 12288, 192, 1))
    assert_size_stride(arg18_1, (1, 1, 832, 1), (832, 832, 1, 1))
    buf0 = empty_strided_cpu((12, 64, 832), (53248, 832, 1), torch.bfloat16)
    # Source Nodes: [bmm], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(arg12_1, (12, 64, 64), (64, 768, 1), 0), reinterpret_tensor(arg14_1, (12, 64, 832), (64, 1, 768), 0), out=buf0)
    buf1 = empty_strided_cpu((1, 12, 64, 1), (768, 64, 1, 768), torch.float32)
    buf2 = empty_strided_cpu((1, 12, 64, 832), (638976, 53248, 832, 1), torch.float32)
    buf3 = empty_strided_cpu((1, 12, 64, 1), (768, 64, 1, 768), torch.float32)
    buf4 = empty_strided_cpu((1, 12, 64, 832), (638976, 53248, 832, 1), torch.bfloat16)
    cpp_fused__softmax_add_mul_rsub_0(buf0, arg16_1, buf1, buf2, buf3, buf4)
    buf5 = empty_strided_cpu((12, 64, 64), (4096, 64, 1), torch.bfloat16)
    # Source Nodes: [bmm_1], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf4, (12, 64, 832), (53248, 832, 1), 0), reinterpret_tensor(arg15_1, (12, 832, 64), (64, 768, 1), 0), out=buf5)
    buf18 = empty_strided_cpu((132, 3), (3, 1), torch.int32)
    buf6 = reinterpret_tensor(buf18, (11, 3), (3, 1), 0)  # alias
    buf7 = reinterpret_tensor(buf18, (11, 3), (3, 1), 33)  # alias
    buf8 = reinterpret_tensor(buf18, (11, 3), (3, 1), 66)  # alias
    buf9 = reinterpret_tensor(buf18, (11, 3), (3, 1), 99)  # alias
    buf10 = reinterpret_tensor(buf18, (11, 3), (3, 1), 132)  # alias
    buf11 = reinterpret_tensor(buf18, (11, 3), (3, 1), 165)  # alias
    buf12 = reinterpret_tensor(buf18, (11, 3), (3, 1), 198)  # alias
    buf13 = reinterpret_tensor(buf18, (11, 3), (3, 1), 231)  # alias
    buf14 = reinterpret_tensor(buf18, (11, 3), (3, 1), 264)  # alias
    buf15 = reinterpret_tensor(buf18, (11, 3), (3, 1), 297)  # alias
    buf16 = reinterpret_tensor(buf18, (11, 3), (3, 1), 330)  # alias
    buf17 = reinterpret_tensor(buf18, (11, 3), (3, 1), 363)  # alias
    buf19 = empty_strided_cpu((12, 11, 3), (33, 3, 1), torch.int64)
    buf25 = empty_strided_cpu((1, 12, 448, 64), (344064, 28672, 64, 1), torch.bfloat16)
    buf20 = reinterpret_tensor(buf25, (1, 12, 64, 64), (344064, 28672, 64, 1), 0)  # alias
    buf78 = empty_strided_cpu((1, 12, 448, 64), (344064, 28672, 64, 1), torch.bfloat16)
    buf73 = reinterpret_tensor(buf78, (1, 12, 64, 64), (344064, 28672, 64, 1), 0)  # alias
    buf21 = reinterpret_tensor(buf25, (1, 12, 64, 64), (344064, 28672, 64, 1), 4096)  # alias
    buf22 = reinterpret_tensor(buf25, (1, 12, 64, 64), (344064, 28672, 64, 1), 8192)  # alias
    buf23 = reinterpret_tensor(buf25, (1, 12, 64, 64), (344064, 28672, 64, 1), 12288)  # alias
    buf76 = reinterpret_tensor(buf78, (1, 12, 64, 64), (344064, 28672, 64, 1), 12288)  # alias
    buf24 = reinterpret_tensor(buf25, (1, 12, 192, 64), (344064, 28672, 64, 1), 16384)  # alias
    buf77 = reinterpret_tensor(buf78, (1, 12, 192, 64), (344064, 28672, 64, 1), 16384)  # alias
    cpp_fused__to_copy_cat_stack_1(arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, buf18, arg14_1, buf6, buf7, buf8, buf9, buf10, buf11, buf12, buf13, buf14, buf15, buf16, buf17, buf19, buf20, buf73, buf21, buf22, buf23, buf76, buf24, buf77)
    del arg0_1
    del arg10_1
    del arg11_1
    del arg1_1
    del arg2_1
    del arg3_1
    del arg4_1
    del arg5_1
    del arg6_1
    del arg7_1
    del arg8_1
    del arg9_1
    del buf10
    del buf11
    del buf12
    del buf13
    del buf14
    del buf15
    del buf16
    del buf17
    del buf18
    del buf20
    del buf21
    del buf22
    del buf23
    del buf24
    del buf6
    del buf7
    del buf8
    del buf9
    buf26 = empty_strided_cpu((12, 64, 448), (28672, 448, 1), torch.bfloat16)
    # Source Nodes: [bmm_2], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(arg12_1, (12, 64, 64), (64, 768, 1), 49152), reinterpret_tensor(buf25, (12, 64, 448), (28672, 1, 64), 0), out=buf26)
    buf30 = empty_strided_cpu((1, 1, 1, 448), (448, 448, 448, 1), torch.float32)
    buf27 = reinterpret_tensor(buf30, (1, 1, 1, 192), (448, 448, 448, 1), 0)  # alias
    buf28 = reinterpret_tensor(buf30, (1, 1, 1, 64), (448, 448, 448, 1), 192)  # alias
    buf29 = reinterpret_tensor(buf30, (1, 1, 1, 192), (448, 448, 448, 1), 256)  # alias
    buf33 = empty_strided_cpu((1, 12, 64, 448), (344064, 28672, 448, 1), torch.float32)
    buf31 = reinterpret_tensor(buf33, (1, 12, 64, 256), (344064, 28672, 448, 1), 0)  # alias
    buf32 = reinterpret_tensor(buf33, (1, 12, 64, 192), (344064, 28672, 448, 1), 256)  # alias
    buf86 = empty_strided_cpu((1, 12, 64, 448), (344064, 28672, 448, 1), torch.float32)
    buf85 = reinterpret_tensor(buf86, (1, 12, 64, 192), (344064, 28672, 448, 1), 256)  # alias
    buf34 = buf3; del buf3  # reuse
    buf35 = empty_strided_cpu((1, 12, 64, 448), (344064, 28672, 448, 1), torch.float32)
    buf36 = buf1; del buf1  # reuse
    buf43 = reinterpret_tensor(buf25, (1, 12, 64, 448), (344064, 28672, 448, 1), 0); del buf25  # reuse
    buf42 = empty_strided_cpu((1, 12, 448, 64), (344064, 28672, 64, 1), torch.bfloat16)
    buf37 = reinterpret_tensor(buf42, (1, 12, 64, 64), (344064, 28672, 64, 1), 0)  # alias
    buf95 = empty_strided_cpu((1, 12, 448, 64), (344064, 28672, 64, 1), torch.bfloat16)
    buf90 = reinterpret_tensor(buf95, (1, 12, 64, 64), (344064, 28672, 64, 1), 0)  # alias
    buf38 = reinterpret_tensor(buf42, (1, 12, 64, 64), (344064, 28672, 64, 1), 4096)  # alias
    buf39 = reinterpret_tensor(buf42, (1, 12, 64, 64), (344064, 28672, 64, 1), 8192)  # alias
    buf40 = reinterpret_tensor(buf42, (1, 12, 64, 64), (344064, 28672, 64, 1), 12288)  # alias
    buf93 = reinterpret_tensor(buf95, (1, 12, 64, 64), (344064, 28672, 64, 1), 12288)  # alias
    buf41 = reinterpret_tensor(buf42, (1, 12, 192, 64), (344064, 28672, 64, 1), 16384)  # alias
    buf94 = reinterpret_tensor(buf95, (1, 12, 192, 64), (344064, 28672, 64, 1), 16384)  # alias
    cpp_fused__softmax_add_cat_minimum_mul_new_ones_rsub_2(arg16_1, arg13_1, buf19, buf26, buf30, buf33, arg15_1, buf27, buf28, buf29, buf31, buf32, buf85, buf34, buf35, buf36, buf43, buf37, buf90, buf38, buf39, buf40, buf93, buf41, buf94)
    del buf26
    del buf27
    del buf28
    del buf29
    del buf31
    del buf32
    del buf33
    del buf37
    del buf38
    del buf39
    del buf40
    del buf41
    buf44 = empty_strided_cpu((12, 64, 64), (4096, 64, 1), torch.bfloat16)
    # Source Nodes: [bmm_3], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf43, (12, 64, 448), (28672, 448, 1), 0), reinterpret_tensor(buf42, (12, 448, 64), (28672, 64, 1), 0), out=buf44)
    del buf42
    buf45 = empty_strided_cpu((12, 576, 64), (36864, 64, 1), torch.bfloat16)
    # Source Nodes: [first_band_product], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(arg12_1, (12, 576, 64), (64, 768, 1), 98304), reinterpret_tensor(arg14_1, (12, 64, 64), (64, 1, 768), 0), out=buf45)
    buf49 = empty_strided_cpu((1, 12, 9, 192, 64), (1327104, 110592, 12288, 64, 1), torch.bfloat16)
    buf46 = reinterpret_tensor(buf49, (1, 12, 9, 64, 64), (1327104, 110592, 12288, 64, 1), 0)  # alias
    buf47 = reinterpret_tensor(buf49, (1, 12, 9, 64, 64), (1327104, 110592, 12288, 64, 1), 4096)  # alias
    buf48 = reinterpret_tensor(buf49, (1, 12, 9, 64, 64), (1327104, 110592, 12288, 64, 1), 8192)  # alias
    buf50 = empty_strided_cpu((1, 12, 9, 64, 64), (442368, 36864, 4096, 64, 1), torch.bfloat16)
    cpp_fused_cat_clone_3(arg14_1, arg12_1, buf46, buf47, buf48, buf50)
    del buf46
    del buf47
    del buf48
    buf51 = empty_strided_cpu((108, 64, 192), (12288, 192, 1), torch.bfloat16)
    # Source Nodes: [bmm_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf50, (108, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf49, (108, 64, 192), (12288, 1, 64), 0), out=buf51)
    buf52 = buf49; del buf49  # reuse
    buf69 = empty_strided_cpu((1, 12, 9, 192, 64), (1327104, 110592, 12288, 64, 1), torch.bfloat16)
    cpp_fused_clone_4(buf19, arg14_1, arg15_1, buf52, buf69)
    buf53 = empty_strided_cpu((108, 64, 192), (12288, 192, 1), torch.bfloat16)
    # Source Nodes: [bmm_5], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf50, (108, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf52, (108, 64, 192), (12288, 1, 64), 0), out=buf53)
    buf54 = reinterpret_tensor(buf50, (12, 576, 64), (36864, 64, 1), 0); del buf50  # reuse
    # Source Nodes: [last_band_product], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(arg12_1, (12, 576, 64), (64, 768, 1), 98304), reinterpret_tensor(arg14_1, (12, 64, 64), (64, 1, 768), 589824), out=buf54)
    buf59 = empty_strided_cpu((1, 12, 9, 64, 512), (3538944, 294912, 32768, 512, 1), torch.bfloat16)
    buf55 = reinterpret_tensor(buf59, (1, 12, 9, 64, 64), (3538944, 294912, 32768, 512, 1), 0)  # alias
    buf56 = reinterpret_tensor(buf59, (1, 12, 9, 64, 192), (3538944, 294912, 32768, 512, 1), 64)  # alias
    buf57 = reinterpret_tensor(buf59, (1, 12, 9, 64, 192), (3538944, 294912, 32768, 512, 1), 256)  # alias
    buf58 = reinterpret_tensor(buf59, (1, 12, 9, 64, 64), (3538944, 294912, 32768, 512, 1), 448)  # alias
    buf60 = empty_strided_cpu((1, 12, 9, 64, 1), (6912, 576, 64, 1, 6912), torch.float32)
    buf61 = empty_strided_cpu((1, 12, 9, 64, 512), (3538944, 294912, 32768, 512, 1), torch.float32)
    buf62 = empty_strided_cpu((1, 12, 9, 64, 1), (6912, 576, 64, 1, 6912), torch.float32)
    buf67 = empty_strided_cpu((1, 12, 9, 64, 512), (3538944, 294912, 32768, 512, 1), torch.bfloat16)
    buf66 = buf52; del buf52  # reuse
    buf63 = reinterpret_tensor(buf66, (1, 12, 9, 64, 64), (1327104, 110592, 12288, 64, 1), 0)  # alias
    buf64 = reinterpret_tensor(buf66, (1, 12, 9, 64, 64), (1327104, 110592, 12288, 64, 1), 4096)  # alias
    buf65 = reinterpret_tensor(buf66, (1, 12, 9, 64, 64), (1327104, 110592, 12288, 64, 1), 8192)  # alias
    cpp_fused__softmax__to_copy_add_cat_mul_rsub_5(buf45, arg16_1, buf51, arg17_1, buf53, arg13_1, buf19, buf54, buf59, arg15_1, buf55, buf56, buf57, buf58, buf60, buf61, buf62, buf67, buf63, buf64, buf65)
    del arg13_1
    del arg17_1
    del buf51
    del buf53
    del buf55
    del buf56
    del buf57
    del buf58
    del buf59
    del buf60
    del buf61
    del buf62
    del buf63
    del buf64
    del buf65
    buf68 = reinterpret_tensor(buf54, (108, 64, 64), (4096, 64, 1), 0); del buf54  # reuse
    # Source Nodes: [bmm_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf67, (108, 64, 192), (32768, 512, 1), 64), reinterpret_tensor(buf66, (108, 192, 64), (12288, 64, 1), 0), out=buf68)
    del buf66
    buf70 = reinterpret_tensor(buf45, (108, 64, 64), (4096, 64, 1), 0); del buf45  # reuse
    # Source Nodes: [bmm_7], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf67, (108, 64, 192), (32768, 512, 1), 256), reinterpret_tensor(buf69, (108, 192, 64), (12288, 64, 1), 0), out=buf70)
    del buf69
    buf71 = empty_strided_cpu((12, 576, 64), (36864, 64, 1), torch.bfloat16)
    # Source Nodes: [einsum_3], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf67, (12, 576, 64), (294912, 512, 1), 0), reinterpret_tensor(arg15_1, (12, 64, 64), (64, 768, 1), 0), out=buf71)
    buf72 = empty_strided_cpu((12, 576, 64), (36864, 64, 1), torch.bfloat16)
    # Source Nodes: [einsum_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf67, (12, 576, 64), (294912, 512, 1), 448), reinterpret_tensor(arg15_1, (12, 64, 64), (64, 768, 1), 589824), out=buf72)
    del buf67
    buf74 = reinterpret_tensor(buf78, (1, 12, 64, 64), (344064, 28672, 64, 1), 4096)  # alias
    buf75 = reinterpret_tensor(buf78, (1, 12, 64, 64), (344064, 28672, 64, 1), 8192)  # alias
    cpp_fused_cat_6(arg14_1, buf74, buf75)
    del buf73
    del buf74
    del buf75
    del buf76
    del buf77
    buf79 = reinterpret_tensor(buf43, (12, 64, 448), (28672, 448, 1), 0); del buf43  # reuse
    # Source Nodes: [bmm_8], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(arg12_1, (12, 64, 64), (64, 768, 1), 540672), reinterpret_tensor(buf78, (12, 64, 448), (28672, 1, 64), 0), out=buf79)
    buf83 = buf30; del buf30  # reuse
    buf80 = reinterpret_tensor(buf83, (1, 1, 1, 64), (448, 448, 448, 1), 0)  # alias
    buf81 = reinterpret_tensor(buf83, (1, 1, 1, 192), (448, 448, 448, 1), 64)  # alias
    buf82 = reinterpret_tensor(buf83, (1, 1, 1, 192), (448, 448, 448, 1), 256)  # alias
    buf84 = reinterpret_tensor(buf86, (1, 12, 64, 256), (344064, 28672, 448, 1), 0)  # alias
    buf87 = buf36; del buf36  # reuse
    buf88 = buf35; del buf35  # reuse
    buf89 = buf34; del buf34  # reuse
    buf96 = reinterpret_tensor(buf78, (1, 12, 64, 448), (344064, 28672, 448, 1), 0); del buf78  # reuse
    buf91 = reinterpret_tensor(buf95, (1, 12, 64, 64), (344064, 28672, 64, 1), 4096)  # alias
    buf92 = reinterpret_tensor(buf95, (1, 12, 64, 64), (344064, 28672, 64, 1), 8192)  # alias
    cpp_fused__softmax_add_cat_minimum_mul_rsub_7(arg16_1, buf79, buf83, buf86, arg15_1, buf80, buf81, buf82, buf84, buf87, buf88, buf89, buf96, buf91, buf92)
    del buf79
    del buf80
    del buf81
    del buf82
    del buf83
    del buf84
    del buf85
    del buf86
    del buf88
    del buf90
    del buf91
    del buf92
    del buf93
    del buf94
    buf97 = empty_strided_cpu((12, 64, 64), (4096, 64, 1), torch.bfloat16)
    # Source Nodes: [bmm_9], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf96, (12, 64, 448), (28672, 448, 1), 0), reinterpret_tensor(buf95, (12, 448, 64), (28672, 64, 1), 0), out=buf97)
    del buf95
    del buf96
    buf98 = reinterpret_tensor(buf4, (12, 64, 832), (53248, 832, 1), 0); del buf4  # reuse
    # Source Nodes: [bmm_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(arg12_1, (12, 64, 64), (64, 768, 1), 589824), reinterpret_tensor(arg14_1, (12, 64, 832), (64, 1, 768), 0), out=buf98)
    del arg12_1
    del arg14_1
    buf99 = buf89; del buf89  # reuse
    buf100 = buf2; del buf2  # reuse
    buf101 = buf87; del buf87  # reuse
    buf102 = reinterpret_tensor(buf0, (1, 12, 64, 832), (638976, 53248, 832, 1), 0); del buf0  # reuse
    cpp_fused__softmax_add_mul_rsub_8(buf98, arg16_1, buf99, buf100, buf101, buf102)
    del arg16_1
    del buf101
    del buf98
    del buf99
    buf103 = empty_strided_cpu((12, 64, 64), (4096, 64, 1), torch.bfloat16)
    # Source Nodes: [bmm_11], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf102, (12, 64, 832), (53248, 832, 1), 0), reinterpret_tensor(arg15_1, (12, 832, 64), (64, 768, 1), 0), out=buf103)
    del arg15_1
    buf109 = reinterpret_tensor(buf102, (1, 12, 13, 64, 64), (638976, 53248, 4096, 64, 1), 0); del buf102  # reuse
    buf104 = reinterpret_tensor(buf109, (1, 12, 1, 64, 64), (638976, 53248, 4096, 64, 1), 0)  # alias
    buf105 = reinterpret_tensor(buf109, (1, 12, 1, 64, 64), (638976, 53248, 4096, 64, 1), 4096)  # alias
    buf106 = reinterpret_tensor(buf109, (1, 12, 9, 64, 64), (638976, 53248, 4096, 64, 1), 8192)  # alias
    buf107 = reinterpret_tensor(buf109, (1, 12, 1, 64, 64), (638976, 53248, 4096, 64, 1), 45056)  # alias
    buf108 = reinterpret_tensor(buf109, (1, 12, 1, 64, 64), (638976, 53248, 4096, 64, 1), 49152)  # alias
    buf110 = reinterpret_tensor(buf100, (1, 12, 832, 64), (638976, 53248, 64, 1), 0); del buf100  # reuse
    cpp_fused_cat_mul_9(buf5, buf44, buf68, buf70, buf71, buf72, buf97, buf103, buf109, arg18_1, buf104, buf105, buf106, buf107, buf108, buf110)
    del arg18_1
    return (reinterpret_tensor(buf110, (1, 832, 12, 64), (638976, 64, 53248, 1), 0), reinterpret_tensor(buf19, (1, 12, 11, 3), (396, 33, 3, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((11, 3), (3, 1), device='cpu', dtype=torch.int32)
    arg1_1 = rand_strided((11, 3), (3, 1), device='cpu', dtype=torch.int32)
    arg2_1 = rand_strided((11, 3), (3, 1), device='cpu', dtype=torch.int32)
    arg3_1 = rand_strided((11, 3), (3, 1), device='cpu', dtype=torch.int32)
    arg4_1 = rand_strided((11, 3), (3, 1), device='cpu', dtype=torch.int32)
    arg5_1 = rand_strided((11, 3), (3, 1), device='cpu', dtype=torch.int32)
    arg6_1 = rand_strided((11, 3), (3, 1), device='cpu', dtype=torch.int32)
    arg7_1 = rand_strided((11, 3), (3, 1), device='cpu', dtype=torch.int32)
    arg8_1 = rand_strided((11, 3), (3, 1), device='cpu', dtype=torch.int32)
    arg9_1 = rand_strided((11, 3), (3, 1), device='cpu', dtype=torch.int32)
    arg10_1 = rand_strided((11, 3), (3, 1), device='cpu', dtype=torch.int32)
    arg11_1 = rand_strided((11, 3), (3, 1), device='cpu', dtype=torch.int32)
    arg12_1 = rand_strided((1, 12, 832, 64), (638976, 64, 768, 1), device='cpu', dtype=torch.bfloat16)
    arg13_1 = rand_strided((1, 13, 64), (832, 64, 1), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((1, 12, 832, 64), (638976, 64, 768, 1), device='cpu', dtype=torch.bfloat16)
    arg15_1 = rand_strided((1, 12, 832, 64), (638976, 64, 768, 1), device='cpu', dtype=torch.bfloat16)
    arg16_1 = rand_strided((1, 1, 1, 832), (832, 832, 832, 1), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((1, 1, 9, 64, 192), (110592, 110592, 12288, 192, 1), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((1, 1, 832, 1), (832, 832, 1, 1), device='cpu', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('hf_BigBird', benchmark_compiled_module)

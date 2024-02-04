
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

from torch import device, empty, empty_strided
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


cpp_fused_clone_0 = async_compile.cpp_pybinding(['const bfloat16*', 'bfloat16*', 'bfloat16*'], '''
#include "/tmp/torchinductor_jianan/kf/ckfqpz6yp2sujhwvtvlb2vb43nqje6bvriedz3vj5dms52hfmvis.h"
extern "C" void kernel(const bfloat16* in_ptr0,
                       bfloat16* out_ptr0,
                       bfloat16* out_ptr1)
{
    #pragma omp parallel num_threads(56)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(32L))
                        {
                            auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (2304L*x2) + (2359296L*x0)), 32);
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (786432L*x0)), 32);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(32L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(32L))
                    {
                        bfloat16 tmp0[32*32] __attribute__ ((aligned (32)));
                        at::vec::transpose_mxn<bfloat16,32,32>(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (2359296L*x0)), static_cast<long>(2304L), tmp0, 32);
                        for (long x1_inner = 0; x1_inner < 32; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<bfloat16>::loadu(tmp0 + static_cast<long>(32L*x1_inner), 32);
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (1024L*x1) + (1024L*x1_inner) + (786432L*x0)), 32);
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_div_full_where_1 = async_compile.cpp_pybinding(['const bool*', 'const bfloat16*', 'const bfloat16*', 'float*', 'float*', 'float*', 'bfloat16*', 'bfloat16*'], '''
#include "/tmp/torchinductor_jianan/kf/ckfqpz6yp2sujhwvtvlb2vb43nqje6bvriedz3vj5dms52hfmvis.h"
extern "C" void kernel(const bool* in_ptr0,
                       const bfloat16* in_ptr1,
                       const bfloat16* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       bfloat16* out_ptr3,
                       bfloat16* out_ptr4)
{
    #pragma omp parallel num_threads(56)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    auto buffer = std::make_unique<float []>(1024);
                    float* buffer_data = buffer.get();
                    {
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(16L))
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (1024L*x1)));
                            auto tmp1 = at::vec::Vectorized<bfloat16>::loadu(in_ptr1 + static_cast<long>(x2 + (1024L*x1) + (1048576L*x0)), 16);
                            auto tmp2 = cvt_lowp_fp_to_fp32<bfloat16>(tmp1);
                            auto tmp3 = static_cast<float>(8.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 / tmp4;
                            auto tmp6 = static_cast<float>(-3.3895313892515355e+38);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = decltype(tmp5)::blendv(tmp7, tmp5, tmp0);
                            auto tmp9 = (tmp8);
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp9);
                            tmp8.store(buffer_data + x2);
                        }
                        float max_val = at::vec::vec_reduce_all([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec);
                        at::vec::Vectorized<float> sum_fvec = at::vec::Vectorized<float>(float(0));                  
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(16L))
                        {
                            at::vec::Vectorized<float> data_fvec = (at::vec::Vectorized<float>::loadu(buffer_data + x2) - at::vec::Vectorized<float>(max_val)).exp();
                            sum_fvec += data_fvec;
                            data_fvec.store(buffer_data + x2);
                        }
                        float sum_val = at::vec::vec_reduce_all([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, sum_fvec);                                                                     
                        sum_val = 1.0 / sum_val;
                        int64_t d2 = 0;
                        for (; d2 < 1024; d2 += 16) {
                            at::vec::Vectorized<float> out_fvec0 = at::vec::Vectorized<float>::loadu(buffer_data + d2) * at::vec::Vectorized<float>(sum_val);
                            at::vec::Vectorized<bfloat16> out_vec = cvt_fp32_to_lowp_fp<bfloat16>(out_fvec0);
                            out_vec.store(out_ptr3 + static_cast<long>(d2 + (1024L*x1) + (1048576L*x0)), 16);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(32L))
                        {
                            auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(in_ptr2 + static_cast<long>(1536L + x3 + (64L*x1) + (2304L*x2) + (2359296L*x0)), 32);
                            tmp0.store(out_ptr4 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (786432L*x0)), 32);
                        }
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
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 1024, 2304), (2359296, 2304, 1))
    assert_size_stride(arg1_1, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    buf0 = empty((4, 12, 1024, 64), device='cpu', dtype=torch.bfloat16)
    buf1 = empty((4, 12, 64, 1024), device='cpu', dtype=torch.bfloat16)
    cpp_fused_clone_0(arg0_1, buf0, buf1)
    buf2 = empty((48, 1024, 1024), device='cpu', dtype=torch.bfloat16)
    # Source Nodes: [attn_weights], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf0, (48, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf1, (48, 64, 1024), (65536, 1024, 1), 0), out=buf2)
    buf3 = empty_strided((4, 12, 1024, 1), (12288, 1024, 1, 49152), device='cpu', dtype=torch.float32)
    buf4 = empty((4, 12, 1024, 1024), device='cpu', dtype=torch.float32)
    buf5 = empty_strided((4, 12, 1024, 1), (12288, 1024, 1, 49152), device='cpu', dtype=torch.float32)
    buf6 = empty((4, 12, 1024, 1024), device='cpu', dtype=torch.bfloat16)
    buf7 = reinterpret_tensor(buf1, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf1  # reuse
    # print("buf2.size() is: {}".format(buf2.size()), flush=True)
    # print("buf2.stride() is: {}".format(buf2.stride()), flush=True)
    cpp_fused__softmax_clone_div_full_where_1(arg1_1, buf2, arg0_1, buf3, buf4, buf5, buf6, buf7)
    del arg0_1
    del arg1_1
    del buf2
    del buf3
    del buf4
    del buf5
    buf8 = reinterpret_tensor(buf0, (48, 1024, 64), (65536, 64, 1), 0); del buf0  # reuse
    # Source Nodes: [attn_output], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf6, (48, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf7, (48, 1024, 64), (65536, 64, 1), 0), out=buf8)
    return (reinterpret_tensor(buf8, (4, 12, 1024, 64), (786432, 65536, 64, 1), 0), buf6, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 1024, 2304), (2359296, 2304, 1), device='cpu', dtype=torch.bfloat16)
    arg1_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)

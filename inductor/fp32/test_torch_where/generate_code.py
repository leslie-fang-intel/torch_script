from ctypes import c_void_p, c_long
import torch
import random
from torch import empty_strided, as_strided, device
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
async_compile = AsyncCompile()


kernel_cpp_0 = async_compile.cpp('''
#include "/tmp/torchinductor_root/77/c7773nj5pwikpmm2pwa62rcudlf7p3if7eyqb5k4sjsvewwje4le.h"
extern "C" void kernel(const bool* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       float* __restrict__ out_ptr0)
{
    {
        for(long i0=0; i0<17; i0+=1)
        {
            std::cout<<"in_ptr0[0] is:"<<in_ptr0[0]<<std::endl;
            auto tmp0 = at::vec::Vectorized<float>(in_ptr0[0]);
            std::cout<<"tmp0 is:"<<tmp0<<std::endl;
            std::cout<<"tmp0 != 0 is:"<<(tmp0 != 0)<<std::endl;
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp2 = at::vec::Vectorized<float>(in_ptr2[0]);

            std::cout<<"tmp1 is:"<<tmp1<<std::endl;
            std::cout<<"tmp2 is:"<<tmp2<<std::endl;
            // auto tmp3 = decltype(tmp1)::blendv(tmp2, tmp1, tmp0 != 0);
            auto tmp3 = decltype(tmp1)::blendv(tmp2, tmp1, tmp0 != 0);
            std::cout<<"tmp3 fduuuie is:"<<tmp3<<std::endl;
            tmp3.store(out_ptr0 + 16*i0);
        }
        #pragma omp simd simdlen(8)
        for(long i0=272; i0<273; i0+=1)
        {
            auto tmp0 = in_ptr0[0];
            auto tmp1 = in_ptr1[i0];
            auto tmp2 = in_ptr2[0];
            auto tmp3 = tmp0 ? tmp1 : tmp2;
            out_ptr0[i0] = tmp3;
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    buf0 = empty_strided((13, 7, 3), (21, 3, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_0(c_void_p(arg0_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(buf0.data_ptr()))
    del arg0_1
    del arg1_1
    del arg2_1
    print("buf0[0, 0] is: {}".format(buf0[0, 0]))
    return (buf0, )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    torch._dynamo.reset()
    torch._inductor.metrics.reset()

    from torch._inductor import codecache
    # codecache.CppCodeCache.clear()
    # arg0_1 = rand_strided((1, 1), (1, 1), device='cpu', dtype=torch.bool)
    arg0_1 = torch.tensor([[True, True], [True, True]], device='cpu', dtype=torch.bool)
    # arg1_1 = rand_strided((13, 7, 3), (21, 3, 1), device='cpu', dtype=torch.float32)
    # arg2_1 = rand_strided((1, 1), (1, 1), device='cpu', dtype=torch.float32)
    torch.manual_seed(0)

    arg1_1 = torch.rand(13, 7, 3)
    arg2_1 = torch.rand(1, 1)
    #print_performance(lambda: call([arg0_1, arg1_1, arg2_1]))
    call([arg0_1, arg1_1, arg2_1])

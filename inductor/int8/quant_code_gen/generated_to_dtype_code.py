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
#include "/tmp/torchinductor_root/dm/cdmaihqxwe73zkb3he2zizktpq5uujetg2db26c3r4lgsmlx3b4c.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       uint8_t* __restrict__ out_ptr0)
{
    {
        for(long i0=0; i0<2352; i0+=1)
        {
            // std::array<at::vec::Vectorized<float>, 4> input;
            // for (long j0=0; j0<4; j0+=1) {
            //    input[j0] = at::vec::Vectorized<float>::loadu(in_ptr0 + 64*i0 + 16*j0);
            //}
            //auto res2 = at::vec::Vectorized<c10::quint8>::convert(input);

            // once 64 float32 (4*_m512) number will be converted to 64 uint8
            auto res2 = at::vec::Vectorized<c10::quint8>::convert(in_ptr0 + 64*i0);
            res2.store(out_ptr0 + 64*i0);
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    buf0 = empty_strided((1, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.uint8)
    kernel_cpp_0(c_void_p(arg0_1.data_ptr()), c_void_p(buf0.data_ptr()))


    print("input of arg0_1 is: {}".format(arg0_1), flush=True)
    print("result of buf0 is: {}".format(buf0), flush=True)
    del arg0_1
    return (buf0, )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    print_performance(lambda: call([arg0_1]))

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
    // Directly test intrinsics quant
    {
        std::array<at::vec::Vectorized<float>, 4> input;
        float scale = 1.0;
        int32_t zero_point = 0;
        float inverse_scale = 1.0;
        for (long j0=0; j0<4; j0+=1) {
            input[j0] = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*j0);
        }
        
        // auto res = at::vec::Vectorized<c10::quint8>::quantize(input, scale, zero_point, inverse_scale);

        for (long j0=0; j0<4; j0+=1) {
            std::cout<<"input idx: "<<j0<<" data: "<<input[j0]<<std::endl;
        }
        auto res2 = at::vec::Vectorized<c10::quint8>::convert(input);
        std::cout<<"res2 is: "<<res2<<std::endl;
    }
    std::cout<<"---- Finish2 the test purpose 2345 ----"<<std::endl;

    {
        constexpr auto min_val = std::numeric_limits<typename c10::quint8::underlying>::min();
        constexpr auto max_val = std::numeric_limits<typename c10::quint8::underlying>::max();
        // std::cout<<"min_val is: "<<min_val<<std::endl;
        // std::cout<<"max_val is: "<<max_val<<std::endl;
        for(long i0=0; i0<2352; i0+=1)
        {
            uint8_t quantized_values[64];
            //std::vector<at::vec::Vectorized<uint8_t>> tmp1[4]; 
            for (long j0=0; j0<4; j0+=1) {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 64*i0 + 16*j0);
            }
            // auto tmp1 = (tmp0);
            // tmp1.store(out_ptr0 + 16*i0);
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
    del arg0_1
    return (buf0, )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    print_performance(lambda: call([arg0_1]))


from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from torch._inductor.utils import maybe_profile

from torch import empty_strided, as_strided, device
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
async_compile = AsyncCompile()


kernel_cpp_0 = async_compile.cpp('''
#include "/tmp/torchinductor_root/hw/chwr6vy6e6sd25sfh42qtywkuf2emodexm2aomp3lbrcxwznfwyi.h"
extern "C" void kernel(const unsigned char* in_ptr0,
                       float* out_ptr0)
{
    {
        // 9408 = 150528 /16
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0); i0<static_cast<long>(9408); i0+=static_cast<long>(1))
        {
            
            std::cout<<"---- start a round22 ----"<<std::endl;
            auto tmp = in_ptr0 + (static_cast<long>(i0)) * 16;
            auto output_tmp = out_ptr0 + (static_cast<long>(i0)) * 16;
            at::vec::Vectorized<float> res = at::vec::Vectorized<uint8_t>::convert_to_float(tmp);
            res.store(output_tmp);

            

            //auto tmp0 = in_ptr0[static_cast<long>(i0)];
            //auto tmp1 = static_cast<float>(tmp0);
            //out_ptr0[static_cast<long>(i0)] = tmp1;
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    buf0 = empty_strided((1, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_0(c_void_p(arg0_1.data_ptr()), c_void_p(buf0.data_ptr()))
    print("arg0_1 is: {}".format(arg0_1), flush=True)
    print("buf0 is: {}".format(buf0), flush=True)
    diff = torch.max(torch.abs((buf0-arg0_1)))
    print(diff < 0.01, flush=True)
    del arg0_1
    return (buf0, )


def benchmark_compiled_module():
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32) * 100
    arg0_1 = arg0_1.to(torch.uint8)
    #print_performance(lambda: call([arg0_1]))
    call([arg0_1])


if __name__ == "__main__":
    import argparse
    from torch._inductor.utils import benchmark_all_kernels

    import numpy as np
    local_seed = 2023
    torch.manual_seed(local_seed) # Set PyTorch seed
    np.random.seed(seed=local_seed) # Set Numpy seed
    random.seed(local_seed) # Set the Python seed

    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-kernels", "-k", action="store_true", help="Whether to benchmark each individual kernels")
    parser.add_argument("--benchmark-all-configs", "-c", action="store_true", help="Whether to benchmark each individual config for a kernel")
    parser.add_argument("--profile", "-p", action="store_true", help="Whether to profile the compiled module")
    args = parser.parse_args()

    if args.benchmark_kernels:
        benchmark_all_kernels('None', args.benchmark_all_configs)
    else:
        with maybe_profile(args.profile) as p:
            benchmark_compiled_module()

        if p:
            path = f"{tempfile.gettempdir()}/compiled_module_profile.json"
            p.export_chrome_trace(path)
            print(f"Chrome trace for the profile is written to {path}")

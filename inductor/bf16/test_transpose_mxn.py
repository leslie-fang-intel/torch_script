
# CMD: clear && rm -rf /tmp/torchinductor_jianan/* && python test_transpose_mxn.py

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
import numpy as np

local_seed=2024
torch.manual_seed(local_seed) # Set PyTorch seed
np.random.seed(seed=local_seed) # Set Numpy seed
random.seed(local_seed) # Set the Python seed

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()

cpp_fused_native_group_norm_0 = async_compile.cpp('''
#include "/home/jianan/leslie/torch_inductor_lz/pytorch/torch/_inductor/codegen/cpp_prefix.h"
extern "C" void kernel(const bfloat16* in_ptr0,
                       bfloat16* out_ptr0)
{
    int M = 16;
    int N = 16;
    int64_t ld_src = 16;
    int64_t ld_dst = 16;                           
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            out_ptr0[j*ld_dst + i] = in_ptr0[i*ld_src + j];
        }
    }
    return;
}
''')

cpp_fused_native_group_norm_1 = async_compile.cpp('''
#include "/home/jianan/leslie/torch_inductor_lz/pytorch/torch/_inductor/codegen/cpp_prefix.h"
extern "C" void kernel(const bfloat16* in_ptr0,
                       bfloat16* out_ptr0)
{
    at::vec::transpose_mxn<bfloat16,16,16>(in_ptr0, 16, out_ptr0, 16);
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    args.clear()
    from torch._dynamo.testing import rand_strided
    buf0 = rand_strided((16, 16), (16, 1), device='cpu', dtype=torch.bfloat16)
    buf1 = rand_strided((16, 16), (16, 1), device='cpu', dtype=torch.bfloat16)
    buf2 = rand_strided((16, 16), (16, 1), device='cpu', dtype=torch.bfloat16)

    print("buf0 is: {}".format(buf0), flush=True)

    cpp_fused_native_group_norm_0(c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))

    print("buf1 is: {}".format(buf1), flush=True)

    cpp_fused_native_group_norm_1(c_void_p(buf0.data_ptr()), c_void_p(buf2.data_ptr()))

    print("buf2 is: {}".format(buf2), flush=True)

    print(torch.allclose(buf1, buf2, atol=0.01, rtol=0.01), flush=True)
    del buf0
    del buf2
    return buf1


def benchmark_compiled_module(times=1, repeat=1):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((2, 960, 96, 96), (8847360, 1, 92160, 960), device='cpu', dtype=torch.bfloat16)
    fn = lambda: call([primals_1, primals_2, primals_3])
    # return print_performance(fn, times=times, repeat=repeat)
    fn()
    return -1

if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)

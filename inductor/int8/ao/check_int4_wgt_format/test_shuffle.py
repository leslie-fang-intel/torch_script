import torch
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
import time
from torch._export import capture_pre_autograd_graph
from torchao.quantization.quant_api import (
    Int4WeightOnlyQuantizer,
)
import numpy as np
import random
from torch._inductor.async_compile import AsyncCompile

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()

local_seed = 2024

torch.manual_seed(local_seed) # Set PyTorch seed
np.random.seed(seed=local_seed) # Set Numpy seed
random.seed(local_seed) # Set the Python see

M0 = 2
N0 = 16
K0 = 256

check_wgt = async_compile.cpp_pybinding(['const uint16_t*', 'const uint16_t*', 'uint16_t*', 'uint16_t*'], '''
#include "/localdisk/leslie/torch_inductor_community/pytorch/torch/_inductor/codegen/cpp_prefix.h"
#include <iostream>
#include <c10/util/Unroll.h>
                                                                                                                                                                            
extern "C" void kernel(const uint16_t* a, const uint16_t* b, uint16_t* result, uint16_t* result2)
{
    __m512i vec_a = _mm512_loadu_si512(a);
    __m512i vec_b = _mm512_loadu_si512(b);
    uint16_t control[32] = {
        0, 32, 1, 33, 2, 34, 3, 35, 4, 36, 5, 37, 6, 38, 7, 39,
        8, 40, 9, 41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46, 15, 47
    };
    __m512i vec_control = _mm512_loadu_si512(control);

    uint16_t control2[32] = {
        16, 48, 17, 49, 18, 50, 19, 51, 20, 52, 21, 53, 22, 54, 23, 55, 24, 56,
        25, 57, 26, 58, 27, 59, 28, 60, 29, 61, 30, 62, 31, 63
    };
    __m512i vec_control2 = _mm512_loadu_si512(control2);
                                        
    __m512i shuffled = _mm512_permutex2var_epi16(vec_a, vec_control, vec_b);
    __m512i shuffled2 = _mm512_permutex2var_epi16(vec_a, vec_control2, vec_b);
    
    _mm512_storeu_si512(result, shuffled);
    _mm512_storeu_si512(result2, shuffled2);

                                        
}
''')

async_compile.wait(globals())
del async_compile


def test_shuffle():
    with torch.no_grad():
        a = torch.arange(32, dtype=torch.int32).to(torch.uint16)
        b = (torch.arange(32, dtype=torch.int32) + 32).to(torch.uint16)
        c = torch.zeros(32, dtype=torch.uint16)
        d = torch.zeros(32, dtype=torch.uint16)
        print("---- a is: {}".format(a), flush=True)
        print("---- b is: {}".format(b), flush=True)
        check_wgt(a, b, c, d)
        print("---- c is: {}".format(c), flush=True)
        print("---- d is: {}".format(d), flush=True)

if __name__ == "__main__":
    test_shuffle()

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
    uint16_t control[16] = {
        0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23,
    };
    __m256i vec_control = _mm256_loadu_si256((__m256i*)control);

    uint16_t control2[16] = {
        8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31
    };
    __m256i vec_control2 = _mm256_loadu_si256((__m256i*)control2);
                                        

    __m256i higher_256 = _mm512_extracti64x4_epi64(vec_a, 0);
    __m256i lower_256 = _mm512_extracti64x4_epi64(vec_a, 1);

    __m256i shuffled = _mm256_permutex2var_epi16(higher_256, vec_control, lower_256);
    __m256i shuffled2 = _mm256_permutex2var_epi16(higher_256, vec_control2, lower_256);
    
    _mm256_storeu_si256((__m256i*)result, shuffled);
    _mm256_storeu_si256((__m256i*)result2, shuffled2);

                                        
}
''')

async_compile.wait(globals())
del async_compile


def test_shuffle():
    with torch.no_grad():
        a = torch.arange(32, dtype=torch.int32).to(torch.bfloat16)
        b = (torch.arange(32, dtype=torch.int32) + 32).to(torch.bfloat16)
        c = torch.zeros(16, dtype=torch.bfloat16)
        d = torch.zeros(16, dtype=torch.bfloat16)
        print("---- a is: {}".format(a), flush=True)
        print("---- b is: {}".format(b), flush=True)
        check_wgt(a, b, c, d)
        print("---- c is: {}".format(c), flush=True)
        print("---- d is: {}".format(d), flush=True)

if __name__ == "__main__":
    test_shuffle()

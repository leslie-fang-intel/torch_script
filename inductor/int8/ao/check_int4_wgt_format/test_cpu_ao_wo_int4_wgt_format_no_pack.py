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

check_wgt = async_compile.cpp_pybinding(['const unsigned char*', 'long', 'long'], '''
#include "/localdisk/leslie/torch_inductor_community/pytorch/torch/_inductor/codegen/cpp_prefix.h"
#include <iostream>
#include <c10/util/Unroll.h>
                                                                                                                                                                            
extern "C" void kernel(const unsigned char* __restrict__  in_ptr0, long N, long K)
{
    for (int n = 0; n < N; n += 1) {
        for (int k = 0; k < K; k += 1) {
            long idx = (n * K + k) / 2;
            long offset = 1 - (n * K + k) % 2;
            unsigned char val = in_ptr0[idx];
            unsigned char q_val_shift = ((val & (0xF << (offset * 4))) >> (offset * 4));                                        
            std::cout<<"--- n: "<<n<<", k: "<<k<<", val: "<<static_cast<int32_t>(q_val_shift)<<std::endl;
        }
    }

}
''')

async_compile.wait(globals())
del async_compile

class M(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=K0, out_features=N0, bias=False)

    def forward(self, attn_weights):
        attn_weights = self.linear(attn_weights)  
        return attn_weights

lut = (
    -8.0, -7.0, -6.0, -5.0,
    -4.0, -3.0, -2.0, -1.0,
    0.0, 1.0, 2.0, 3.0,
    4.0, 5.0, 6.0, 7.0,
)

# def matrix_mul(x, y):
#     return torch.matmul(x, y)

def check_int4_wgt_wo_pack(ref_w, q_w, scale_zp):
    # pass
    print("ref_w is: {}".format(ref_w.size()), flush=True) # [16, 256]
    print("q_w is: {}".format(q_w.size()), flush=True) # [2, 2, 32, 4]
    print("scale_zp is: {}".format(scale_zp.size()), flush=True) # [1, 16, 2]

    group_size = 256
    N = ref_w.size(0)  # 16
    K = ref_w.size(1)  # 256
    print("---- N is: {}".format(N), flush=True)
    print("---- K is: {}".format(K), flush=True)

    # flatten q_w
    q_w = q_w.view(N * K // 8)
    q_w = q_w.view(torch.uint8)

    dq_b = []

    # Python Check function
    for n in range(N):
        for k in range(K):
            ref_val = ref_w[n][k].item()
            scale = scale_zp[k//group_size][n][0] 
            zp = scale_zp[k//group_size][n][1]

            idx = (n * K + k) // 2
            offset = 1 - (n * K + k) % 2
            q_val = q_w[idx].item()
            q_val_shift = (q_val & (0xF << (offset * 4))) >> (offset * 4)

            print("n is: {}; k is: {}; q_val_shift is: {}".format(n, k, q_val_shift), flush=True)

            dq_val = lut[q_val_shift] * float(scale) + float(zp)
            dq_b.append(dq_val)
            # print("---- ref_val is: {}, dq_val is: {}".format(ref_val, dq_val), flush=True)

    dq_b = torch.tensor(dq_b).view(N, K).transpose(0, 1)

    # C++ Check funtion
    # check_wgt(q_w, N, K)
    return dq_b

def torchao_GPTQ_int4():
    with torch.no_grad():
        x = torch.randn(M0, K0)
        model = M().eval()
        ref_weight = model.linear.weight
        ref_res = model(x)
        qmodel = Int4WeightOnlyQuantizer(device=torch.device("cpu")).quantize(model)
        true_res = qmodel(x)
        dq_b = check_int4_wgt_wo_pack(ref_weight, qmodel.linear.weight, qmodel.linear.scales_and_zeros)
        print("ref_res is: {}".format(ref_res), flush=True)  # FP32
        print("true_res is: {}".format(true_res), flush=True) # WOQ Kernel 
        print("true_res2 is: {}".format(torch.matmul(x, dq_b)), flush=True) # Ref WOQ Kernel
        # print(torch.allclose(ref_res, true_res, rtol=0.01, atol=0.01), flush=True)

if __name__ == "__main__":
    # test_pt2e_quant()
    torchao_GPTQ_int4()

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

ref_int4_gemm = async_compile.cpp_pybinding(['const bfloat16*', 'const int*', 'const bfloat16*', 'bfloat16*', 'long', 'long', 'long', 'long'], '''
#include "/localdisk/leslie/torch_inductor_community/pytorch/torch/_inductor/codegen/cpp_prefix.h"
#include <iostream>
#include <c10/util/Unroll.h>
                                            
extern "C" void kernel(
    const bfloat16*   in_ptr0,
    const int*   in_ptr1,
    const bfloat16*  ScaleAndZeros,
    bfloat16*  out_ptr0,
    long qGroupSize,
    long M,
    long N,
    long K)
{

    static constexpr float lut[16] = {
        -8.0f, -7.0f, -6.0f, -5.0f,
        -4.0f, -3.0f, -2.0f, -1.0f,
        0.0f, 1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f, 7.0f
    };

    std::cout<<"in_ptr 2 is: "<<in_ptr1[2]<<std::endl;
    std::cout<<"in_ptr 7 is: "<<in_ptr1[7]<<std::endl;
    std::cout<<"in_ptr 8 is: "<<in_ptr1[8]<<std::endl;
                                                                                     
    const unsigned char* in_ptr1_cast = reinterpret_cast<const unsigned char*>(in_ptr1);
                                            
    for (int m = 0; m < M; m += 1) {
        for (int n = 0; n < N; n += 1) {
            float c_val = 0;
            for (int k = 0; k < K; k += 1) {

                // Get the scale and zp
                int kb = k / qGroupSize;  
                                            
                const auto scale = static_cast<float>(ScaleAndZeros[kb * N * 2 + n * 2]);
                const auto zero = static_cast<float>(ScaleAndZeros[kb * N * 2 + n * 2 + 1]);
                
                // Get the val of matrix A as float
                const auto a_val = static_cast<float>(in_ptr0[m * K + k]);

                // Get the val of matrix B as float
                long idx = (n * K + k) / 2;
                long offset = 1 - (n * K + k) % 2;
                const unsigned char val = in_ptr1_cast[idx];
                int index = ((val & (0xF << (offset * 4))) >> (offset * 4));
                const bfloat16 b_val = static_cast<bfloat16>(lut[index] * scale + zero);


                // std::cout<<"---- n is: "<<n<<" k is: "<<k<<" val is: "<<static_cast<int>(val)<<" deqaunt_val is: "<<b_val<<std::endl;

                // std::cout<<"---- m: "<<m<<", n: "<<n<<", k: "<<k<<std::endl;
                // std::cout<<"---- A[m * lda + k]: "<<a_val<<std::endl;
                // std::cout<<"---- B[k * ldb + n]: "<<b_val<<std::endl;
                                            
                c_val += a_val * static_cast<float>(b_val);    
                // std::cout<<"---- result: "<<c_val<<std::endl;                                        
            }
            out_ptr0[m * N + n] = c_val;
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

    # # flatten q_w
    # q_w = q_w.view(N * K // 8)
    # q_w = q_w.view(torch.uint8)

    inner_k_tiles = 8

    unpack_b = torch.ops.aten._convert_weight_to_int4unpack(q_w, inner_k_tiles, K0)

    print("----- unpack_b is: {}".format(unpack_b), flush=True)
    print("----- unpack_b size is: {}".format(unpack_b.size()), flush=True)

    def quant_b(unpack_b):
        unpack_b = unpack_b.flatten()
        dq_b = []
        for n in range(N):
            for k in range(K):
                scale = scale_zp[k//group_size][n][0] 
                zp = scale_zp[k//group_size][n][1]

                idx = (n * K + k) // 8
                offset = 7 - (n * K + k) % 8
                q_val = unpack_b[idx].item()
                q_val_shift = (q_val & (0xF << (offset * 4))) >> (offset * 4)

                # print("n is: {}; k is: {}; q_val_shift is: {}".format(n, k, q_val_shift), flush=True)

                dq_val = lut[q_val_shift] * float(scale) + float(zp)
                dq_b.append(dq_val)
        return dq_b

    dq_b = torch.tensor(quant_b(unpack_b)).view(N, K).transpose(0, 1)

    return unpack_b, dq_b

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
        ref_res = model.to(torch.bfloat16)(x.to(torch.bfloat16)).to(torch.float32)
        qmodel = Int4WeightOnlyQuantizer(device=torch.device("cpu")).quantize(model)
        true_res = qmodel(x)
        unpack_b, dq_b = check_int4_wgt_wo_pack(ref_weight, qmodel.linear.weight, qmodel.linear.scales_and_zeros)
        # true_res2 = torch.matmul(x, dq_b)
        
        print("--- start true_res2", flush=True)

        true_res2 = torch.ops.aten._weight_int4pack_mm(
            x.to(torch.bfloat16),
            torch.ops.aten._convert_weight_to_int4pack(unpack_b.view(torch.uint8), 8),
            256, # qGroupSize
            qmodel.linear.scales_and_zeros,
        ).to(torch.float32)

        print("--- start true_res3", flush=True)

        true_res3 = torch.zeros(M0, N0, dtype=torch.bfloat16)
        ref_int4_gemm(
            x.to(torch.bfloat16),
            unpack_b,
            qmodel.linear.scales_and_zeros,
            true_res3,
            256, # qGroupSize
            M0,
            N0,
            K0,
        )
        true_res3 = true_res3.to(torch.float32)

        print("--- finish true_res3", flush=True)
    
        cqmodel = torch.compile(qmodel)
        ctrue_res = cqmodel(x)

        print("ref_res is: {}".format(ref_res), flush=True)  # FP32
        print("true_res is: {}".format(true_res), flush=True) # WOQ Kernel 
        print("true_res2 is: {}".format(true_res2), flush=True) # Ref WOQ Kernel
        print("true_res3 is: {}".format(true_res3), flush=True) # Ref WOQ Kernel
        print(torch.allclose(ref_res, true_res, rtol=0.01, atol=0.01), flush=True)
        print(torch.allclose(ref_res, true_res2, rtol=0.01, atol=0.01), flush=True)
        print(torch.allclose(ref_res, true_res3, rtol=0.01, atol=0.01), flush=True)
        print(torch.allclose(true_res, true_res2, rtol=0.01, atol=0.01), flush=True)
        print(torch.allclose(true_res, true_res3, rtol=0.1, atol=0.01), flush=True)

        print("ctrue_res is: {}".format(ctrue_res), flush=True) # Ref WOQ Kernel
        print(torch.allclose(true_res, ctrue_res, rtol=0.1, atol=0.01), flush=True)

if __name__ == "__main__":
    # test_pt2e_quant()
    torchao_GPTQ_int4()

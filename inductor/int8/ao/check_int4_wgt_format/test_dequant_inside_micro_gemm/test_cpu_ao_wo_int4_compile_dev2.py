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
from torch.testing._internal.common_quantization import (
    _generate_qdq_quantized_model,
    _group_quantize_tensor,
)
from torch._inductor.compile_fx import (
    compile_fx,
    compile_fx_inner,
    complex_memory_overlap,
)
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

# Single Thread: Worse than ATen
# Multi Thread: 
M0 = 2
N0 = 64
K0 = 128  # K should be multiple of 256 GroupSize

# # Single Thread: Worse than ATen
# # Multi Thread: 
# M0 = 2
# N0 = 4096
# K0 = 1024

# # Single Thread: Better than ATen
# # Multi Thread: 
# M0 = 4096
# N0 = 4096
# K0 = 1024 

# # Single Thread: Better
# # Multi Thread: 
# M0 = 1024
# N0 = 16384
# K0 = 4096 

q_group_size = 32
inner_k_tiles = 2

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
                                                                                     
    const unsigned char* in_ptr1_cast = reinterpret_cast<const unsigned char*>(in_ptr1);
    bfloat16* b_vals = std::make_unique<bfloat16[]>(N*K).get();

    for (int k = 0; k < K; k += 1) {        
        for (int n = 0; n < N; n += 1) {

            // Get the scale and zp
            int kb = k / qGroupSize;  
                                        
            const auto scale = static_cast<float>(ScaleAndZeros[kb * N * 2 + n * 2]);
            const auto zero = static_cast<float>(ScaleAndZeros[kb * N * 2 + n * 2 + 1]);

            // Get the val of matrix B as float
            long idx = (n * K + k) / 2;
            long offset = 1 - (n * K + k) % 2;
            const unsigned char val = in_ptr1_cast[idx];
            int index = ((val & (0xF << (offset * 4))) >> (offset * 4));
            const bfloat16 b_val = static_cast<bfloat16>(lut[index] * scale + zero);

            b_vals[n*K+k] = b_val;

            // std::cout<<"---- n is: "<<n<<" k is: "<<k<<" scale is: "<<scale<<" zero is: "<<zero<<" index is: "<<index<<std::endl;
            // std::cout<<"---- n is: "<<n<<" k is: "<<k<<" index is: "<<index<<" val is: "<<static_cast<int>(val)<<" deqaunt_val is: "<<b_val<<std::endl;
                                        
        }
        // out_ptr0[m * N + n] = c_val;
    }

    for (int m = 0; m < M; m += 1) {
        for (int n = 0; n < N; n += 1) {
            float c_val = 0;
            for (int k = 0; k < K; k += 1) {
                const auto a_val = static_cast<float>(in_ptr0[m * K + k]);
                bfloat16 b_val = b_vals[n*K+k];
                c_val += a_val * static_cast<float>(b_val);   
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

    # group_size = 256
    K = ref_w.size(0)  # 16
    N = ref_w.size(1)  # 256
    print("---- t N is: {}".format(N), flush=True)
    print("---- t K is: {}".format(K), flush=True)

    # inner_k_tiles = 8

    # uint8, N * K //2
    unpack_b = torch.ops.aten._convert_weight_to_int4unpack(q_w, inner_k_tiles, K0)

    print("scale_zp is: {}".format(scale_zp.size()), flush=True)

    def quant_b(unpack_b):
        unpack_b = unpack_b.flatten()
        dq_b = []
        for k in range(K):
            for n in range(N):
                scale = scale_zp[k//q_group_size][n][0] 
                zp = scale_zp[k//q_group_size][n][1]

                idx = (n * K + k) // 2
                offset = 1 - (n * K + k) % 2
                q_val = unpack_b[idx].item()
                q_val_shift = (q_val & (0xF << (offset * 4))) >> (offset * 4)

                # print("n is: {}; k is: {}; scale is: {}, zp: {}, q_val_shift:{}".format(n, k, scale, zp, q_val_shift), flush=True)

                dq_val = lut[q_val_shift] * float(scale) + float(zp)
                dq_b.append(dq_val)
        return dq_b

    # N * K
    dq_b = torch.tensor(quant_b(unpack_b)).view(N, K)
    # K * N
    dq_b = dq_b.transpose(0, 1)

    # int32 N * K // 8
    unpack_b = unpack_b.view(torch.int32)
    return unpack_b, dq_b


def torchao_GPTQ_int4():
    with torch.no_grad():

        class WeightOnlyInt4Linear(torch.nn.Module):
            __constants__ = ["in_features", "out_features"]
            in_features: int
            out_features: int
            weight: torch.Tensor
            scales_and_zeros: torch.Tensor

            def __init__(
                self,
                weight: torch.Tensor,
                scales_and_zeros: torch.Tensor,
                groupsize: int = 128,
                inner_k_tiles: int = 8,
            ) -> None:
                super().__init__()

                self.out_features = N0
                self.groupsize = groupsize
                self.inner_k_tiles = inner_k_tiles

                # Shape (out_features // 8, in_features // (inner_k_tiles * 16), 32, inner_k_tiles // 2)
                # dtype is torch.int32
                self.register_buffer(
                    "weight",
                    weight
                    # Shape (in_features // groupsize, out_features, 2)
                    # dtype of scales & zeros is torch.bfloat16
                )
                self.register_buffer("scales_and_zeros", scales_and_zeros)

            def forward(self, input: torch.Tensor) -> torch.Tensor:
                return torch.ops.aten._weight_int4pack_mm(
                    input, self.weight, self.groupsize, self.scales_and_zeros
                )

        torch.manual_seed(1)
        a = torch.randn((M0, K0), dtype=torch.bfloat16)
        b = torch.randn((K0, N0), dtype=torch.bfloat16)

        b_uint8 = None

        def convert_weight_to_int4pack(b):
            nonlocal b_uint8
            b_uint8, b_scales_and_zeros = _group_quantize_tensor(
                b, n_bit=4, q_group_size=q_group_size
            )
            b_int4pack = torch._convert_weight_to_int4pack(b_uint8, inner_k_tiles)

            return b_int4pack, b_scales_and_zeros

        b_int4pack, b_scales_and_zeros = convert_weight_to_int4pack(b)
        ref_res = torch.mm(a, b)
        
        
        
        mod = WeightOnlyInt4Linear(
            b_int4pack, b_scales_and_zeros, q_group_size, inner_k_tiles
        ).eval()
        true_res = mod(a)


        

        unpack_b, dq_b = check_int4_wgt_wo_pack(b, mod.weight, mod.scales_and_zeros)


        true_res2 = torch.ops.aten._weight_int4pack_mm(
            a,
            torch.ops.aten._convert_weight_to_int4pack(unpack_b.view(torch.uint8), 8),
            q_group_size, # qGroupSize
            mod.scales_and_zeros,
        )

        print("--- start true_res3", flush=True)
        true_res3 = torch.zeros(M0, N0, dtype=torch.bfloat16)
        ref_int4_gemm(
            a,
            unpack_b,
            mod.scales_and_zeros,
            true_res3,
            q_group_size, # qGroupSize
            M0,
            N0,
            K0,
        )
        true_res3 = true_res3
        print("--- finish true_res3", flush=True)
    


        cmod = torch.compile(mod)
        true_res4 = cmod(a)

        # print("ref_res is: {}".format(ref_res), flush=True)  # FP32
        print("true_res is: {}".format(true_res), flush=True) # WOQ Kernel 
        # print("true_res2 is: {}".format(true_res2), flush=True) # DQ->RQ-> WOQ Kernel
        print("true_res3 is: {}".format(true_res3), flush=True) # DQ -> Ref WOQ Kernel
        print("true_res4 is: {}".format(true_res4), flush=True) # WOQ GEMM Template
        print(torch.allclose(ref_res, true_res, rtol=0.01, atol=0.01), flush=True)
        print(torch.allclose(true_res, true_res2, rtol=0.1, atol=0.01), flush=True)
        print(torch.allclose(true_res, true_res3, rtol=0.1, atol=0.01), flush=True)
        print(torch.allclose(true_res, true_res4, rtol=0.1, atol=0.01), flush=True)
        # torch.testing.assert_close(true_res, true_res3, rtol=0.1, atol=0.01)
        torch.testing.assert_close(true_res, true_res4, rtol=0.1, atol=0.01)

if __name__ == "__main__":
    # test_pt2e_quant()
    torchao_GPTQ_int4()

# TORCHINDUCTOR_FREEZING=1 TORCH_LOGS="+output_code" numactl -C 0-27 -m 0 python test_amax_sub.py
# * Eager: 5.996412754058838
# * Inductor: 7.722699880599976
# * Inductor Loop fusion: 5.99749493598938

import torch
import time
import random
import numpy as np

local_seed= 2024

torch.manual_seed(local_seed) # Set PyTorch seed
np.random.seed(seed=local_seed) # Set Numpy seed
random.seed(local_seed) # Set the Python seed

class M(torch.nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, attn_weights):
        # return torch.nn.functional.softmax(attn_weights, dim=-1)
        # max = torch.exp(attn_weights)
        
        max = torch.amax(attn_weights, dim=-1, keepdim=True)
        # max = torch.sum(attn_weights, dim=-1, keepdim=True)
        
        return attn_weights - max
 
if __name__ == "__main__":

    with torch.no_grad():
        m = M().eval()
        input = torch.randn(4, 12, 1024, 1024).to(torch.bfloat16)

        m(input)

        warmup_steps = 100
        steps = 1000

        # Refer path
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            ref_res = m(input)

            for _ in range(warmup_steps):
                m(input)

            ref_start = time.time()
            for _ in range(steps):
                m(input)
            ref_end = time.time()

        # Compiler Path
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            c_m = torch.compile(m)
            inductor_res = c_m(input)

            for _ in range(warmup_steps):
                c_m(input)

            inductor_start = time.time()
            for _ in range(steps):
                c_m(input)
            inductor_end = time.time()
        print("ref time is: {}".format(ref_end - ref_start), flush=True)
        # # print("jit time is: {}".format(jit_end - jit_start), flush=True)
        print("inductor time is: {}".format(inductor_end - inductor_start), flush=True)
        print(torch.allclose(ref_res, inductor_res, atol=0.01, rtol=0.01), flush=True)
        # print(torch.allclose(ref_res[1], inductor_res[1], atol=0.01, rtol=0.01), flush=True)




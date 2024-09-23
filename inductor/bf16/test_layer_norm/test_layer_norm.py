# TORCHINDUCTOR_FREEZING=1  TORCH_LOGS="+output_code" numactl -C 56-111 -m 1 python test_layer_norm.py
# Test on 61 system

# BF16 input
# * Eager time: 2.809540271759033
# * Inductor Baseline time: 4.56529688835144

# FP32 Input
# * Eager time: 3.952343463897705
# * Inductor Baseline time: 5.076903581619263

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
        self.layer_norm = torch.nn.LayerNorm(1024, 1e-12)

    def forward(self, input):
        input = self.layer_norm(input)  
        return input
 
if __name__ == "__main__":

    # test_bf16 = True
    test_bf16 = False

    with torch.no_grad():
        m = M().eval()
        input = torch.randn(56, 384, 1024)
        if test_bf16:
            input = input.to(torch.bfloat16)

        m(input)

        warmup_steps = 1000
        steps = 10000

        # Refer path
        with torch.autocast(enabled=test_bf16, device_type="cpu", dtype=torch.bfloat16):
            ref_res = m(input)

            for _ in range(warmup_steps):
                m(input)

            ref_start = time.time()
            for _ in range(steps):
                m(input)
            ref_end = time.time()

        # Compiler Path
        with torch.autocast(enabled=test_bf16, device_type="cpu", dtype=torch.bfloat16):
            c_m = torch.compile(m)
            inductor_res = c_m(input)

            for _ in range(warmup_steps):
                c_m(input)

            inductor_start = time.time()
            for _ in range(steps):
                c_m(input)
            inductor_end = time.time()
        print("ref time is: {}".format(ref_end - ref_start), flush=True)
        print("inductor time is: {}".format(inductor_end - inductor_start), flush=True)
        print(torch.allclose(ref_res[0], inductor_res[0], atol=0.01, rtol=0.01), flush=True)
        print(torch.allclose(ref_res[1], inductor_res[1], atol=0.01, rtol=0.01), flush=True)




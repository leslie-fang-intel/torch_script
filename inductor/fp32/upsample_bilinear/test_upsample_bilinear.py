# TORCHINDUCTOR_FREEZING=1  TORCH_LOGS="+output_code" numactl -C 56-111 -m 1 python test_upsample_bilinear.py


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
        self.attn_dropout = torch.nn.Dropout(0.1)
        self.up = torch.nn.Upsample(scale_factor=2,mode='bilinear',align_corners = True)

    def forward(self, attn_weights):
        return self.up(attn_weights)
 
if __name__ == "__main__":

    with torch.no_grad():
        m = M().eval()
        input = torch.randn(4, 12, 1024, 1024).to(torch.float32)

        m(input)

        warmup_steps = 100
        steps = 1000

        # Refer path
        ref_res = m(input)

        for _ in range(warmup_steps):
            m(input)

        ref_start = time.time()
        for _ in range(steps):
            m(input)
        ref_end = time.time()

        # Compiler Path
        c_m = torch.compile(m)
        inductor_res = c_m(input)

        for _ in range(warmup_steps):
            c_m(input)

        inductor_start = time.time()
        for _ in range(steps):
            c_m(input)
        inductor_end = time.time()
        print("ref time is: {}".format(ref_end - ref_start), flush=True)
        # print("jit time is: {}".format(jit_end - jit_start), flush=True)
        print("inductor time is: {}".format(inductor_end - inductor_start), flush=True)
        print(torch.allclose(ref_res[0], inductor_res[0], atol=0.01, rtol=0.01), flush=True)
        print(torch.allclose(ref_res[1], inductor_res[1], atol=0.01, rtol=0.01), flush=True)




# rm -rf /tmp/torchinductor_leslie/* && rm -rf torch_compile_debug/* && clear && TORCH_LOGS="+schedule,+inductor,+output_code" TORCH_COMPILE_DEBUG=1  TORCHINDUCTOR_FREEZING=1 numactl -C 56-56 --membind=1 python arrange.py

# Test Commit: gh/CaoE/36/orig  53a9251a50838d50b1b2c5b5348cc2559377b8cb 

# Ref Inductor Commit: 7a694f66835ab18512a723c1761f2945c831416f


import torch
import time
import random
import numpy as np
import torch.nn.functional as F
import torch._inductor.config

local_seed= 2024

torch.manual_seed(local_seed) # Set PyTorch seed
np.random.seed(seed=local_seed) # Set Numpy seed
random.seed(local_seed) # Set the Python seed

dtype = torch.bfloat16
autocast = True if dtype == dtype else False

class M(torch.nn.Module):
    def __init__(self,) -> None:
        super(M, self).__init__()
    def forward(self, x, x2):
        return (x + x2).to(torch.float32)

if __name__ == "__main__":

    with torch.no_grad():
        m = M().eval()
        
        input = torch.randn(2, 960, 96, 96).to(dtype)
        input2 = torch.randn(2, 960, 96, 96).to(dtype)
        inputs = (input, input2)

        warmup_steps = 100
        steps = 1000

        # Compiler Path
        with torch.autocast(device_type="cpu", dtype=dtype, enabled=autocast):
            ref_res = m(*inputs)

            for _ in range(warmup_steps):
                m(*inputs)

            ref_start = time.time()
            for _ in range(steps):
                m(*inputs)
            ref_end = time.time()
    
            c_m = torch.compile(m)
            inductor_res = c_m(*inputs)

            for _ in range(warmup_steps):
                c_m(*inputs)

            inductor_start = time.time()
            for _ in range(steps):
                c_m(*inputs)
            inductor_end = time.time()

            print("ref time is: {}".format(ref_end - ref_start), flush=True)
            print("inductor time is: {}".format(inductor_end - inductor_start), flush=True)
            print(torch.allclose(ref_res, inductor_res, atol=0.01, rtol=0.01), flush=True)


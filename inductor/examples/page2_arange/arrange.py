# rm -rf /tmp/torchinductor_leslie/* && rm -rf torch_compile_debug/* && clear && TORCH_LOGS="+schedule,+inductor,+output_code" TORCH_COMPILE_DEBUG=1  numactl -C 56-56 --membind=1 python arrange.py

# Test Commit: 7a694f66835ab18512a723c1761f2945c831416f
# ref time is: 0.3226466178894043
# inductor time is: 0.10808634757995605

# Ref Inductor Commit: a254fbfd611309dbe9b7b9f9cb20e50900c474c4
# ref time is: 0.3129715919494629
# inductor time is: 0.6549742221832275

import torch
import time
import random
import numpy as np

local_seed= 2024

torch.manual_seed(local_seed) # Set PyTorch seed
np.random.seed(seed=local_seed) # Set Numpy seed
random.seed(local_seed) # Set the Python seed

dtype = torch.float32
autocast = True if dtype == dtype else False

class M(torch.nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x):
        pad = torch.nn.functional.pad(x, (0, 16))
        relu = torch.relu(pad)
        output = torch.sum(relu, -1)
        return output

if __name__ == "__main__":

    with torch.no_grad():
        m = M().eval()
        input = torch.randn(512, 256).to(dtype)

        warmup_steps = 100
        steps = 1000

        # Compiler Path
        with torch.autocast(device_type="cpu", dtype=dtype, enabled=autocast):
            ref_res = m(input)

            for _ in range(warmup_steps):
                m(input)

            ref_start = time.time()
            for _ in range(steps):
                m(input)
            ref_end = time.time()
    
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

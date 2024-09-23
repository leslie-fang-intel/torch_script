# rm -rf /tmp/torchinductor_leslie/* && rm -rf torch_compile_debug/* && clear && TORCH_LOGS="+schedule,+inductor,+output_code" TORCH_COMPILE_DEBUG=1  TORCHINDUCTOR_FREEZING=1 numactl -C 56-56 --membind=1 python arrange.py

# Test Commit: gh/jiayisunx/18/orig 4c41fadb09bf37828e0f5a319d7a38a3083b982d 
# ref time is: 18.426328420639038
# inductor time is: 26.548088312149048

# Ref Inductor Commit: 7a694f66835ab18512a723c1761f2945c831416f
# ref time is: 17.674598455429077
# inductor time is: 65.5637719631195


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
        self.gn = torch.nn.GroupNorm(32, 960)
    def forward(self, x):
        return self.gn(x)

if __name__ == "__main__":

    with torch.no_grad():
        m = M().eval()
        
        input = torch.randn(2, 960, 96, 96).to(dtype).to(memory_format=torch.channels_last)
        inputs = (input,)

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


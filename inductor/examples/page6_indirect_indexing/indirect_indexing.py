# rm -rf /tmp/torchinductor_leslie/* && rm -rf torch_compile_debug/* && clear && TORCH_LOGS="+schedule,+inductor,+output_code" TORCH_COMPILE_DEBUG=1  TORCHINDUCTOR_FREEZING=1 numactl -C 56-56 --membind=1 python arrange.py

# Test Commit: 7a694f66835ab18512a723c1761f2945c831416f
# ref time is: 4.274747371673584
# inductor time is: 1.662907361984253


# Ref Inductor Commit: 3e1abde46d7904dea60cc4fe317730a0c47b6e9e
# ref time is: 4.235726356506348
# inductor time is: 8.21605920791626


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

dtype = torch.float32
autocast = True if dtype == dtype else False

class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(64, 8192)

    def forward(self, idx, x):
        return F.layer_norm(self.emb(idx) + x, (4, 32, 8192))

if __name__ == "__main__":

    with torch.no_grad():
        m = M().eval()
        
        # input = torch.randn(256, 64, 128).to(dtype)
        idx = torch.randint(0, 64, (4, 32))
        input = torch.randn(4, 32, 8192)
        inputs = (idx, input)

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


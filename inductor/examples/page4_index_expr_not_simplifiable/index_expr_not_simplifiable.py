# rm -rf /tmp/torchinductor_leslie/* && rm -rf torch_compile_debug/* && clear && TORCH_LOGS="+schedule,+inductor,+output_code" TORCH_COMPILE_DEBUG=1  TORCHINDUCTOR_FREEZING=1 numactl -C 56-56 --membind=1 python arrange.py

# Test Commit: 7a694f66835ab18512a723c1761f2945c831416f
# ref time is: 1.289675235748291
# inductor time is: 0.6799483299255371


# Ref Inductor Commit: 6c624aad377723bde32026c8ddd58a3091c9efc4
# ref time is: 1.0963525772094727
# inductor time is: 2.6741366577148438


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
    def __init__(self,):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(512)
        self.conv = torch.nn.Conv2d(512, 1536, 1)

    def forward(self, x):
        # the pattern from sebotnet33ts_256
        x = x.transpose(-1, -2).reshape(64, 512, 8, 8)
        tmp = F.silu(self.bn(x))
        return self.conv(tmp)

if __name__ == "__main__":

    with torch.no_grad():
        m = M().eval()
        input = torch.randn(256, 64, 128).to(dtype)

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
            print("inductor time is: {}".format(inductor_end - inductor_start), flush=True)
            print(torch.allclose(ref_res, inductor_res, atol=0.01, rtol=0.01), flush=True)


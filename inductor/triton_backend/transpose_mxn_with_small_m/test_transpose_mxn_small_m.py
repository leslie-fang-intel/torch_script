# TORCHINDUCTOR_FREEZING=1  TORCH_LOGS="+output_code" numactl -C 56-111 -m 1 python test_softmax.py

import torch
import time
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch._inductor.config

torch._inductor.config.cpp.enable_kernel_profile=True
torch._inductor.config.profiler_mark_wrapper_call = True

local_seed= 2024

torch.manual_seed(local_seed) # Set PyTorch seed
np.random.seed(seed=local_seed) # Set Numpy seed
random.seed(local_seed) # Set the Python seed

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

    def forward(self, x):
        return x.contiguous()
 
if __name__ == "__main__":

    with torch.no_grad():
        m = Up(128, 64, True).eval()
        input = torch.randn(1, 2, 640, 959).to(memory_format=torch.channels_last)


        m(input)


        # Multi Thread
        warmup_steps = 50
        steps = 100

        # # Single Thread
        # warmup_steps = 10
        # steps = 20

        # Refer path
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False):
            ref_res = m(input)

            for _ in range(warmup_steps):
                m(input)

            ref_start = time.time()
            for _ in range(steps):
                m(input)
            ref_end = time.time()

        # Compiler Path
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False):
            c_m = torch.compile(m)
            inductor_res = c_m(input)

            for _ in range(warmup_steps):
                c_m(input)

            inductor_start = time.time()
            for step in range(steps):
                if step == 19:
                    with torch.profiler.profile(on_trace_ready=torch.profiler.tensorboard_trace_handler("./int8_log")) as prof:
                        c_m(input)
                    print(prof.key_averages().table(sort_by="self_cpu_time_total"))
                else:
                    c_m(input)
            inductor_end = time.time()
        print("ref time is: {}".format(ref_end - ref_start), flush=True)
        # print("jit time is: {}".format(jit_end - jit_start), flush=True)
        print("inductor time is: {}".format(inductor_end - inductor_start), flush=True)
        print(torch.allclose(ref_res, inductor_res, atol=0.01, rtol=0.01), flush=True)
        # print(torch.allclose(ref_res, inductor_res, atol=0.01, rtol=0.01), flush=True)




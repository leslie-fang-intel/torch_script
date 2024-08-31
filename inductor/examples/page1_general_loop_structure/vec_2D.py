# TORCHINDUCTOR_FREEZING=1 TORCH_LOGS="+output_code" numactl -C 56-111 -m 1 python test.py

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

    def forward(self, x, groups):
        # channel shuffle
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x.contiguous(memory_format=torch.channels_last)

if __name__ == "__main__":

    with torch.no_grad():
        m = M().eval()
        input = torch.randn(64, 58, 28, 28).to(dtype)
        group = 2

        # Compiler Path
        with torch.autocast(device_type="cpu", dtype=dtype, enabled=autocast):
            ref_res = m(input, group)
            c_m = torch.compile(m)
            inductor_res = c_m(input, group)
            print(torch.allclose(ref_res, inductor_res, rtol=1e-3, atol=1e-3), flush=True)

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

    def forward(self, input):
        relu = torch.relu(input)
        output = torch.sum(relu, -1)
        return output

if __name__ == "__main__":

    with torch.no_grad():
        m = M().eval()
        input = torch.randn(1024, 1025).to(dtype)

        # Compiler Path
        with torch.autocast(device_type="cpu", dtype=dtype, enabled=autocast):
            ref_res = m(input)
            c_m = torch.compile(m)
            inductor_res = c_m(input)
            print(torch.allclose(ref_res, inductor_res, rtol=1e-3, atol=1e-3), flush=True)

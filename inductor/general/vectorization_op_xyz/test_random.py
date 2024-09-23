# TORCHINDUCTOR_FREEZING=1 TORCH_LOGS="+output_code" numactl -C 56-111 -m 1 python test_softmax.py
# * Eager time: 1.025406198501587
# * Inductor Baseline time after outer loop fusion: 5.541730642318726
# * Option 1: 1.1507635116577148
# * Option 2: 1.6568920612335205


import torch
import time
import random
import numpy as np

from torch._inductor import config as inductor_config
# inductor_config.cpp_wrapper = True

local_seed = 2024

# torch.manual_seed(local_seed) # Set PyTorch seed
# np.random.seed(seed=local_seed) # Set Numpy seed
# random.seed(local_seed) # Set the Python seed

class M(torch.nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, seed):
        torch.manual_seed(seed)
        return torch.randn(64, dtype=torch.float32)
        # return x2

# dynamic = True
dynamic = False

def single_check():
    with torch.no_grad():
        m = M().eval()

        # input = torch.randn(128)
        # print(input, flush=True)

        # torch.manual_seed(local_seed)
        # print(torch.randn(128), flush=True)
        # print(torch.randn(128), flush=True)
        # torch.manual_seed(local_seed)
        # print(torch.randn(128), flush=True)

        # exit(-1)

        ref_res = m(2024)
        c_m = torch.compile(m, dynamic=dynamic)
        inductor_res = c_m(2024)

        print("ref_res is; {}".format(ref_res), flush=True)
        print("inductor_res is; {}".format(inductor_res), flush=True)
        print(m(2024), flush=True)
        print(c_m(2024), flush=True)
        print(c_m(2024), flush=True)

        # c_m(2025)

        print(torch.allclose(c_m(2024), c_m(2024), atol=0.01, rtol=0.01), flush=True)

if __name__ == "__main__":
    # for_check()
    single_check()






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

local_seed= 2024

torch.manual_seed(local_seed) # Set PyTorch seed
np.random.seed(seed=local_seed) # Set Numpy seed
random.seed(local_seed) # Set the Python seed

class M(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        self.attn_dropout = torch.nn.Dropout(0.1)

    def forward(self, x, x2):
        # return torch.div(x, x2, rounding_mode="floor")
        return torch.div(x, x2, rounding_mode="trunc")
        # return torch.div(x, x2)

# dynamic = True
dynamic = False

def for_check():
    with torch.no_grad():
        m = M().eval()
        for dtype in [torch.bool, torch.uint8, torch.int8, torch.int32, torch.int64]:
        # for dtype in [torch.bool]:
            input = torch.randn(4, 12, 1025, 1024)
            input2 = torch.randn(4, 12, 1025, 1024)

            if dtype == torch.bool:
                input = input > 0
                input2 = input2 > 0
            else:
                input = input.to(dtype)
                input2 = input2.to(dtype)
            
            print(input, flush=True)

            ref_res = m(input, input2)
            c_m = torch.compile(m, dynamic=dynamic)
            inductor_res = c_m(input, input2)

            print(torch.allclose(ref_res, inductor_res, atol=0.01, rtol=0.01), flush=True)

def single_check():
    with torch.no_grad():
        m = M().eval()
        # for dtype in [torch.bool, torch.uint8, torch.int8, torch.int32, torch.int64]:
        # for dtype in [torch.bool]:
        # input = torch.randn(128)
        # input2 = torch.randn(128)

        # input = input.to(torch.uint8)
        # input2 = input2.to(torch.uint8)

        dtype = torch.uint8
        
        input = torch.randint(1, 100, (64,), dtype=dtype)
        input2 = torch.randint(1, 100, (64,), dtype=dtype)

        # input = torch.tensor([-5]*128)
        # input2 = torch.tensor([-4]*128)

        
        # print(input, flush=True)
        # print(input2, flush=True)

        ref_res = m(input, input2)
        c_m = torch.compile(m, dynamic=dynamic)
        inductor_res = c_m(input, input2)

        print("ref_res is; {}".format(ref_res), flush=True)
        print("inductor_res is; {}".format(inductor_res), flush=True)

        print(torch.allclose(ref_res, inductor_res, atol=0.01, rtol=0.01), flush=True)

if __name__ == "__main__":
    # for_check()
    single_check()






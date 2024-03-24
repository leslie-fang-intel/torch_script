# CMD to reproduce the failure: rm -rf /tmp/torchinductor_leslie/* && clear && TORCH_LOGS="+output_code" numactl -C 56-56 -m 1 python test_outer_loop_fusion_no_kernel.py


import torch
import time
import random
import numpy as np

local_seed= 2024

torch.manual_seed(local_seed) # Set PyTorch seed
np.random.seed(seed=local_seed) # Set Numpy seed
random.seed(local_seed) # Set the Python seed

def fn(x):
    return (
        torch.var(x, True),
        torch.var(x, False),
        torch.var(x, -1, True),
        torch.var(x, -1, False),
        torch.std(x, False),
        torch.std(x, [0, 1], True),
        torch.std(x, [0, 1], False),
        torch.std(x, -2, True, keepdim=True),
    )
 
if __name__ == "__main__":

    with torch.no_grad():
        x = torch.randn([2, 4, 4, 8])
        ref_res = fn(x) 
        c_m = torch.compile(fn)

        res = c_m(x)

        for item1, item2 in zip(ref_res, res):
            print(torch.allclose(item1, item2), flush=True)





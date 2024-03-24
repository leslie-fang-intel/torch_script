# TORCHINDUCTOR_FREEZING=1 TORCH_LOGS="+output_code" numactl -C 0-27 -m 0 python test_amax_sub.py
# * Eager: 5.996412754058838
# * Inductor: 7.722699880599976
# * Inductor Loop fusion: 5.99749493598938

import torch
import time
import random
import numpy as np
from torch._inductor import config

local_seed= 2024

torch.manual_seed(local_seed) # Set PyTorch seed
np.random.seed(seed=local_seed) # Set Numpy seed
random.seed(local_seed) # Set the Python seed

torch._inductor.config.realize_opcount_threshold = 0  # Force realize each pointwise

class M(torch.nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, w1, w2):
        o1 = x * w1
        o2 = x * w2
        output = o1 + o2
        return output
 
if __name__ == "__main__":
    with torch.no_grad():
        m = M().eval()
        input = torch.randn(2, 3, 10, 11)
        input2 = torch.randn(11)
        input3 = torch.randn(11)
        m(input, input2, input3)
        # Compiler Path
        c_m = torch.compile(m)
        inductor_res = c_m(input, input2, input3)
        c_m(input, input2, input3)





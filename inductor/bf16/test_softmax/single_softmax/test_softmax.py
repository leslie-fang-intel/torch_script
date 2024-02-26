# TORCHINDUCTOR_FREEZING=1  TORCH_LOGS="+output_code" numactl -C 56-111 -m 1 python test_softmax.py
# * Eager time: 1.025406198501587
# * Step 1: Inductor Baseline time: 10.193575143814087

# Modfied the generated kernel implementation
# Refer to: https://github.com/pytorch/pytorch/blob/da0635d17c8fc777010fc3a2c5efedfade499432/aten/src/ATen/native/cpu/SoftMaxKernel.cpp#L147-L211

# * Step 2: Inductor optimize 1: naive loop fusion time: 5.443028450012207
# * Step 3: Inductor optimize 2: The max val calculation in loop 1 don't need to store for the usage in Loop 2. No Obvious improvement
# * Step 4: Inductor optimize 3: In loop 2, load elements and convert to FP32 again. We can save the FP32 data in loop 1, it will save the convert time.  No Obvious improvement
# * Step 5: Inductor optimize 4: The sum val calculation in loop 2 don't need to store for the usage in Loop 3. No Obvious improvement
# * Step 6: Inductor optimize 5: Multi inv_sum instead of div sum. No Obvious improvement
# * Step 7: Inductor optimize 6: In Loop 2, the val after exp will save to tmp buffer (size is dim_size) instead of entire buffer (size is dim_size * outer_size)  1.0165352821350098
  
# * Step 8: best_performance_generated_code.py time: 1.0116770267486572
#   * Step 2 and Step 7 are the only 2 steps we needed to get similar performance as aten.

import torch
import time
import random
import numpy as np

local_seed= 2024

torch.manual_seed(local_seed) # Set PyTorch seed
np.random.seed(seed=local_seed) # Set Numpy seed
random.seed(local_seed) # Set the Python seed

class M(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        self.attn_dropout = torch.nn.Dropout(0.1)

    def forward(self, attn_weights):
        # attn_weights:
        # size(4, 12, 1024, 1024)
        # stride(12582912, 1048576, 1024, 1)
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)  
        return attn_weights
 
if __name__ == "__main__":

    with torch.no_grad():
        m = M().eval()
        input = torch.randn(4, 12, 1024, 1024).to(torch.bfloat16)

        m(input)

        warmup_steps = 100
        steps = 1000

        # Refer path
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            ref_res = m(input)

            for _ in range(warmup_steps):
                m(input)

            ref_start = time.time()
            for _ in range(steps):
                m(input)
            ref_end = time.time()

        # Compiler Path
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            c_m = torch.compile(m)
            inductor_res = c_m(input)

            for _ in range(warmup_steps):
                c_m(input)

            inductor_start = time.time()
            for _ in range(steps):
                c_m(input)
            inductor_end = time.time()
        print("ref time is: {}".format(ref_end - ref_start), flush=True)
        # print("jit time is: {}".format(jit_end - jit_start), flush=True)
        print("inductor time is: {}".format(inductor_end - inductor_start), flush=True)
        print(torch.allclose(ref_res[0], inductor_res[0], atol=0.01, rtol=0.01), flush=True)
        print(torch.allclose(ref_res[1], inductor_res[1], atol=0.01, rtol=0.01), flush=True)




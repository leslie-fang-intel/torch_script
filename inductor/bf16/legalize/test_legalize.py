# TORCHINDUCTOR_FREEZING=1  TORCH_LOGS="+output_code" numactl -C 56-111 -m 1 python test_softmax.py

import torch
import time
import random
import numpy as np

local_seed= 2024

torch.manual_seed(local_seed) # Set PyTorch seed
np.random.seed(seed=local_seed) # Set Numpy seed
random.seed(local_seed) # Set the Python seed

DEVICE='cpu'

p0 = torch.tensor([1.0879], dtype=torch.float16).to(DEVICE)

class Model0(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        cat = torch.cat((args[3], args[2], args[1], args[0]), dim = 2)
        max_1 = torch.max(args[4], p0)
        mul = torch.mul(cat, max_1)
        tan = torch.tan(mul)
        return (tan,)
 
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        cat = torch.cat((args[3], args[2], args[1], args[0]), dim = 2)
        max_1 = torch.max(args[4], p0)
        mul = torch.mul(cat, max_1)
        tan = torch.tan(mul)
        return (mul, tan)

if __name__ == "__main__":

    with torch.no_grad():
        # m = Model0().eval()
        m = Model1().eval()
        data_0 = np.random.normal(5, 1, size=(17, 5, 1, 7)).astype(np.float16)
        data_1 = np.random.normal(5, 1, size=(17, 5, 1, 7)).astype(np.float16)
        data_2 = np.random.normal(5, 1, size=(17, 5, 11, 7)).astype(np.float16)
        data_3 = np.random.normal(5, 1, size=(17, 5, 1, 7)).astype(np.float16)
        data_4 = np.array(4.39, dtype=np.float16)
        input_data_0 = [data_0,data_1,data_2,data_3,data_4,]

        m(*[torch.from_numpy(v).to(DEVICE) for v in input_data_0])

        warmup_steps = 100
        steps = 1000

        # Refer path
        # with torch.autocast(device_type="cpu", dtype=torch.float16):
        ref_res = m(*[torch.from_numpy(v).to(DEVICE) for v in input_data_0])

        c_m = torch.compile(m)
        inductor_res = c_m(*[torch.from_numpy(v).to(DEVICE) for v in input_data_0])




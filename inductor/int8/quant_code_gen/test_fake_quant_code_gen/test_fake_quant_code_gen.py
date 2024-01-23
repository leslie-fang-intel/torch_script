# CMD: clear && TORCHINDUCTOR_FREEZING=1 TORCH_LOGS="+output_code" python test_fake_quant_code_gen.py
import torch
from torch import _dynamo, _inductor
from torch._inductor import config
import logging
import numpy as np
import random

# torch._dynamo.config.log_level = logging.DEBUG
torch._dynamo.config.verbose = True
torch._inductor.config.trace.enabled = True
torch._inductor.config.debug = True

local_seed = 2023
torch.manual_seed(local_seed) # Set PyTorch seed
np.random.seed(seed=local_seed) # Set Numpy seed
random.seed(local_seed) # Set the Python seed

def test1():
    def fn(input, scale, zero_point, quant_min, quant_max, dtype):
        inv_scale = 1.0 / scale
        quant = torch.clamp(torch.round(input * inv_scale) + zero_point, quant_min, quant_max).to(dtype)
        dequant = (quant.to(torch.float32) - zero_point) * scale
        return dequant

    # **TODO** Also test channel_last format
    x = torch.randn((1, 3, 224, 224), dtype=torch.float) * 100

    with torch.no_grad():
        opt_fn = torch._dynamo.optimize("inductor")(fn)
        opt_fn(x, 1.0, 1, 0, 255, torch.uint8)

        real_out = fn(x, 1.0, 1, 0, 255, torch.uint8)
        compiled_out = opt_fn(x, 1.0, 1, 0, 255, torch.uint8)
        # print("compiled_out is: {}".format(compiled_out), flush=True)
        tol = 0.0001
        print(torch.allclose(real_out, compiled_out, atol=tol, rtol=tol))

if __name__ == "__main__":
    test1()

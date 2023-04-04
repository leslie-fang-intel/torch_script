import torch
from torch import _dynamo, _inductor
from torch._inductor import config
import logging
import numpy as np
import random

torch._dynamo.config.log_level = logging.DEBUG
torch._dynamo.config.verbose = True
torch._inductor.config.trace.enabled = True
torch._inductor.config.debug = True

local_seed = 2023
torch.manual_seed(local_seed) # Set PyTorch seed
np.random.seed(seed=local_seed) # Set Numpy seed
random.seed(local_seed) # Set the Python seed

def test1():
    def fn(x, zero_point, scale):
        # x = torch.ops.aten.sigmoid.default(x)
        #return torch.ops.aten.mean.dim(x, [-1, -2], True)
        
        y = (x.to(torch.float) - zero_point) * scale
        # y = torch.clamp(
        #     torch.round(x * 0.001) + 2, 0, 255
        # ).to(torch.uint8)
        return y

    # **TODO** Also test channel_last format
    x = torch.randn((1, 3, 224, 224), dtype=torch.float) * 100
    x = x.to(torch.uint8)

    zero_point = 10
    scale = 0.001

    print("x is: {}".format(x), flush=True)

    opt_fn = torch._dynamo.optimize("inductor")(fn)
    opt_fn(x, zero_point, scale)

    real_out = fn(x, zero_point, scale)
    compiled_out = opt_fn(x, zero_point, scale)
    print("compiled_out is: {}".format(compiled_out), flush=True)
    tol = 0.0001
    print(torch.allclose(real_out, compiled_out, atol=tol, rtol=tol))



if __name__ == "__main__":
    test1()


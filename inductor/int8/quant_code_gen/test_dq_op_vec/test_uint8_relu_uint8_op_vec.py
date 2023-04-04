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

relu = torch.nn.ReLU()

def test1():
    def fn(x):
        # x = torch.ops.aten.sigmoid.default(x)
        #return torch.ops.aten.mean.dim(x, [-1, -2], True)
        
        tmp = x.to(torch.float)
        # print("tmp is: {}".format(tmp), flush=True)

        tmp = torch.relu(tmp)

        y = tmp.to(torch.uint8)

        return y

    # **TODO** Also test channel_last format
    x = torch.randn((1, 3, 224, 224), dtype=torch.float) * 100
    x = x.to(torch.uint8)

    print("x is: {}".format(x), flush=True)

    opt_fn = torch._dynamo.optimize("inductor")(fn)
    opt_fn(x)

    real_out = fn(x)

    compiled_out = opt_fn(x)
    print("compiled_out is: {}".format(compiled_out), flush=True)
    tol = 0.0001
    print(torch.allclose(real_out, compiled_out, atol=tol, rtol=tol))

if __name__ == "__main__":
    test1()


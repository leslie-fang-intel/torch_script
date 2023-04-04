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
    def fn(x):
        # x = torch.ops.aten.sigmoid.default(x)
        #return torch.ops.aten.mean.dim(x, [-1, -2], True)
        
        y = x.to(torch.uint8)
        # y = torch.clamp(
        #     torch.round(x * 0.001) + 2, 0, 255
        # ).to(torch.uint8)
        return y

    # **TODO** Also test channel_last format
    x = torch.randn((1, 3, 224, 224), dtype=torch.float)

    opt_fn = torch._dynamo.optimize("inductor")(fn)
    opt_fn(x)

    real_out = fn(x)
    compiled_out = opt_fn(x)
    tol = 0.0001
    print(torch.allclose(real_out, compiled_out, atol=tol, rtol=tol))


def test2():
    def fn(x):
        # x = torch.ops.aten.sigmoid.default(x)
        # return torch.ops.aten.mean.dim(x, [-1, -2], True)
        y = x.to(torch.bool)
        return y

    # **TODO** Also test channel_last format
    x = torch.ones((1, 3, 224, 224), dtype=torch.float)

    real_out = fn(x)
    print("input x is: {}".format(x), flush=True)
    print("real_out is: {}".format(real_out), flush=True)


    opt_fn = torch._dynamo.optimize("inductor")(fn)

    print("---- start step1 execution ---", flush=True)
    opt_fn(x)
    print("---- start original execution ---", flush=True)

    compiled_out = opt_fn(x)
    tol = 0.0001
    print(torch.allclose(real_out, compiled_out, atol=tol, rtol=tol))

def test3():
    x = torch.ones((1, 3, 224, 224), dtype=torch.float)
    y = x.to(torch.uint8)
    return y

def test4():
    x = torch.ones((1, 3, 224, 224), dtype=torch.float)
    scale, zero_point = 1e-4, 2
    dtype = torch.quint8
    q_per_tensor = torch.quantize_per_tensor(x, scale, zero_point, dtype)
    return q_per_tensor

if __name__ == "__main__":
    test1()
    #test2()
    #test3()
    #test4()

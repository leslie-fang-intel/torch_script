import torch
from torch import _dynamo, _inductor
from torch._inductor import config
import logging
import numpy as np
import random
from torch._inductor import codecache, config, metrics, test_operators

torch._dynamo.config.log_level = logging.DEBUG
torch._dynamo.config.verbose = True
torch._inductor.config.trace.enabled = True
torch._inductor.config.trace.debug_log = True
torch._inductor.config.debug = True

local_seed = 2023
torch.manual_seed(local_seed) # Set PyTorch seed
np.random.seed(seed=local_seed) # Set Numpy seed
random.seed(local_seed) # Set the Python seed

relu = torch.nn.ReLU()

def test1():
    def fn(x, zero_point, scale):
        # x = torch.ops.aten.sigmoid.default(x)
        #return torch.ops.aten.mean.dim(x, [-1, -2], True)
        
        tmp = (x.to(torch.float) - zero_point) * scale

        # print("tmp is: {}".format(tmp), flush=True)

        tmp = relu(tmp)

        # print("tmp is: {}".format(tmp), flush=True)

        inv_scale = 1.0 / scale
        y = torch.clamp(
                torch.round(tmp * inv_scale) + zero_point, 0, 255
            ).to(torch.uint8)
        # y = torch.clamp(
        #     torch.round(x * 0.001) + 2, 0, 255
        # ).to(torch.uint8)
        return y

    # **TODO** Also test channel_last format
    x = torch.randn((1, 3, 224, 224), dtype=torch.float) * 100
    x = x.to(torch.uint8)

    zero_point = 100
    scale = 0.001

    print("x is: {}".format(x), flush=True)

    opt_fn = torch._dynamo.optimize("inductor")(fn)
    opt_fn(x, zero_point, scale)

    real_out = fn(x, zero_point, scale)

    # exit(-1)

    compiled_out = opt_fn(x, zero_point, scale)
    print("real_out is: {}".format(real_out), flush=True)
    print("compiled_out is: {}".format(compiled_out), flush=True)
    tol = 0.0001
    print(torch.allclose(real_out, compiled_out, atol=tol, rtol=tol))
    assert torch.allclose(real_out, compiled_out, atol=tol, rtol=tol), "Fail to compare result of real_out and compiled_out"



if __name__ == "__main__":
    simdlen = None # Default, on this system is avx512 version
    simdlen = 1 # scalar version
    simdlen = 255 # scalar version
    simdlen = 256 # Unspported avx2 version
    #simdlen = 257 # scalar version
    #simdlen = 512 # avx512 version
    #simdlen = 513 # scalar version

    simdlens = [None, 1, 255, 256, 257, 512, 513]

    #simdlens = [512]

    for simdlen in simdlens:
        with config.patch({"cpp.simdlen": simdlen}):
            torch._dynamo.reset()
            metrics.reset()
            test1()
            print("simdlen is: {}".format(simdlen), flush=True)
            if simdlen in [None, 256, 512]:
                assert metrics.generated_cpp_vec_kernel_count >= 1
            print("metrics.generated_cpp_vec_kernel_count is: {}".format(metrics.generated_cpp_vec_kernel_count), flush=True)


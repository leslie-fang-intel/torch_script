import torch
from torch import _dynamo, _inductor
from torch._inductor import config
import logging
import numpy as np
import random
from torch.fx.experimental.proxy_tensor import make_fx
from torch._inductor.compile_fx import (
    compile_fx,
    compile_fx_inner,
    complex_memory_overlap,
)
from torch._inductor import codecache, config, metrics, test_operators

# torch._dynamo.config.log_level = logging.DEBUG
torch._dynamo.config.verbose = True
torch._inductor.config.trace.enabled = True
torch._inductor.config.trace.debug_log = True
torch._inductor.config.debug = True

local_seed = 2023
torch.manual_seed(local_seed) # Set PyTorch seed
np.random.seed(seed=local_seed) # Set Numpy seed
random.seed(local_seed) # Set the Python seed

relu = torch.nn.ReLU()

def test():

    class M(torch.nn.Module):
        def __init__(self, bias=True):
            super().__init__()
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            return self.relu(x)

    def fn(x):
        tmp = torch.ops.quantized_decomposed.dequantize_per_tensor.tensor(
            x,
            scale=torch.tensor(0.1, dtype=torch.float),
            zero_point=torch.tensor(1, dtype=torch.int64),
            quant_min=0,
            quant_max=255,
            dtype=torch.uint8,
        )
        y = torch.relu(tmp)
        return y


    # **TODO** Also test channel_last format
    with torch.no_grad():
        m = M().eval()
        x = torch.randn((1, 3, 257, 257), dtype=torch.float) * 100
        x = x.to(torch.uint8)

        # print("x is: {}".format(x), flush=True)
        # res = fn(x)
        # print("res is: {}".format(res), flush=True)

        # return

        print("x is: {}".format(x), flush=True)

        torch._dynamo.reset()


        traced = make_fx(fn)(x)
        
        print("traced graph is: {}".format(traced), flush=True)
        opt_fn = torch._dynamo.optimize("inductor")(fn)
        print("finish compile fx", flush=True)
        
        opt_fn(x)

        real_out = fn(x)

        compiled_out = opt_fn(x)
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

    simdlens = [None, 1, 255, 256, 257, 512, 513]

    simdlens = [512]

    for simdlen in simdlens:
        print("simdlen is: {}".format(simdlen), flush=True)
        with config.patch({"cpp.simdlen": simdlen}):
            torch._dynamo.reset()
            metrics.reset()
            test()


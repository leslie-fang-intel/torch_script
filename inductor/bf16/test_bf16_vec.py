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

# torch._dynamo.config.log_level = logging.DEBUG
torch._dynamo.config.verbose = True
torch._inductor.config.trace.enabled = True
torch._inductor.config.debug = True

local_seed = 2023
torch.manual_seed(local_seed) # Set PyTorch seed
np.random.seed(seed=local_seed) # Set Numpy seed
random.seed(local_seed) # Set the Python seed

relu = torch.nn.ReLU()

def test1():
    def fn(x1, x2):
        y = torch.relu(x1)
        return (y + x2, )

    dtype = torch.bfloat16
    torch.manual_seed(0)
    x1 = torch.randn((5, 20), dtype=dtype)
    x2 = torch.randn((5, 20), dtype=dtype)

    tol = 1e-2 if dtype == torch.bfloat16 else 1e-4
    # with config.patch({"cpp.simdlen": 1}):
    #     torch._dynamo.reset()
    #     traced = make_fx(fn)(x1, x2)
    #     compiled = compile_fx_inner(traced, [x1, x2])
    #     # assert same(
    #     #     fn(x1, x2)[0], compiled([x1, x2])[0], equal_nan=True, tol=tol
    #     # )
    #     # assert metrics.generated_cpp_vec_kernel_count == 0

    with config.patch({"cpp.simdlen": None}):
        torch._dynamo.reset()
        # metrics.reset()
        traced = make_fx(fn)(x1, x2)

        print("traced graph is: {}".format(traced), flush=True)

        compiled = compile_fx_inner(traced, [x1, x2])
        # assert same(fn(x1, x2)[0], compiled([x1, x2])[0], equal_nan=True)
        # assert metrics.generated_cpp_vec_kernel_count == 1

if __name__ == "__main__":
    test1()


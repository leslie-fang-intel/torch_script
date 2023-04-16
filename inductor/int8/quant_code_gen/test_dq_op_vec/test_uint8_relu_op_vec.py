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

def test1():
    def fn(x):
        # x = torch.ops.aten.sigmoid.default(x)
        #return torch.ops.aten.mean.dim(x, [-1, -2], True)
        
        tmp = x.to(torch.float)
        # print("tmp is: {}".format(tmp), flush=True)

        y = torch.relu(tmp)

        # y = tmp.to(torch.uint8)

        return y
        #return (y, )

    # **TODO** Also test channel_last format
    x = torch.randn((1, 3, 257, 257), dtype=torch.float) * 100
    x = x.to(torch.uint8)

    #x2 = torch.randn((1, 3, 224, 224), dtype=torch.float) * 100

    print("x is: {}".format(x), flush=True)

    torch._dynamo.reset()

    """
    Pass Graph before enter into compile_fx_inner:
    def forward(self, arg0_1):
        convert_element_type = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        relu = torch.ops.aten.relu.default(convert_element_type);  convert_element_type = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(relu, torch.uint8);  relu = None
        return (convert_element_type_1,)
    """
    # opt_fn = torch._dynamo.optimize("inductor")(fn)


    # """
    # Failed Graph before enter into compile_fx_inner:
    # def forward(self, x_1):
    # _to_copy = torch.ops.aten._to_copy.default(x_1, dtype = torch.float32);  x_1 = None
    # relu = torch.ops.aten.relu.default(_to_copy);  _to_copy = None
    # _to_copy_1 = torch.ops.aten._to_copy.default(relu, dtype = torch.uint8);  relu = None
    # return _to_copy_1
    # """
    traced = make_fx(fn)(x)
    print("traced graph is: {}".format(traced), flush=True)
    opt_fn = compile_fx(traced, [x, ])
    # opt_fn = compile_fx_inner(traced, [x, ])
    
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
    #simdlen = 257 # scalar version
    #simdlen = 512 # avx512 version
    #simdlen = 513 # scalar version

    simdlens = [None, 1, 255, 256, 257, 512, 513]

    # simdlens = [None]

    for simdlen in simdlens:
        print("simdlen is: {}".format(simdlen), flush=True)
        with config.patch({"cpp.simdlen": simdlen}):
            torch._dynamo.reset()
            metrics.reset()
            test1()


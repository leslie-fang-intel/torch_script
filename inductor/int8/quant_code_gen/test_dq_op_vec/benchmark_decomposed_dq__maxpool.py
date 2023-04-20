import torch
from torch import _dynamo, _inductor
from torch._inductor import config
import logging
import numpy as np
import random
from torch._inductor import codecache, config, metrics, test_operators
import torch.ao.quantization.fx._decomposed

torch._dynamo.config.verbose = True
torch._inductor.config.trace.enabled = True
torch._inductor.config.trace.debug_log = True
torch._inductor.config.debug = True

local_seed = 2023
torch.manual_seed(local_seed) # Set PyTorch seed
np.random.seed(seed=local_seed) # Set Numpy seed
random.seed(local_seed) # Set the Python seed

def benchmark():
    def fn(x, scale, zero_point):
        x = torch.ops.quantized_decomposed.dequantize_per_tensor(x, scale, zero_point, 0, 255, torch.uint8)
        max_pool2d_with_indices_default = torch.ops.aten.max_pool2d_with_indices.default(x, [3, 3], [2, 2], [1, 1])[0]
        return max_pool2d_with_indices_default

    x = torch.clamp(
        torch.randn((116, 64, 112, 112), dtype=torch.float32) * 100, 0, 255
    ).to(torch.uint8).to(memory_format=torch.channels_last)
    zero_point = torch.tensor(100, dtype=torch.int32)
    scale = torch.tensor(0.01)
    with config.patch({"cpp.simdlen": None}):
        torch._dynamo.reset()
        metrics.reset()
        opt_fn = torch._dynamo.optimize("inductor")(fn)
        opt_fn(x, scale, zero_point)

        import datetime
        start = datetime.datetime.now()
        for i in range(100):
            opt_fn(x, scale, zero_point)
        end = datetime.datetime.now()
        print("time is:{} ms".format((end-start).microseconds / 1000.0), flush=True)


if __name__ == "__main__":
    benchmark()


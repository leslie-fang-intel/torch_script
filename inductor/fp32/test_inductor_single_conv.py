import torch
import torch._dynamo as torchdynamo
from torch.ao.quantization import (
    get_default_qconfig,
    QConfigMapping,
)
from torch.ao.quantization._quantize_pt2e import prepare_pt2e, convert_pt2e
from torch._inductor.compile_fx import compile_fx

def test_single_conv():
    import copy
    from torch import _dynamo, _inductor
    from torch._inductor import config
    import logging
    import numpy as np
    import random

    local_seed = 2023
    torch.manual_seed(local_seed) # Set PyTorch seed
    np.random.seed(seed=local_seed) # Set Numpy seed
    random.seed(local_seed) # Set the Python seed
    torch._dynamo.config.log_level = logging.DEBUG
    torch._dynamo.config.verbose = True
    torch._inductor.config.trace.enabled = True
    torch._inductor.config.debug = True

    class Mod(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(
                # in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1
                in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
            )
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            # return self.conv(x)
            x = self.conv(x)
            return self.relu(x)

    with torch.no_grad():
        # torch._dynamo.reset()

        example_inputs = (torch.randn(1, 3, 16, 16),)
        m = Mod().eval()

        run = torch._dynamo.optimize(compile_fx, nopython=False)(m)
        # first run
        print("start the first run", flush=True)
        print(type(run))
        inductor_result = run(*example_inputs)

        # second run
        print("start the second run", flush=True)
        # import pdb;pdb.set_trace()
        inductor_result = run(*example_inputs)

if __name__ == "__main__":
    test_single_conv()


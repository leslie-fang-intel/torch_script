import torch
import torch._dynamo as torchdynamo

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

    class Mod(torch.nn.Module):
        def __init__(
            self,
            add_fn,
            **kwargs,
        ):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
            self.conv2 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
            self.add_fn = add_fn
            self.relu = torch.nn.ReLU(inplace=True)
            self.conv3 = torch.nn.Conv2d(6, 6, kernel_size=3, stride=1)
            self.conv4 = torch.nn.Conv2d(6, 6, kernel_size=3, stride=1)
            self.add_fn2 = add_fn
            self.relu2 = torch.nn.ReLU(inplace=True)
            self.use_relu = True

        def forward(self, x):
            x1 = self.conv1(x)
            x2 = self.conv2(x)
            tmp = self.add_fn(x1, x2)
            if self.use_relu:
                tmp = self.relu(tmp)
            tmp1 = self.conv3(tmp)
            tmp2 = self.conv4(tmp)
            res = self.add_fn2(tmp1, tmp2)
            if self.use_relu:
                res = self.relu2(res)
            return res

    with torch.no_grad():
        example_inputs = (torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False).add(
            1
        ),)
        m = Mod(lambda x, y: x.add_(y), ).eval()
        om = torch.compile(m)
        om(*example_inputs)
        om(*example_inputs)

if __name__ == "__main__":
    test_single_conv()


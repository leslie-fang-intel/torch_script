import torch
import time
import random
import numpy as np
from dataclasses import dataclass
# import intel_extension_for_pytorch as ipex

local_seed= 2024

torch.manual_seed(local_seed) # Set PyTorch seed
np.random.seed(seed=local_seed) # Set Numpy seed
random.seed(local_seed) # Set the Python seed

@dataclass
class TorchSDPAMetadata():
   max_len: int
 
@torch.library.impl("myops::longrope", "cpu")
def longrope(
    input,
    max_len,
):
    return input.new_empty((5, max_len, 3))

@torch.library.impl("myops::longrope", "Meta")
def longrope(
    input,
    max_len,
):
    return input.new_empty((5, max_len, 3))

torch.library.define(
    "myops::longrope",
    "(Tensor inv_freq, int max_len) -> (Tensor)",
)

class M(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        self.attn_dropout = torch.nn.Dropout(0.1)

    def forward(self, attn_weights, meta_data):
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1) 
        attn_weights = torch.ops.myops.longrope(attn_weights, meta_data.max_len) 
        # attn_weights = attn_weights * 2
        # attn_weights = torch.ops.torch_ipex.longrope(attn_weights, meta_data.max_len) 
        return attn_weights
 
if __name__ == "__main__":

    with torch.no_grad():
        m = M().eval()
        input = torch.randn(4, 12, 1024, 1024).to(torch.bfloat16)

        m(input, TorchSDPAMetadata(1))

        cm = torch.compile(m, dynamic=True)
        for step in range(3):
            print("---- step: {}".format(step), flush=True)
            meta_data = TorchSDPAMetadata(max_len=(step + 1))
            res = cm(input, meta_data)
            print("res.size is: {}".format(res.size()), flush=True)



import requests
import torch
print(torch.__version__, flush=True)
import torch.nn as nn
import os, pickle
import numpy as np 
import torch._inductor.config as config
import torch._dynamo.config as dynamo_config
import gc
import time
import psutil
import refcycle
import torchao
from torchao import autoquant
from torchao.quantization import ALL_AUTOQUANT_CLASS_LIST

# # dynamo_config.inline_inbuilt_nn_modules = False

config.freezing = True
# config.max_autotune = True
output_channels = 1024

dtype = torch.bfloat16


class M(torch.nn.Module):
    def __init__(self, output_channels, dtype):
        super().__init__()
        self.lin = torch.nn.Linear(1024, output_channels, bias=False).to(dtype)

    def forward(self, attn_weights):
        attn_weights = self.lin(attn_weights)
        return attn_weights

manual = True
manual = False
if __name__ == "__main__":
    with torch.no_grad():
        model = M(output_channels, dtype).eval()
        ## Optional: invoke torch.compile
        model = torch.compile(model)
        model = torchao.autoquant(model, manual=manual)
        # model = torchao.autoquant(model, manual=True)
        # model = torchao.autoquant(model, manual=True, set_inductor_config=False)
        # model = torchao.autoquant(model, set_inductor_config=False)

        x = torch.randn(2, 1024).to(dtype)
        print("---- start the formal run ----", flush=True)
        model(x)
        if manual:
            model.finalize_autoquant()
        print("---- start the second run ----", flush=True)
        model(x)

        print("---- start the thrid run ----", flush=True)
        # Both manual = True or False can hit the compile path as _weight_int8pack_mm_cpu
        res = model(x)
        
        print("---- finished ----", flush=True)

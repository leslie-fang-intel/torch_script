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

# config.freezing = True
# config.freezing_discard_parameters = True

# # config.max_autotune = True
# # config.max_autotune_gemm_backends = "CPP,"
# config.fx_graph_cache = False
# config.fx_graph_remote_cache = False
# config.autotune_local_cache = False

# dynamic = True

# output_channels = 4096 * 1024
output_channels = 1024

dtype = torch.bfloat16


class M(torch.nn.Module):
    def __init__(self, output_channels, dtype):
        super().__init__()
        self.lin = torch.nn.Linear(1024, output_channels, bias=False).to(dtype)
        # self.linear2 = torch.nn.Linear(output_channels, 1024)

    def forward(self, attn_weights):
        attn_weights = self.lin(attn_weights)
        # attn_weights = self.linear2(attn_weights)
        return attn_weights


if __name__ == "__main__":
    with torch.no_grad():
        model = M(output_channels, dtype).eval()

        # model = torch.compile(model, mode="max-autotune")
        # model = autoquant(
        #     model,
        #     qtensor_class_list=ALL_AUTOQUANT_CLASS_LIST,
        #     set_inductor_config=False,
        #     # **self.quantization_config.quant_type_kwargs,
        # )

        ## Optional: invoke torch.compile
        # model = torch.compile(model)
        torchao.autoquant(model, set_inductor_config=False)

        x = torch.randn(2, 1024).to(dtype)
        model(x)
        print("---- finished ----", flush=True)

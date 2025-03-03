import requests
import torch
print(torch.__version__)
import torch.nn as nn
import os, pickle
import numpy as np 
import torch._inductor.config as config

config.freezing = True
config.freezing_discard_parameters = True

# config.max_autotune = True
# config.max_autotune_gemm_backends = "CPP,"
config.fx_graph_cache = False
config.fx_graph_remote_cache = False
config.autotune_local_cache = False

dynamic = True

output_channels = 4096 * 1024
# output_channels = 1024


class M(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        self.linear = torch.nn.Linear(1024, output_channels)
        # self.linear2 = torch.nn.Linear(output_channels, 1024)

    def forward(self, attn_weights):
        attn_weights = self.linear(attn_weights)
        # attn_weights = self.linear2(attn_weights)
        return attn_weights

import time
import psutil

if __name__ == "__main__":
    # time.sleep(30)
    print("Init psutil.virtual_memory() is: {}".format(psutil.virtual_memory()), flush=True)

    dtype = torch.bfloat16

    x = torch.randn(2, 1024).to(dtype)
    # x2 = torch.randn(3, 1024).to(dtype)
    with torch.no_grad(), torch.autocast(device_type="cpu"):
        m = M().eval().to(dtype)
        print("After model create psutil.virtual_memory() is: {}".format(psutil.virtual_memory()), flush=True)
        cfn = torch.compile(m, dynamic=dynamic)
        cfn(x)
        import gc
        gc.collect()
        time.sleep(30)
        print("After first run psutil.virtual_memory() is: {}".format(psutil.virtual_memory()), flush=True)

        # cfn(x2)
        # import gc
        # gc.collect()
        # time.sleep(30)
        # print("After recompile run psutil.virtual_memory() is: {}".format(psutil.virtual_memory()), flush=True)

        # for _ in range(10):
        #     cfn(x)
        # time.sleep(30)
        # print("Final psutil.virtual_memory() is: {}".format(psutil.virtual_memory()), flush=True)

        print("Done", flush=True)
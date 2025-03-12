import requests
import torch
print(torch.__version__)
import torch.nn as nn
import os, pickle
import numpy as np 
import torch._inductor.config as config

config.freezing = True
config.max_autotune = True
config.max_autotune_gemm_backends = "CPP,"
config.fx_graph_cache = False
config.fx_graph_remote_cache = False
config.autotune_local_cache = False

class M(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        self.linear = torch.nn.Linear(1024, 4096 * 1024)
        # self.linear = torch.nn.Linear(1024, 1024)
        
        
        # self.linear2 = torch.nn.Linear(4096 * 1024, 4096 * 1024)

    def forward(self, attn_weights):
        attn_weights = self.linear(attn_weights)
        # attn_weights = self.linear2(attn_weights)  
        return attn_weights

import time
import psutil

autocast_enabled = False

if __name__ == "__main__":
    time.sleep(30)
    print("Init psutil.virtual_memory() is: {}".format(psutil.virtual_memory()), flush=True)

    x = torch.randn(1, 1024)
    with torch.no_grad(), torch.autocast(device_type="cpu", enabled=autocast_enabled):
        m = M().eval()
        print(x, flush=True)
        print(m(x), flush=True)

        cfn = torch.compile(m)
        print(cfn(x), flush=True)
        import gc
        gc.collect()
        time.sleep(30)
        print("After first run psutil.virtual_memory() is: {}".format(psutil.virtual_memory()), flush=True)
        for _ in range(10):
            cfn(x)
        time.sleep(30)
        print("Final psutil.virtual_memory() is: {}".format(psutil.virtual_memory()), flush=True)

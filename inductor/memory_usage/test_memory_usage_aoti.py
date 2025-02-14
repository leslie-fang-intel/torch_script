import requests
import torch
print(torch.__version__)
import torch.nn as nn
import os, pickle
import numpy as np 
import torch._inductor.config as config
import gc
import time
import psutil

config.freezing = True
# config.freezing_discard_parameters = True

config.max_autotune = True
config.max_autotune_gemm_backends = "CPP,"
config.fx_graph_cache = False
config.fx_graph_remote_cache = False
config.autotune_local_cache = False

dynamic = True

output_channels = 4096 * 1024
# output_channels = 1024
profile_memory = True
# profile_memory = False

class M(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        self.linear = torch.nn.Linear(1024, output_channels)
        # self.linear2 = torch.nn.Linear(output_channels, 1024)

    def forward(self, x):
        res = self.linear(x)
        # res = self.linear2(res)
        return res

def optimize_model(dtype):
    # Optimize with AOTI
    with torch.no_grad():
        m = M().eval().to(dtype)
        device = "cpu"
        model = m.to(device=device)
        example_inputs=(torch.randn(2, 1024).to(dtype),)
        batch_dim = torch.export.Dim("batch", min=1, max=1024)
        # [Optional] Specify the first dimension of the input x as dynamic.
        exported = torch.export.export(model, example_inputs, dynamic_shapes={"x": {0: batch_dim}})
        # [Note] In this example we directly feed the exported module to aoti_compile_and_package.
        # Depending on your use case, e.g. if your training platform and inference platform
        # are different, you may choose to save the exported model using torch.export.save and
        # then load it back using torch.export.load on your inference platform to run AOT compilation.
        output_path = torch._inductor.aoti_compile_and_package(
            exported,
            # [Optional] Specify the generated shared library path. If not specified,
            # the generated artifact is stored in your system temp directory.
            package_path=os.path.join(os.getcwd(), "model.pt2"),
        )

def load_run(dtype):
    with torch.no_grad():
        if profile_memory:
            gc.collect()
            time.sleep(30)
            print("Before model load psutil.virtual_memory() is: {}".format(psutil.virtual_memory()), flush=True)

        device = "cpu"
        model = torch._inductor.aoti_load_package(os.path.join(os.getcwd(), "model.pt2"))

        if profile_memory:
            gc.collect()
            time.sleep(30)
            print("After model load psutil.virtual_memory() is: {}".format(psutil.virtual_memory()), flush=True)

        input = torch.randn(2, 1024).to(dtype)
        print(model(input), flush=True)

        if profile_memory:
            gc.collect()
            time.sleep(30)
            print("After first run psutil.virtual_memory() is: {}".format(psutil.virtual_memory()), flush=True)


if __name__ == "__main__":
    dtype = torch.bfloat16
    # optimize_model(dtype)
    load_run(dtype)

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

# dynamo_config.inline_inbuilt_nn_modules = False

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

dtype = torch.bfloat16
# dtype = torch.float32

test_conv = True
test_conv = False

profile_memory = True

class M(torch.nn.Module):
    def __init__(self, output_channels, dtype):
        super().__init__()
        self.lin = torch.nn.Linear(1024, output_channels, bias=False).to(dtype)
        self.linear2 = torch.nn.Linear(output_channels, 1024, bias=False).to(dtype)

    def forward(self, attn_weights):
        attn_weights = self.lin(attn_weights)
        attn_weights = self.linear2(attn_weights)
        return attn_weights

class ConvBN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=bias, **kwargs)
        # self.bn = torch.nn.BatchNorm2d(out_channels, eps=0.001, dtype=torch.float)

    def forward(self, x):
        return self.conv(x)
        # return self.bn(self.conv(x))

if __name__ == "__main__":

    if profile_memory:
        gc.collect()
        time.sleep(10)
        print("Init psutil.virtual_memory() is: {}".format(psutil.virtual_memory()), flush=True)

    with torch.no_grad():

        if test_conv:
            m = ConvBN(3, 6, kernel_size=3, stride=2).eval().to(memory_format=torch.channels_last)
            x = torch.rand(2, 3, 6, 6)        
        else:
            m = M(output_channels, dtype).eval()
            x = torch.randn(2, 1024).to(dtype)
            # c = m.lin.weight

        # def finalize_f():
        #     import traceback
        #     traceback.print_stack()

        # from weakref import finalize
        # if test_conv:
        #     # print("m.conv.weight is: {}".format(id(m.conv.weight)), flush=True)
        #     finalize(m.conv.weight, finalize_f)
        # else:
        #     # print("m.lin.weight is: {}".format(id(m.lin.weight)), flush=True)
        #     finalize(m.lin.weight, finalize_f)

        if profile_memory:
            gc.collect()
            time.sleep(10)
            print("After model create psutil.virtual_memory() is: {}".format(psutil.virtual_memory()), flush=True)
        # cfn = torch.compile(m, dynamic=dynamic)
        cfn = torch.compile(m)
        cfn(x)

        # snapshot = refcycle.snapshot()
        # ancestors = snapshot.ancestors(c, 1)
        # # print("---- len() is: {} ancestors is: {} ----".format(
        # #     len(list(ancestors)),
        # #     list(ancestors)),
        # #     flush=True,
        # # )
        # for idx in range(len(list(ancestors))):
        #     print("idx is: {} ancestors is: {}; id is: {} ----".format(
        #             idx,
        #             list(ancestors)[idx],
        #             id(list(ancestors)[idx]),
        #         ),
        #         flush=True,
        #     )
        # annotated_graph = ancestors.annotated()
        # print("---- annotated_graph.to_json is: {}".format(annotated_graph.to_json()), flush=True)

        if profile_memory:
            gc.collect()
            time.sleep(10)
            print("After first run psutil.virtual_memory() is: {}".format(psutil.virtual_memory()), flush=True)
        
        print("---- start the second run after release memory ----", flush=True)
        cfn(x)

        print("Done", flush=True)

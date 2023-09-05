import torch
import torchvision.models as models
import torch._dynamo as torchdynamo
import copy
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import random
import numpy as np
from torch._export import capture_pre_autograd_graph, dynamic_dim
import os
import torch.nn as nn

random.seed(2023)
torch.manual_seed(2023)
np.random.seed(2023)
class SingleConv2dModule(torch.nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 6, (2, 2), stride=(1, 1), padding=(1, 1))
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(x)
        return self.conv(x)

def test():
    model_name = "resnet50"
    torch._dynamo.config.verbose = True
    torch._inductor.config.trace.enabled = True
    torch._inductor.config.trace.debug_log = True
    torch._inductor.config.debug = True
    torch._inductor.config.freezing = True
    torch._dynamo.config.assume_static_by_default = False
    torch._dynamo.config.automatic_dynamic_shapes = True

    print("start fp32 test of model: {}".format(model_name), flush=True)

    # model = models.__dict__[model_name](pretrained=True).eval()
    model = SingleConv2dModule().eval()
    x = torch.randn(17, 3, 224, 224).contiguous(memory_format=torch.channels_last)
    example_inputs = (x,)
    # export_with_dynamic_shape_list = [True, False]
    export_with_dynamic_shape_list = [True,]
    for export_with_dynamic_shape in export_with_dynamic_shape_list:
        os.system("rm -rf /home/leslie/quantization/torch_script/inductor/int8/test_new_export_api/torch_compile_debug/*")
        os.system("rm -rf /tmp/torchinductor_root/*")

        with torch.no_grad():
            # Lower into Inductor
            # optimized_model = torch.compile(model)
            optimized_model = torch.compile(model, dynamic=True)

            print("---- start first run ----", flush=True)
            optimized_model(x)
            # print("---- start second run ----", flush=True)
            # optimized_model(x)
            # print("---- start second run ----", flush=True)
            # optimized_model(x)
            # print("---- start second run ----", flush=True)
            # optimized_model(x)
            # print("---- start run with changed bs less ----", flush=True)
            # x2 = torch.randn(8, 3, 224, 224).contiguous(memory_format=torch.channels_last)
            # optimized_model(x2)

            # print("---- start run with changed bs more ----", flush=True)
            # x3 = torch.randn(34, 3, 224, 224).contiguous(memory_format=torch.channels_last)
            # optimized_model(x3)

    print("Finish fp32 test of model: {}".format(model_name), flush=True)

if __name__ == "__main__":
    test()

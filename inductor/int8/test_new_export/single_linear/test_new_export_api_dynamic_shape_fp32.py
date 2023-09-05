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

class SingleLinearModule(torch.nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(10, 30, True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))

def test():
    model_name = "resnet50"
    torch._dynamo.config.verbose = True
    torch._inductor.config.trace.enabled = True
    torch._inductor.config.trace.debug_log = True
    torch._inductor.config.debug = True
    torch._inductor.config.freezing = True
    torch._dynamo.config.assume_static_by_default = False
    torch._dynamo.config.automatic_dynamic_shapes = True

    print("start int8 test of model: {}".format(model_name), flush=True)

    # model = models.__dict__[model_name](pretrained=True).eval()
    model = SingleLinearModule().to(torch.bfloat16).eval()

    x = torch.randn(17, 10).to(torch.bfloat16)
    example_inputs = (x,)
    # export_with_dynamic_shape_list = [True, False]
    export_with_dynamic_shape_list = [True,]
    for export_with_dynamic_shape in export_with_dynamic_shape_list:
        os.system("rm -rf /home/lesliefang/pytorch_1_7_1/inductor_quant/torch_script/inductor/int8/test_new_export/single_linear/torch_compile_debug/*")
        os.system("rm -rf /tmp/torchinductor_root/*")

        with torch.no_grad():
            # # Generate the FX Module
            # exported_model = capture_pre_autograd_graph(
            #     model,
            #     example_inputs,
            #     constraints=[dynamic_dim(example_inputs[0], 0) >= 16] if export_with_dynamic_shape else [],
            # )

            # print("exported_model is: {}".format(exported_model), flush=True)

            # # Create X86InductorQuantizer
            # quantizer = X86InductorQuantizer()
            # quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())
            # # PT2E Quantization flow
            # prepared_model = prepare_pt2e(exported_model, quantizer)
            # # print("prepared_model is: {}".format(prepared_model), flush=True)
            # # from torch.fx.passes.graph_drawer import FxGraphDrawer
            # # g = FxGraphDrawer(prepared_model, "shuffnetv2")
            # # g.get_dot_graph().write_svg("//home/lesliefang/pytorch_1_7_1/inductor_quant/torch_script/inductor/int8/pytorch_2_1_accuracy_test/new_frontend_shuffnetv2_prepare.svg")
            # # Calibration
            # prepared_model(x)
            # converted_model = convert_pt2e(prepared_model).eval()
            # print("converted_model is: {}".format(converted_model), flush=True)
            # # Lower into Inductor
            # # optimized_model = torch.compile(model, dynamic=True)
            
            # # x2 = torch.randn(8, 224)
            # # x3 = torch.randn(34, 224)
            # # ref_res1 = converted_model(x)
            # # ref_res2 = converted_model(x2)
            # # ref_res3 = converted_model(x3)
            
            optimized_model = torch.compile(model, dynamic=True)
            # optimized_model = torch.compile(model)

            print("---- start first run ----", flush=True)
            act_res1 = optimized_model(x)
            # print("---- start second run ----", flush=True)
            # optimized_model(x)

            # print("---- start run with changed bs less ----", flush=True)
            # # x2 = torch.randn(8, 3, 224, 224).contiguous(memory_format=torch.channels_last)
            # act_res2 = optimized_model(x2)

            # print("---- start run with changed bs more ----", flush=True)
            # # x3 = torch.randn(34, 3, 224, 224).contiguous(memory_format=torch.channels_last)
            # act_res3 = optimized_model(x3)

            # # print("ref_res1 is: {}".format(ref_res1), flush=True)
            # # print("act_res1 is: {}".format(act_res1), flush=True)
            # print("act_res1.size is: {}".format(act_res1.size()), flush=True)
            # print("act_res2.size is: {}".format(act_res2.size()), flush=True)
            # print("act_res3.size is: {}".format(act_res3.size()), flush=True)
            # print(torch.allclose(ref_res1, act_res1, atol=0.05, rtol=0.05), flush=True)
            # print(torch.allclose(ref_res2, act_res2, atol=0.05, rtol=0.05), flush=True)
            # print(torch.allclose(ref_res3, act_res3, atol=0.05, rtol=0.05), flush=True)

    print("Finish int8 test of model: {}".format(model_name), flush=True)

if __name__ == "__main__":
    test()

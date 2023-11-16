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

random.seed(2023)
torch.manual_seed(2023)
np.random.seed(2023)

def test():
    model_name = "resnet50"
    torch._dynamo.config.verbose = True
    torch._inductor.config.trace.enabled = True
    torch._inductor.config.trace.debug_log = True
    torch._inductor.config.debug = True
    torch._inductor.config.freezing = True

    print("start int8 test of model: {}".format(model_name), flush=True)

    model = models.__dict__[model_name](pretrained=True).eval()
    x = torch.randn(116, 3, 224, 224).contiguous(memory_format=torch.channels_last)
    example_inputs = (x,)
    export_with_dynamic_shape_list = [True, False]
    for export_with_dynamic_shape in export_with_dynamic_shape_list:
        os.system("rm -rf /home/leslie/quantization/torch_script/inductor/int8/test_new_export_api/torch_compile_debug/*")
        os.system("rm -rf /tmp/torchinductor_root/*")

        with torch.no_grad():
            # Generate the FX Module
            # exported_model, guards = torchdynamo.export(
            #     model,
            #     *copy.deepcopy(example_inputs),
            #     aten_graph=True,
            # )

            exported_model = capture_pre_autograd_graph(
                model,
                example_inputs,
                # constraints = []
                constraints=[dynamic_dim(example_inputs[0], 0) >= 16] if export_with_dynamic_shape else [],
            )

            print("exported_model is: {}".format(exported_model), flush=True)

            # Create X86InductorQuantizer
            quantizer = X86InductorQuantizer()
            quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())
            # PT2E Quantization flow
            prepared_model = prepare_pt2e(exported_model, quantizer)
            # print("prepared_model is: {}".format(prepared_model), flush=True)
            # from torch.fx.passes.graph_drawer import FxGraphDrawer
            # g = FxGraphDrawer(prepared_model, "shuffnetv2")
            # g.get_dot_graph().write_svg("//home/lesliefang/pytorch_1_7_1/inductor_quant/torch_script/inductor/int8/pytorch_2_1_accuracy_test/new_frontend_shuffnetv2_prepare.svg")
            # Calibration
            prepared_model(x)
            converted_model = convert_pt2e(prepared_model).eval()
            # print("converted_model is: {}".format(converted_model), flush=True)
            # Lower into Inductor
            optimized_model = torch.compile(converted_model)
            print("---- start first run ----", flush=True)
            optimized_model(x)
            print("---- start second run ----", flush=True)
            optimized_model(x)
    print("Finish int8 test of model: {}".format(model_name), flush=True)

if __name__ == "__main__":
    test()

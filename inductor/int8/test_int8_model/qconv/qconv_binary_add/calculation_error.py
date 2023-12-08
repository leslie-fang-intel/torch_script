import torch
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
    prepare_qat_pt2e,
)
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq

def test_inductor_int8_conv_add_relu():
    import copy
    from torch import _dynamo, _inductor
    from torch._inductor import config
    import logging
    import numpy as np
    import random

    local_seed = 2023
    torch.manual_seed(local_seed) # Set PyTorch seed
    np.random.seed(seed=local_seed) # Set Numpy seed
    random.seed(local_seed) # Set the Python seed

    # torch._dynamo.config.log_level = logging.DEBUG
    torch._dynamo.config.verbose = True
    torch._inductor.config.trace.enabled = True
    torch._inductor.config.debug = True

    class Mod(torch.nn.Module):
        def __init__(self, inplace_add=False, use_relu=False) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(
                # in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1
                in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv3 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv4 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
            self.relu = torch.nn.ReLU()
            self.inplace_add = inplace_add
            self.use_relu = use_relu

        def forward(self, x):
            if not self.inplace_add:
                x1 = self.conv(x)
                res = self.conv2(x1) + self.conv3(x1)
                if self.use_relu:
                    res = self.relu(res)
                # res = self.conv4(res)
                return res
            else:
                x1 = self.conv(x)
                accum = self.conv2(x1)
                accum += self.conv3(x1)

                if self.use_relu:
                    return self.relu(accum)
                else:
                    return accum

    torch.backends.quantized.engine = "x86"
    # example_inputs = (torch.randn(1, 1, 224, 224),)
    inputs = (torch.randn(1, 3, 3, 3),)
    #m = Mod(inplace_add=True).eval()
    #for inplace_add in [True, False]:
    import itertools
    inplace_add_inplace_relu_optioins = itertools.product(
        [False],  # inplace add
        [False],  # use relu
    )

    for inplace_add, use_relu in inplace_add_inplace_relu_optioins:
        mod = Mod(inplace_add=inplace_add, use_relu=use_relu).eval()

        maybe_no_grad = torch.no_grad()
        with maybe_no_grad:
            export_model = capture_pre_autograd_graph(
                mod,
                inputs,
            )
            quantizer = X86InductorQuantizer()
            # quantizer._set_annotate_extra_input_of_binary_node(False)
            quantizer.set_global(
                xiq.get_default_x86_inductor_quantization_config()
            )
            prepare_model = prepare_pt2e(export_model, quantizer)
            prepare_model(*inputs)
            convert_model = convert_pt2e(prepare_model, fold_quantize=True)
            torch.ao.quantization.move_exported_model_to_eval(convert_model)

            res_ref = convert_model(*inputs)

            print("convert_model is: {}".format(convert_model), flush=True)
            
            # from torch.fx.passes.graph_drawer import FxGraphDrawer
            # g = FxGraphDrawer(convert_model, "resnet50")
            # g.get_dot_graph().write_svg("/home/lesliefang/pytorch_1_7_1/quantization/torch_script/inductor/int8/test_int8_model/qlinear/prepare_model.svg")
            
            enable_int8_mixed_bf16 = False
            # enable_int8_mixed_bf16 = True

            with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=enable_int8_mixed_bf16):
                compiler_mode = torch.compile(convert_model)
                print("---- start the first run ----", flush=True)
                _ = compiler_mode(*inputs)
                print("---- start the second run ----", flush=True)
                output = compiler_mode(*inputs)
                print("output dtype is: {}".format(output.dtype), flush=True)
            
            print(res_ref, flush=True)
            print(output, flush=True)
            
            print(torch.allclose(res_ref, output, atol=1e-1, rtol=1e-1), flush=True)

    print("Finish the test", flush=True)


if __name__ == "__main__":
    test_inductor_int8_conv_add_relu()


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

_quantized_relu_pattern_example_inputs = (
    torch.randn(1, 1, 3, 3),  # input1
    torch.randn(1, 1, 1, 1),  # weight1
    torch.randn(1),           # bias1
)

def conv_relu_pattern(input1, weight1, bias1):
    conv1 = torch.nn.functional.conv2d(input1, weight1, bias1)
    relu = torch.nn.functional.relu(conv1)
    return relu, {"conv": conv1, "relu": relu}

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
    torch._inductor.config.freezing = True

    class Mod(torch.nn.Module):
        def __init__(self, inplace_add=False, inplace_relu=False) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(
                # in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1
                in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv3 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
            self.relu = torch.nn.ReLU(inplace=inplace_relu)
            self.inplace_add = inplace_add
            self.conv4 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv5 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)

        def forward(self, x):
            if not self.inplace_add:
                x1 = self.conv(x)
                return self.relu(self.conv2(x1) + self.conv3(x1))
            else:
                x1 = self.conv(x)
                accum = self.conv2(x1)
                accum += self.conv3(x1)
                # print("accum.size is: {}".format(accum.size()), flush=True)
                relu_res = self.relu(accum)
                
                #return relu_res
                #return self.conv4(relu_res) - relu_res
                return self.conv5(self.conv4(relu_res)) + relu_res
                #return self.relu(self.conv5(self.relu(self.conv4(relu_res))) + relu_res)

    torch.backends.quantized.engine = "x86"
    # example_inputs = (torch.randn(1, 1, 224, 224),)
    example_inputs = (torch.randn(1, 3, 16, 16),)
    #m = Mod(inplace_add=True).eval()
    #for inplace_add in [True, False]:
    import itertools
    inplace_add_inplace_relu_optioins = itertools.product(
        [False],  # inplace add
        [False],  # inplace relu
    )

    for inplace_add, inplace_relu in inplace_add_inplace_relu_optioins:
        m = Mod(inplace_add=inplace_add, inplace_relu=inplace_relu).eval()

        # program capture
        exported_model = capture_pre_autograd_graph(
            m,
            example_inputs
        )

        print("model after torchdynamo export is: {}".format(m), flush=True)

        conv_add_pattern_gm = capture_pre_autograd_graph(conv_relu_pattern, _quantized_relu_pattern_example_inputs)
        quantizer = X86InductorQuantizer()

        quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())
        # PT2E Quantization flow
        prepared_model = prepare_pt2e(exported_model, quantizer)

        prepared_model(*example_inputs)

        converted_model = convert_pt2e(prepared_model)
        after_quant_result = converted_model(*example_inputs)
        print("model after convert_pt2e is: {}".format(m), flush=True)
        with torch.no_grad():
            optimized_model = torch.compile(converted_model)
            quant_output = optimized_model(*example_inputs)
            print("--- start the final run ----", flush=True)
            quant_output = optimized_model(*example_inputs)
        
    print("Finish the test", flush=True)


if __name__ == "__main__":
    test_inductor_int8_conv_add_relu()

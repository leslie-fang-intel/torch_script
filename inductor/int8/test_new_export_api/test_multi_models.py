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

torch._dynamo.config.verbose = True
torch._inductor.config.trace.enabled = True
torch._inductor.config.trace.debug_log = True
torch._inductor.config.debug = True
torch._inductor.config.freezing = True
class M(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 128, kernel_size=3, stride=1)

    def forward(self, x):
        x = self.conv(x)
        return x

def test(mod, inputs):
    for export_with_dynamic_shape in [True, False]:
        with torch.no_grad():
            export_model = capture_pre_autograd_graph(
                mod,
                inputs,
                constraints=[dynamic_dim(inputs[0], 0)] if export_with_dynamic_shape else [],
            )

            quantizer = X86InductorQuantizer()
            quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())
            prepare_model = prepare_pt2e(export_model, quantizer)
            prepare_model(*inputs)
            convert_model = convert_pt2e(prepare_model).eval()
            _ = torch.compile(convert_model)(*inputs)

if __name__ == "__main__":
    mod = M().eval()
    v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False).add(
        1
    )
    test(mod, (v,))


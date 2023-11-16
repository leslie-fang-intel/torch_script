# Execution Command:
# TORCHINDUCTOR_FREEZING=1 python example_x86inductorquantizer_pytorch_2_1.py
import torch
import torchvision.models as models
import torch._dynamo as torchdynamo
import copy
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
import torch.ao.quantization.pt2e.quantizer.x86_inductor_quantizer as xiq
from torch.ao.quantization.pt2e.quantizer import X86InductorQuantizer
import time



def run_adaptive_avg_pool2d():
    batch_size = 116
    model = torch.nn.AdaptiveAvgPool2d((1, 1)).eval()
    x = torch.randn(batch_size, 2048, 7, 7).contiguous(memory_format=torch.channels_last)
    example_inputs = (x,)
    with torch.no_grad():
        # Generate the FX Module
        exported_model, guards = torchdynamo.export(
            model,
            *copy.deepcopy(example_inputs),
            aten_graph=True,
        )
        print("exported_model is: {}".format(exported_model), flush=True)


if __name__ == "__main__":
    run_adaptive_avg_pool2d()
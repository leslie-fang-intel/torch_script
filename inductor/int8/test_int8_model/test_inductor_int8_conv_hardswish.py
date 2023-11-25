# Execution Command:
# TORCHINDUCTOR_FREEZING=1 python example_x86inductorquantizer_pytorch_2_1.py
import torch
import torchvision.models as models
import torch._dynamo as torchdynamo
import copy
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
import time
from torch._export import capture_pre_autograd_graph, dynamic_dim

torch._dynamo.config.verbose = True
torch._inductor.config.trace.enabled = True
torch._inductor.config.trace.debug_log = True
torch._inductor.config.debug = True
torch._inductor.config.freezing = True
torch._inductor.config.cpp.enable_kernel_profile=True

class Mod(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.relu = torch.nn.Hardswish(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        return self.relu(x)

def run_rn50():
    batch_size = 448
    model = Mod().eval()
    x = torch.randn(batch_size, 3, 224, 224).contiguous(memory_format=torch.channels_last)
    example_inputs = (x,)
    with torch.no_grad():
        # Generate the FX Module
        exported_model = capture_pre_autograd_graph(
            model,
            example_inputs
        )
        
        print("exported_model is: {}".format(exported_model), flush=True)
        
        # Create X86InductorQuantizer
        quantizer = xiq.X86InductorQuantizer()
        quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())
        # PT2E Quantization flow
        prepared_model = prepare_pt2e(exported_model, quantizer)

        print("prepared_model is: {}".format(prepared_model), flush=True)

        # Calibration
        prepared_model(*example_inputs)
        converted_model = convert_pt2e(prepared_model)

        print("converted_model is: {}".format(converted_model), flush=True)

        torch.ao.quantization.move_exported_model_to_eval(converted_model)

        enable_int8_mixed_bf16 = False
        # enable_int8_mixed_bf16 = True

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=enable_int8_mixed_bf16):
            # Lower into Inductor
            optimized_model = torch.compile(converted_model)
            # Benchmark
            optimized_model(*example_inputs)
            optimized_model(*example_inputs)


if __name__ == "__main__":
    run_rn50()
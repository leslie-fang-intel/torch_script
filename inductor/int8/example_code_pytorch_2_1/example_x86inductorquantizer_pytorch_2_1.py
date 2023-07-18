import torch
import torchvision.models as models
import torch._dynamo as torchdynamo
import copy
from torch.ao.quantization._quantize_pt2e import prepare_pt2e_quantizer, convert_pt2e
import torch.ao.quantization.pt2e.quantizer.x86_inductor_quantizer as xiq
from torch.ao.quantization.pt2e.quantizer import X86InductorQuantizer
import time

def run_rn50():
    batch_size = 116
    model = models.__dict__["resnet50"](pretrained=True).eval()
    x = torch.randn(batch_size, 3, 224, 224).contiguous(memory_format=torch.channels_last)
    example_inputs = (x,)
    with torch.no_grad():
        # Generate the FX Module
        exported_model, guards = torchdynamo.export(
            model,
            *copy.deepcopy(example_inputs),
            aten_graph=True,
        )
        # Create X86InductorQuantizer
        quantizer = X86InductorQuantizer()
        quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())
        # PT2E Quantization flow
        prepared_model = prepare_pt2e_quantizer(exported_model, quantizer)
        prepared_model(*example_inputs)
        converted_model = convert_pt2e(prepared_model).eval()
        # Lower into Inductor
        optimized_model = torch.compile(converted_model)
        # Benchmark
        warm_up_step = 100
        measurement_step = 100
        total_time = 0.0
        for _ in range(warm_up_step):
            _ = optimized_model(*example_inputs)
        for i in range(measurement_step):
            start_time = time.time()
            _ = optimized_model(*example_inputs)
            total_time += (time.time() - start_time)
            if (i+1) % 10 == 0:
                print("steps:{0}; throughput is: {1} fps".format((i+1), batch_size/(total_time/(i+1))), flush=True)

if __name__ == "__main__":
    run_rn50()
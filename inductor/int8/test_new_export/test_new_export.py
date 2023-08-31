# Execution Command:
# TORCHINDUCTOR_FREEZING=1 python example_x86inductorquantizer_pytorch_2_1.py
import torch
import torchvision.models as models
import torch._dynamo as torchdynamo
import copy
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
import time

def run_rn50_static_shape():
    batch_size = 116
    model = models.__dict__["resnet50"](pretrained=True).eval()
    x = torch.randn(batch_size, 3, 224, 224).contiguous(memory_format=torch.channels_last)
    example_inputs = (x,)
    with torch.no_grad():
        # Generate the FX Module
        exported_model = torch.export.export(
            model,
            copy.deepcopy(example_inputs),
        )
        # Success of first step
        print("exported_model is: {}".format(exported_model), flush=True)
        # # Create X86InductorQuantizer
        # quantizer = X86InductorQuantizer()
        # quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())
        # # PT2E Quantization flow
        # # Failed at this step
        # prepared_model = prepare_pt2e(exported_model, quantizer)
        # print("prepared_model is: {}".format(prepared_model), flush=True)

        # # Calibration
        # prepared_model(*example_inputs)
        # converted_model = convert_pt2e(prepared_model).eval()
        # # Lower into Inductor
        # optimized_model = torch.compile(converted_model)
        # # Benchmark
        # warm_up_step = 100
        # measurement_step = 100
        # total_time = 0.0
        # for _ in range(warm_up_step):
        #     _ = optimized_model(*example_inputs)
        # for i in range(measurement_step):
        #     start_time = time.time()
        #     _ = optimized_model(*example_inputs)
        #     total_time += (time.time() - start_time)
        #     if (i+1) % 10 == 0:
        #         print("steps:{0}; throughput is: {1} fps".format((i+1), batch_size/(total_time/(i+1))), flush=True)

def run_rn50_dynamic_shape():
    batch_size = 116
    model = models.__dict__["resnet50"](pretrained=True).eval()
    x = torch.randn(batch_size, 3, 224, 224).contiguous(memory_format=torch.channels_last)
    example_inputs = (x,)
    with torch.no_grad():
        # Generate the FX Module

        # dim 0 of x is with dynamic shape
        constraints = [torch.export.dynamic_dim(x, 0)>=16]
        exported_model = torch.export.export(
            model,
            (x,),
            constraints=constraints
        )
        # Success of first step
        print("exported_model is: {}".format(exported_model), flush=True)

if __name__ == "__main__":
    # run_rn50_static_shape()
    run_rn50_dynamic_shape()

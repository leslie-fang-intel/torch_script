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

# torch._dynamo.config.verbose = True
# torch._inductor.config.trace.enabled = True
# torch._inductor.config.trace.debug_log = True
# torch._inductor.config.debug = True
torch._inductor.config.freezing = True

def run_rn50():
    batch_size = 116
    model = models.__dict__["resnet50"](pretrained=True).eval()
    x = torch.randn(batch_size, 3, 224, 224).contiguous(memory_format=torch.channels_last)
    example_inputs = (x,)
    with torch.no_grad():
        enable_int8_mixed_bf16 = False
        # enable_int8_mixed_bf16 = True

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=enable_int8_mixed_bf16):
            # Lower into Inductor
            optimized_model = torch.compile(model)
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
# CMD: TORCHINDUCTOR_FREEZING=1 numactl -C 0-31 -m 0 python test_jira_MLDL_836.py
# Config: TORCH_COMPILE_DEBUG=1 TORCH_LOGS="+schedule,+inductor,+output_code" 

import torch.quantization.quantize_fx as quantize_fx
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx, convert_to_reference_fx, prepare_qat_fx
import copy
import torchvision
import torch
# from utils_vis import make_dot, draw
import time
import numpy as np
import os
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, prepare_qat_pt2e, convert_pt2e
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
from torch._export import capture_pre_autograd_graph, dynamic_dim

torch._inductor.config.freezing = True
# torch._dynamo.config.verbose = True
# torch._inductor.config.trace.enabled = True
# torch._inductor.config.trace.debug_log = True
# torch._inductor.config.debug = True

def inductor_qat_infer_int8(model,data):
    ######   
    iter_print = 0
    prof = 1
    if prof:
        torch._inductor.config.cpp.enable_kernel_profile=True
        torch._inductor.config.profiler_mark_wrapper_call = True
    loop = 20

    data = data.to(memory_format=torch.channels_last)
    
    exported_model = capture_pre_autograd_graph(
        model,
        (data,)
    )
    print("---- finish the graph capture ----", flush=True)

    # Create X86InductorQuantizer
    quantizer = X86InductorQuantizer()
    quantizer.set_global(xiq.get_default_x86_inductor_quantization_config(is_qat=True))
    # PT2E Quantization flow
    with torch.no_grad():
        print("---- start prepare_qat_pt2e ----", flush=True)
        prepared_model = prepare_qat_pt2e(exported_model, quantizer)
        print("---- finish prepare_qat_pt2e ----", flush=True)
        #xx_c = [torch.randn(1, 3, 224, 224) for i in range(10)]
        xx_c = [torch.randn(data.shape).to(memory_format=torch.channels_last) for i in range(10)]    

        # QAT Training
        prepared_model(data)

        print("---- start convert_pt2e ----", flush=True)
        converted_model = convert_pt2e(prepared_model)
        print("---- finish convert_pt2e ----", flush=True)
        torch.ao.quantization.move_exported_model_to_eval(converted_model)

        print("converted_model is: {}".format(converted_model), flush=True)

        print("---- start torch.compile ----", flush=True)
        optimized_model = torch.compile(converted_model)
        print("---- finish torch.compile ----", flush=True)
    
    times = []

    warm_loop = 3
    loop = 5
    times = []

    with torch.no_grad():
        print("---- start warm up", flush=True)

        for warm_step in range(warm_loop):
            print("---- start warm_step is: {}".format(warm_step), flush=True)
            _ = optimized_model(data)
            print("---- finish warm_step is: {}".format(warm_step), flush=True)
        
        print("---- finish warm up", flush=True)

        for step in range(loop):
            print("---- start step is: {}".format(step), flush=True)
            start_time = time.time()
            output = optimized_model(data)
            end_time = time.time()
            print("---- finish step is: {}".format(step), flush=True)
            times.append(end_time - start_time)
            if iter_print:
                print('time: %0.3f ms ' % ((end_time - start_time) * 1000.0))
        print ('Average latency: %0.3f ms.' % (np.median(times) * 1000.0), flush=True)
        print ('throughput: %0.3f fps.' % (data.size(0) / np.median(times)), flush=True)

        if prof:
            # with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p:
            with torch.profiler.profile(on_trace_ready=torch.profiler.tensorboard_trace_handler("./int8_log")) as prof:
                out = optimized_model(data)
            # print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))
            print(prof.key_averages().table(sort_by="self_cpu_time_total"))

# class Conv2dAddModule(torch.nn.Module):
#     def __init__(self,
#                     inplace_add: bool = False,
#                     use_bias: bool = False,
#                     ) -> None:
#         super().__init__()
#         self.conv = torch.nn.Conv2d(
#             in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, bias=use_bias
#         )
#         self.conv2 = torch.nn.Conv2d(
#             in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, bias=use_bias
#         )
#         self.bn = torch.nn.BatchNorm2d(3)
#         self.relu = torch.nn.ReLU()
#         self.relu2 = torch.nn.ReLU()

#     def forward(self, x):
#         return self.conv(x) + self.relu(x)

class Conv2dAddModule(torch.nn.Module):
    def __init__(self,
                    inplace_add: bool = False,
                    use_bias: bool = False,
                    ) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, bias=use_bias
        )
        self.bn = torch.nn.BatchNorm2d(3)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.bn(self.conv(x)) + self.relu(x)

class ConvWithBNRelu(torch.nn.Module):
    def __init__(self, relu, dim=2, bn=True, bias=True):
        super().__init__()
        convs = {1: torch.nn.Conv1d, 2: torch.nn.Conv2d}
        bns = {1: torch.nn.BatchNorm1d, 2: torch.nn.BatchNorm2d}
        self.conv = convs[dim](3, 3, 3, bias=bias)

        if bn:
            self.bn = bns[dim](3)
        else:
            self.bn = torch.nn.Identity()
        if relu:
            self.relu = torch.nn.ReLU()
        else:
            self.relu = torch.nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

class ConvWithBN(torch.nn.Module):
    def __init__(self, dim=2, bn=True, bias=True):
        super().__init__()
        convs = {1: torch.nn.Conv1d, 2: torch.nn.Conv2d}
        bns = {1: torch.nn.BatchNorm1d, 2: torch.nn.BatchNorm2d}
        self.conv = convs[dim](3, 3, 3, bias=bias)
        self.bn = bns[dim](3)


    def forward(self, x):
        x = self.conv(x)
        return self.bn(x)

if __name__ == "__main__":

    data = torch.randn(64, 3, 224, 224)
    
    # model_fp = torchvision.models.resnet50(pretrained=True)

    # model_fp = ConvWithBN()
    # model_fp = ConvWithBNRelu(relu=True)
    model_fp = Conv2dAddModule()

    # model_fp(data)


    print("--------------inductor QAT -----------")
    inductor_qat_infer_int8(model_fp,data)

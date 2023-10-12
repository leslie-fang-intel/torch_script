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

def draw_graph(model,data,graph_name):
    model.eval()
    with torch.no_grad():
        traced_model = torch.jit.trace(model, data)
        traced_model = torch.jit.freeze(traced_model)
        y = traced_model(data)
        y = traced_model(data)

        graph = traced_model.graph_for(data)
        # print(graph)
        print(graph_name,traced_model.code)
        # draw(graph).render(graph_name)

def quant_fx(model_fp,x):
    model_fp.eval()

    #
    # quantization aware training for static quantization
    #

    model_to_quantize = copy.deepcopy(model_fp)
    qconfig_dict = {"": torch.quantization.get_default_qat_qconfig('fbgemm')}
    model_to_quantize.train()
    # prepare
    #x = torch.randn(1, 3, 224, 224)
    model_prepared = quantize_fx.prepare_qat_fx(model_to_quantize, qconfig_dict, example_inputs=x)

    # training loop (not shown)
    
    # quantize
    #model_quantized = quantize_fx.convert_fx(model_prepared)
    model_quantized = convert_to_reference_fx(model_prepared)
    print("quantized model after convert_to_reference_fx: ", model_quantized, flush=True)

    #draw_graph(model_quantized,data,"resnet50_QAT_reference_1024")
    #torch.save(model_quantized.state_dict(),"resent50_qat.pth")

    return model_quantized


def Test_ipex_QATint8(model,input):
    ######
    import intel_extension_for_pytorch as ipex

    model=model.eval()
    model_ipex = model.to(memory_format=torch.channels_last)
    data = input.to(memory_format=torch.channels_last)

    with torch.no_grad():
        model_ipex = torch.jit.trace(model_ipex, data).eval()
        model_ipex = torch.jit.freeze(model_ipex)
        print("before jit trace warm up run", model_ipex.graph_for(data), flush=True)
        y = model_ipex(data)
        y = model_ipex(data)
    model_ipex.eval()

    warm_loop = 50
    loop = 150
    times =[]
    with torch.no_grad():
        for _ in range(warm_loop):
            _ = model_ipex(data)

        for _ in range(loop):
            start_time = time.time()
            output = model_ipex(data)
            end_time = time.time()
            times.append(end_time - start_time)
            #print('time: %0.3f ms ' % ((end_time - start_time) * 1000.0))
        print ('Average latency: %0.3f ms.' % (np.median(times) * 1000.0))

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p:
        out_ipex = model_ipex(data)
    print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))



        
def ipex_infer_int8(model,data):
    ######
    import intel_extension_for_pytorch as ipex
    from intel_extension_for_pytorch.quantization import prepare, convert
    from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, QConfig    
    iter_print = 0
    prof = 1
    loop = 20

    model = model.eval()

    qconfig = QConfig(activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.quint8),
        weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))    

    #xx_c = [torch.randn(1, 3, 224, 224) for i in range(10)]
    xx_c = [torch.randn(data.shape) for i in range(10)]    
    prepared_model = prepare(model, qconfig, example_inputs=xx_c[0], inplace=False)
    with torch.no_grad():
        for x in xx_c:
            y = model(x)


    data = data.to(memory_format=torch.channels_last)
    times = []

    with torch.no_grad():
        convert_model = convert(prepared_model)
        traced_model = torch.jit.trace(convert_model, data)
        traced_model = torch.jit.freeze(traced_model)
        y = traced_model(data)
        y = traced_model(data)

    warm_loop = 50
    loop = 150
    times = []

    with torch.no_grad():
        for _ in range(warm_loop):
            _ = traced_model(data)

        print("finish warm up", flush=True)

        for _ in range(loop):
            start_time = time.time()
            output = traced_model(data)
            end_time = time.time()
            times.append(end_time - start_time)
            if iter_print:
                print('time: %0.3f ms ' % ((end_time - start_time) * 1000.0))
        print ('Average latency: %0.3f ms.' % (np.median(times) * 1000.0))

    if prof:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p:
            out = traced_model(data)
        print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1)) 

def inductor_ptq_infer_int8(model,data):
    ######   
    iter_print = 0
    prof = 1
    if prof:
        torch._inductor.config.cpp.enable_kernel_profile=True
        torch._inductor.config.profiler_mark_wrapper_call = True
    loop = 20

    model = model.eval()

    # qconfig = QConfig(activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.quint8),
    #     weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))    
    
    data = data.to(memory_format=torch.channels_last)
    
    exported_model = capture_pre_autograd_graph(
        model,
        (data,)
    )

    # Create X86InductorQuantizer
    quantizer = X86InductorQuantizer()
    quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())
    # PT2E Quantization flow
    with torch.no_grad():
        prepared_model = prepare_pt2e(exported_model, quantizer)
        #xx_c = [torch.randn(1, 3, 224, 224) for i in range(10)]
        xx_c = [torch.randn(data.shape).to(memory_format=torch.channels_last) for i in range(10)]    

        prepared_model(data)

        converted_model = convert_pt2e(prepared_model)
        torch.ao.quantization.move_exported_model_to_eval(converted_model)

        optimized_model = torch.compile(converted_model)

    with torch.no_grad():
        for x in xx_c:
            y = optimized_model(x)

    
    times = []

    warm_loop = 50
    loop = 150
    times = []

    with torch.no_grad():
        for _ in range(warm_loop):
            _ = optimized_model(data)
        
        print("finish warm up", flush=True)

        for _ in range(loop):
            start_time = time.time()
            output = optimized_model(data)
            end_time = time.time()
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

    warm_loop = 50
    loop = 150
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

if __name__ == "__main__":

    data = torch.randn(64, 3, 224, 224)
    model_fp = torchvision.models.resnet50(pretrained=True)
    
    # print("--------------ipex PTQ -----------")
    # ipex_infer_int8(model_fp,data)   

    # print("-------------- PyTorch 1.X QAT-----------")
    # model_quantized=quant_fx(model_fp,data)
    # print("--------------Ipex QAT-----------")          
    # Test_ipex_QATint8(model_quantized,data)

    # print("--------------inductor PTQ -----------")
    # inductor_ptq_infer_int8(model_fp,data)

    print("--------------inductor QAT -----------")
    inductor_qat_infer_int8(model_fp,data)

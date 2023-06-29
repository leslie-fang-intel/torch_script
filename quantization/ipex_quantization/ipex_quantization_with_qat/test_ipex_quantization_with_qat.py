import torch.quantization.quantize_fx as quantize_fx
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx, convert_to_reference_fx, prepare_qat_fx
import copy
import torchvision
import torch
# from utils_vis import make_dot, draw
import time
import numpy as np
import os

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
        draw(graph).render(graph_name)

def quant_fx(model_fp,x):
    model_fp.eval()

    # quantization aware training for static quantization

    model_to_quantize = copy.deepcopy(model_fp)
    qconfig_dict = {"": torch.quantization.get_default_qat_qconfig('x86')}
    model_to_quantize.train()
    # prepare
    #x = torch.randn(1, 3, 224, 224)
    model_prepared = quantize_fx.prepare_qat_fx(model_to_quantize, qconfig_dict, example_inputs=x)

    # training loop (not shown)
    
    # quantize
    #model_quantized = quantize_fx.convert_fx(model_prepared)
    model_quantized = convert_to_reference_fx(model_prepared)
    #print("quantized model: ", model_quantized)

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
        y = model_ipex(data)
        y = model_ipex(data)
    model_ipex.eval()

    print(model_ipex.graph_for(data))

    times =[]
    with torch.no_grad():
        for _ in range(20):
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

    loop = 20
    times = []

    with torch.no_grad():
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

if __name__ == "__main__":

    data = torch.randn(1, 3, 224, 224)
    model_fp = torchvision.models.resnet50(pretrained=True)

    # print("--------------ipex PTQ -----------")
    # ipex_infer_int8(model_fp,data)   

    print("--------------QAT-----------")
    model_quantized=quant_fx(model_fp,data)
    print("--------------Ipex QAT-----------")          
    Test_ipex_QATint8(model_quantized,data)

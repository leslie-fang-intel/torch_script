import torch.quantization.quantize_fx as quantize_fx
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx, convert_to_reference_fx, prepare_qat_fx
import copy
import torchvision
import torch
import time
import numpy as np
import os
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, prepare_qat_pt2e, convert_pt2e
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
from torch._export import capture_pre_autograd_graph, dynamic_dim

def inductor_ptq_infer_int8(model,data):  
    iter_print = 0
    prof = 1
    model = model.eval()
    data = data.to(memory_format=torch.channels_last)
    exported_model = capture_pre_autograd_graph(
        model,
        (data,)
    )
    quantizer = X86InductorQuantizer()
    quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())

    with torch.no_grad():
        prepared_model = prepare_pt2e(exported_model, quantizer)
        xx_c = [torch.randn(data.shape).to(memory_format=torch.channels_last) for i in range(10)]    
        prepared_model(data)
        converted_model = convert_pt2e(prepared_model)
        torch.ao.quantization.move_exported_model_to_eval(converted_model)
        optimized_model = torch.compile(converted_model)
        for x in xx_c:
            y = optimized_model(x)


if __name__ == "__main__":
    data = torch.randn(112, 3, 224, 224)
    model_fp = torchvision.models.resnet50(pretrained=True)
    inductor_ptq_infer_int8(model_fp, data)


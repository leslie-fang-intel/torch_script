# CMD: TORCHINDUCTOR_FREEZING=1 python qlinear_input_exceed_2_not_contiguous.py
import torch
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
)
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq

import numpy as np
import random

local_seed = 2024

torch.manual_seed(local_seed) # Set PyTorch seed
np.random.seed(seed=local_seed) # Set Numpy seed
random.seed(local_seed) # Set the Python seed

def _generate_qdq_quantized_model(mod, inputs, is_qat=False):
    with torch.no_grad():
        export_model = capture_pre_autograd_graph(
            mod,
            inputs,
        )
        quantizer = X86InductorQuantizer()
        quantizer.set_global(
            xiq.get_default_x86_inductor_quantization_config(is_qat=is_qat)
        )
        prepare_model = prepare_pt2e(export_model, quantizer)
        prepare_model(*inputs)
        convert_model = convert_pt2e(prepare_model, fold_quantize=True)
        torch.ao.quantization.move_exported_model_to_eval(convert_model)
        # print("convert_model is: {}".format(convert_model), flush=True)
        return convert_model

class M(torch.nn.Module):
    def __init__(self, use_bias, do_permute=False):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4, use_bias)
        self.linear2 = torch.nn.Linear(4, 4, use_bias)
        self.do_permute = do_permute

    def forward(self, x):
        if self.do_permute:
            x= torch.reshape(torch.permute(x, (0, 2, 3, 1)), (2, 12, 4))
        return self.linear2(self.linear(x))

def test_qlinear_input_exceed_2_contiguous():
    mod = M(use_bias=False)
    inputs = (torch.randn((2, 3, 4)),)
    convert_model = _generate_qdq_quantized_model(mod, inputs)
    with torch.no_grad():
        compiled_model = torch.compile(convert_model)
        _ = compiled_model(*inputs)
        _ = compiled_model(*inputs)


def test_qlinear_input_exceed_2_non_contiguous():
    use_bias_list = [True, False]

    # for use_bias in use_bias_list:
    import itertools
    for autocast_enabled, use_bias in itertools.product(
        [True, False], use_bias_list
    ):
        # autocast_enabled = True
        mod = M(use_bias=use_bias, do_permute=True)
        inputs = (torch.randn((2, 4, 3, 4)),)
        convert_model = _generate_qdq_quantized_model(mod, inputs)
        with torch.no_grad(), torch.cpu.amp.autocast(enabled=autocast_enabled):
            res_ref = convert_model(*inputs)

            compiled_model = torch.compile(convert_model)
            _ = compiled_model(*inputs)
            res = compiled_model(*inputs)
            # print("res_ref is: {}".format(res_ref), flush=True)
            # print("res is: {}".format(res), flush=True)
            print(torch.allclose(res, res_ref, atol=0.01, rtol=0.01), flush=True)

def test_fp32_linear_input_exceed_2_non_contiguous():
    mod = M(use_bias=False, do_permute=True)
    inputs = (torch.randn((2, 4, 3, 4)),)
    with torch.no_grad():
        # mod = mod.eval()
        compiled_model = torch.compile(mod)
        _ = compiled_model(*inputs)
        _ = compiled_model(*inputs)  

if __name__ == "__main__":
    # test_fp32_linear_input_exceed_2_non_contiguous()
    # test_qlinear_input_exceed_2_contiguous()
    test_qlinear_input_exceed_2_non_contiguous()

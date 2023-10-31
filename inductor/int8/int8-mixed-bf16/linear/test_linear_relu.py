# TORCHINDUCTOR_FREEZING=1 python test_linear.py

import torch
import contextlib
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
from torch._export import capture_pre_autograd_graph

class M(torch.nn.Module):
    def __init__(self, use_bias):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4, use_bias)
        self.unary_fn = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(4, 4, use_bias)
        self.unary_fn2 = torch.nn.ReLU()

    def forward(self, x):
        tmp = self.unary_fn(self.linear(x))
        return self.unary_fn2(self.linear2(tmp))

def test_qlinear():
    bias_list = [True, False]
    for int8_mixed_bf16 in (
        [False, True] if torch.ops.mkldnn._is_mkldnn_bf16_supported() else [False]
    ):
        for bias in bias_list:
            # int8_mixed_bf16 = True
            # bias = True
            with torch.no_grad():
                mod = M(bias).eval()
                v = torch.randn((2, 4))
                inputs = (v,)
                maybe_autocast = contextlib.nullcontext()
                if int8_mixed_bf16:
                    maybe_autocast = torch.cpu.amp.autocast()
                
                with maybe_autocast:
                    ref_res = mod(*inputs)

                export_model = capture_pre_autograd_graph(
                    mod,
                    inputs,
                )
                quantizer = X86InductorQuantizer()
                quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())
                prepare_model = prepare_pt2e(export_model, quantizer)
                prepare_model(*inputs)
                convert_model = convert_pt2e(prepare_model)
                torch.ao.quantization.move_exported_model_to_eval(convert_model)
                with maybe_autocast:
                    compiled_model = torch.compile(convert_model)
                    _ = compiled_model(*inputs)
                    _ = compiled_model(*inputs)
                    res = compiled_model(*inputs)
                
                print("ref_res is: {}".format(ref_res), flush=True)
                print("res is: {}".format(res), flush=True)
                print(torch.allclose(ref_res, res, atol=1e-2, rtol=1e-2), flush=True)

if __name__ == "__main__":
    test_qlinear()

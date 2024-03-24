import torch
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
    prepare_qat_pt2e,
)
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq

torch._inductor.config.freezing = True

class M(torch.nn.Module):
    def __init__(self, use_bias):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32, use_bias)
        # self.unary_fn = torch.nn.ReLU()
        # self.linear2 = torch.nn.Linear(4, 4, use_bias)
        # self.unary_fn2 = torch.nn.ReLU()

    def forward(self, x):
        # tmp = self.unary_fn(self.linear(x))
        # return self.unary_fn2(self.linear2(tmp))
        return self.linear(x)

# bias = True
bias = True
mod = M(bias).eval()

v = torch.randn((2, 3, 16))
# v = torch.randn((2, 16))
inputs = (v,)
enable_int8_mixed_bf16 = True

maybe_no_grad = torch.no_grad()
with maybe_no_grad, torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=enable_int8_mixed_bf16):
    export_model = capture_pre_autograd_graph(
        mod,
        inputs,
    )
    quantizer = X86InductorQuantizer()
    quantizer.set_global(
        xiq.get_default_x86_inductor_quantization_config()
    )
    prepare_model = prepare_pt2e(export_model, quantizer)
    prepare_model(*inputs)
    convert_model = convert_pt2e(prepare_model, fold_quantize=True)
    torch.ao.quantization.move_exported_model_to_eval(convert_model)

    print("convert_model is: {}".format(convert_model), flush=True)
    ref_res = convert_model(*inputs)

    compiler_mode = torch.compile(convert_model)
    compiler_mode(*inputs)
    _ = compiler_mode(*inputs)
    output = compiler_mode(*inputs)

    print(output.size(), flush=True)

    print("ref_res is: {}".format(ref_res), flush=True)
    print("output is: {}".format(output), flush=True)

    print(torch.allclose(ref_res, output, atol=1e-3, rtol=1e-3), flush=True)

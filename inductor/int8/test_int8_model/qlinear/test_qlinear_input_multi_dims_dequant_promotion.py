import torch
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
    prepare_qat_pt2e,
)
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq

class M(torch.nn.Module):
    def __init__(
        self,
        use_bias,
        **kwargs,
    ):
        super().__init__()
        self.linear1 = torch.nn.Linear(16, 16, use_bias)
        self.linear2 = torch.nn.Linear(16, 16, use_bias)
        self.linear3 = torch.nn.Linear(16, 16, use_bias)

    def forward(self, x):
        temp = self.linear1(x)
        temp = self.linear2(temp) + self.linear3(temp)
        return temp

bias = True
mod = M(bias).eval()

v = torch.randn((2, 3, 16))
inputs = (v,)

maybe_no_grad = torch.no_grad()
with maybe_no_grad:
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
    
    enable_int8_mixed_bf16 = False
    # enable_int8_mixed_bf16 = True

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=enable_int8_mixed_bf16):
        compiler_mode = torch.compile(convert_model)
        compiler_mode(*inputs)
        output = compiler_mode(*inputs)

    print(output.size(), flush=True)

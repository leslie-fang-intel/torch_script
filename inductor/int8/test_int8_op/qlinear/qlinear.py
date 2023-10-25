# TORCHINDUCTOR_FREEZING=1 numactl -C 0-27 -m 0 python qlinear.py

import torch
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer


torch._inductor.config.freezing = True
# torch._dynamo.config.verbose = True
# torch._inductor.config.trace.enabled = True
# torch._inductor.config.trace.debug_log = True
# torch._inductor.config.debug = True

class M(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        self.linear = torch.nn.Linear(4, 8, True)
        self.unary_fn = torch.nn.ReLU()

    def forward(self, x):
        temp = self.linear(x)
        # print(temp.size(), flush=True)
        return self.unary_fn(temp)

def test_qlinear():
    print("---- start test_qlinear ----", flush=True)
    mod = M().eval()
    v = torch.randn((2, 4))
    inputs = (v,)
    with torch.no_grad():
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
        compiler_model = torch.compile(convert_model)

        r1 = compiler_model(v)
        r2 = compiler_model(v)


def test_linear_bf16():
    print("---- start test_linear_bf16 ----", flush=True)
    mod = M().eval()
    v = torch.randn((2, 4))
    inputs = (v,)
    
    # mod(v)
    
    with torch.no_grad(), torch.cpu.amp.autocast():
        compiler_model = torch.compile(mod)

        r1 = compiler_model(v)
        r2 = compiler_model(v)

if __name__ == "__main__":
    test_linear_bf16()
    test_qlinear()

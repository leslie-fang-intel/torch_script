import torch
import torch.nn as nn
import torch._dynamo as torchdynamo
import copy

from torch.ao.quantization._quantize_pt2e import (
    convert_pt2e,
)
from torch._inductor.compile_fx import compile_fx
import torch.ao.quantization._pt2e.quantizer.qnnpack_quantizer as qq
from torch.ao.quantization._quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e_quantizer,
)
from intel_extension_for_pytorch._inductor.compile_fx import compile_fx as ipex_compile_fx

torch._dynamo.config.verbose = True
torch._inductor.config.trace.enabled = True
torch._inductor.config.debug = True
torch._inductor.config.freezing = True

class M(torch.nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(3, 6, 2, bias=bias, stride=2, padding=0, dilation=1)

    def forward(self, x):
        return self.conv(x)

def test_qnnpack_quantizer():
    example_inputs = (torch.randn(1, 3, 224, 224),)
    m = M(bias=False).eval()
    
    m_copy = copy.deepcopy(m)

    export_model, guards = torchdynamo.export(
        m,
        *copy.deepcopy(example_inputs),
        aten_graph=True
    )

    from torch.ao.quantization._pt2e.quantizer import QNNPackQuantizer
    import torch.ao.quantization._pt2e.quantizer.qnnpack_quantizer as xiq
    quantizer = QNNPackQuantizer()
    operator_config = xiq.get_symmetric_quantization_config(is_per_channel=True)
    quantizer.set_global(operator_config)

    with torch.no_grad():
        prepare_model = prepare_pt2e_quantizer(export_model, quantizer)
        print("prepared model is: {}".format(prepare_model), flush=True)
        prepare_model(*example_inputs)

        convert_model = convert_pt2e(prepare_model)
        print("converted model is: {}".format(convert_model), flush=True)

        convert_model.eval()

        # compiler_model = torch.compile(convert_model)
        compiler_model = compile_fx(convert_model, example_inputs)
        # compiler_model = ipex_compile_fx(convert_model, example_inputs)

        print("start the first run", flush=True)
        compiler_model(*example_inputs)

        print("start the second run", flush=True)
        compiler_model(*example_inputs)

if __name__ == "__main__":
    test_qnnpack_quantizer()

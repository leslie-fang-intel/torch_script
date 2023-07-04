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

torch._dynamo.config.verbose = True
torch._inductor.config.trace.enabled = True
torch._inductor.config.debug = True
torch._inductor.config.trace.debug_log = True

torch._inductor.config.freezing = True


class M(torch.nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 7, bias=bias, stride=2, padding=3, dilation=1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 1, bias=False, stride=1, padding=0, dilation=1)
        self.conv3 = nn.Conv2d(64, 64, 1, bias=False, stride=1, padding=0, dilation=1)

    def forward(self, x):
        temp = self.maxpool(self.relu(self.conv(x)))
        return self.conv2(temp) + self.conv3(temp)

def test_qnnpack_quantizer():
    example_inputs = (torch.randn(116, 3, 224, 224).contiguous(memory_format=torch.channels_last),)
    use_bias = True
    m = M(bias=use_bias).eval()
    
    m_copy = copy.deepcopy(m)

    export_model, guards = torchdynamo.export(
        m,
        *copy.deepcopy(example_inputs),
        aten_graph=True
    )

    import torch.ao.quantization._pt2e.quantizer.x86_inductor_quantizer as xiq
    from torch.ao.quantization._pt2e.quantizer import X86InductorQuantizer
    from torch.ao.quantization._quantize_pt2e import prepare_pt2e, convert_pt2e, prepare_pt2e_quantizer
    quantizer = X86InductorQuantizer()
    operator_spec = xiq.get_default_x86_inductor_quantization_config()
    quantizer.set_global(operator_spec)

    prepare_model = prepare_pt2e_quantizer(export_model, quantizer)
    print("prepared model is: {}".format(prepare_model), flush=True)
    prepare_model(*example_inputs)

    convert_model = convert_pt2e(prepare_model)
    print("converted model is: {}".format(convert_model), flush=True)
    
    with torch.no_grad():
        convert_model.eval()

        # compiler_model = torch.compile(convert_model)
        # compiler_model = compile_fx(convert_model, example_inputs)
        compiler_model = compile_fx(convert_model, example_inputs)

        print("start the first run", flush=True)
        compiler_model(*example_inputs)

        print("start the second run", flush=True)
        out_comp = compiler_model(*example_inputs)

        out_eager = convert_model(*example_inputs)

        # print("out_comp is: {}".format(out_comp), flush=True)
        # print("out_eager[0] is: {}".format(out_eager[0]), flush=True)

        # self.assertEqual(out_eager[0], out_comp, atol=5e-2, rtol=5e-2)
        print(torch.allclose(out_eager[0], out_comp, atol=5e-2, rtol=5e-2))

if __name__ == "__main__":
    test_qnnpack_quantizer()

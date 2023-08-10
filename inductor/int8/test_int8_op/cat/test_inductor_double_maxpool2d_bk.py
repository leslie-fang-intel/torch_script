import torch
import torch.ao.quantization.fx._decomposed
import torch._dynamo as torchdynamo
import copy

torch._dynamo.config.verbose = True
torch._inductor.config.trace.enabled = True
torch._inductor.config.trace.debug_log = True
torch._inductor.config.debug = True
torch._inductor.config.freezing = True

class ConvMaxpool2d(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, 7, bias=True, stride=2, padding=3, dilation=1)
        self.conv2 = torch.nn.Conv2d(3, 64, 7, bias=True, stride=2, padding=3, dilation=1)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(128, 128, 7, bias=True, stride=2, padding=3, dilation=1)

    def forward(self, x):
        temp1 = self.relu(self.conv(x))
        temp2 = self.conv2(x + 1)
        temp3 = torch.cat((temp1, temp2), 1)
        temp4 = self.maxpool(temp3)
        temp5 = self.conv3(temp4)
        return temp5

def test_qmaxpool2d():
    input = torch.randn(16, 3, 224, 224).contiguous(memory_format=torch.channels_last)
    example_inputs = (input, )
    with torch.no_grad():
        m = ConvMaxpool2d().eval()
        export_model, guards = torchdynamo.export(
            m,
            *copy.deepcopy(example_inputs),
            aten_graph=True
        )

        import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
        from torch.ao.quantization.quantizer import X86InductorQuantizer
        from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
        quantizer = X86InductorQuantizer()
        operator_spec = xiq.get_default_x86_inductor_quantization_config()
        quantizer.set_global(operator_spec)
        prepare_model = prepare_pt2e(export_model, quantizer)
        print("prepared model is: {}".format(prepare_model), flush=True)
        prepare_model(*example_inputs)

        convert_model = convert_pt2e(prepare_model).eval()
        print("converted model is: {}".format(convert_model), flush=True)

        ref_res = convert_model(*example_inputs)

        compiler_model = torch.compile(convert_model)
        print("start the first run", flush=True)
        compiler_model(*example_inputs)

        print("start the second run", flush=True)
        out_comp = compiler_model(*example_inputs)
        print(torch.allclose(out_comp, ref_res, atol=5e-2, rtol=5e-2))


if __name__ == "__main__":
    test_qmaxpool2d()

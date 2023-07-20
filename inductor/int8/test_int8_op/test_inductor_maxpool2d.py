import torch
import torch.ao.quantization.fx._decomposed
import torch._dynamo as torchdynamo
import copy

torch._dynamo.config.verbose = True
torch._inductor.config.trace.enabled = True
torch._inductor.config.trace.debug_log = True
torch._inductor.config.debug = True
torch._inductor.config.freezing = True

# class QMaxpool2d(torch.nn.Module):
#     def __init__(self,):
#         super().__init__()
#         self.maxpool = torch.nn.MaxPool2d(3, stride=2)

#     def forward(self, x):
#         return self.maxpool(x)

def test_qmaxpool2d_op():
    input = torch.randint(0, 8, (1, 3, 8, 8), dtype=torch.uint8).contiguous(memory_format=torch.channels_last)
    with torch.no_grad():
        # m = QMaxpool2d().eval()

        # dq-maxpool-q
        dequant_input =  torch.ops.quantized_decomposed.dequantize_per_tensor.tensor(
            input,
            scale=torch.tensor(0.1, dtype=torch.float),
            zero_point=torch.tensor(1, dtype=torch.int64),
            quant_min=0,
            quant_max=255,
            dtype=torch.uint8,
        )
        dequant_maxpool2d = torch.ops.aten.max_pool2d_with_indices.default(dequant_input, [3, 3], [2, 2])[0]
        res_ref = torch.ops.quantized_decomposed.quantize_per_tensor.tensor(
            dequant_maxpool2d,
            scale=torch.tensor(0.1, dtype=torch.float),
            zero_point=torch.tensor(1, dtype=torch.int64),
            quant_min=0,
            quant_max=255,
            dtype=torch.uint8,
        )
        print("res_ref: {}".format(res_ref), flush=True)

        # maxpool uint8
        res = torch.ops.aten.max_pool2d_with_indices.default(input, [3, 3], [2, 2])[0]
        print("res: {}".format(res), flush=True)
        print(torch.allclose(res, res_ref, atol=5e-2, rtol=5e-2))

class ConvMaxpool2d(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, 7, bias=True, stride=2, padding=3, dilation=1)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        return self.maxpool(self.relu(self.conv(x)))

def test_qmaxpool2d():
    input = torch.randn(116, 3, 224, 224).contiguous(memory_format=torch.channels_last)
    example_inputs = (input, )
    with torch.no_grad():
        m = ConvMaxpool2d().eval()
        export_model, guards = torchdynamo.export(
            m,
            *copy.deepcopy(example_inputs),
            aten_graph=True
        )

        import torch.ao.quantization.pt2e.quantizer.x86_inductor_quantizer as xiq
        from torch.ao.quantization.pt2e.quantizer import X86InductorQuantizer
        from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e_quantizer
        quantizer = X86InductorQuantizer()
        operator_spec = xiq.get_default_x86_inductor_quantization_config()
        quantizer.set_global(operator_spec)
        prepare_model = prepare_pt2e_quantizer(export_model, quantizer)
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
    #test_qmaxpool2d_op()
    test_qmaxpool2d()

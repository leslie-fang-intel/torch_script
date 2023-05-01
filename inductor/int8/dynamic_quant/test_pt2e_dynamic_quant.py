import torch
from torch.ao.quantization import (
    get_default_qconfig,
    default_qconfig,
    default_dynamic_qconfig,
    per_channel_dynamic_qconfig,
    QConfigMapping,
)
from torch.ao.quantization.quantize_fx import prepare_fx, prepare_qat_fx, convert_to_reference_fx, convert_fx
import torch._dynamo as torchdynamo
from torch.ao.quantization._pt2e.quantizer import (
    OperatorConfig,
    QNNPackQuantizer,
    Quantizer,
)
from torch.ao.quantization._quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e_quantizer,
    prepare_qat_pt2e_quantizer,
)

class Mod(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            # in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1
            in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=1
        )
        #self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.conv(x)
        #x = self.conv(x)
        #return self.relu(x)

class Mod2(torch.nn.Module):
    def __init__(self,) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.maxpool = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = torch.nn.ReLU()

        self.conv4 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        x1 = self.conv(x)
        res1 = self.relu(self.conv2(x1) + self.conv3(x1))
        res2 = self.relu2(self.conv4(res1) + res1)
        return res2

class LinearRelu(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # x = torch.squeeze(x, 1)
        x2 = torch.flatten(x, start_dim=1)

        # print(x.size())
        #x = self.relu(x)
        return self.linear(x2)
        #x = self.linear(x)
        #return self.relu(x)

def test_qnnpack_quantizer():
    import copy, os
    import torch.nn as nn
    import torch.ao.nn.intrinsic.quantized.dynamic as nniqd

    # torch.backends.quantized.engine = "x86"
    # example_inputs = (torch.randn(1, 3, 224, 224),)
    # m = Mod().eval()

    # example_inputs = (torch.randn(1, 2, 2),)
    # m = LinearRelu().eval()

    import torchvision
    example_inputs = (torch.randn(1, 3, 224, 224),)
    m = torchvision.models.resnet50().eval()

    r = m(*example_inputs)
    print(r.size())

    # exit(-1)
    original_model = copy.deepcopy(m)

    import torch.ao.quantization._pt2e.quantizer.qnnpack_quantizer as qq

    quantizer = QNNPackQuantizer()
    operator_config = qq.get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True)
    # operator_config = qq.get_symmetric_quantization_config(is_per_channel=True)
    quantizer.set_global(operator_config)

    with torch.no_grad():
        # program capture
        export_model, guards = torchdynamo.export(
            m,
            *copy.deepcopy(example_inputs),
            aten_graph=True,
        )

        # original_model = copy.deepcopy(m)
        print("export_model is: {}".format(export_model), flush=True)

        prepared_model = prepare_pt2e_quantizer(export_model, quantizer)
        print("prepared_model is: {}".format(prepared_model), flush=True)
        prepared_model(*example_inputs)
        # print("prepared_model is: {}".format(prepared_model), flush=True)
        quantized_model = convert_pt2e(prepared_model)
        print("quantized_model is: {}".format(quantized_model), flush=True)

        from torch.fx.passes.graph_drawer import FxGraphDrawer
        g = FxGraphDrawer(quantized_model, "resnet50")
        g.get_dot_graph().write_svg("./nrn50_dynamic_quant.svg")

        for i in range(3):
            quantized_model(*example_inputs)

        def print_size_of_model(model):
            torch.save(model.state_dict(), "temp.p")
            print('Size (MB):', os.path.getsize("temp.p")/1e6)
            os.remove('temp.p')

        print_size_of_model(export_model)
        print_size_of_model(quantized_model)

def test_x86inductor_quantizer():
    import copy, os
    import torch.nn as nn
    import torch.ao.nn.intrinsic.quantized.dynamic as nniqd

    # torch.backends.quantized.engine = "x86"
    # example_inputs = (torch.randn(1, 3, 224, 224),)
    # m = Mod().eval()

    # example_inputs = (torch.randn(1, 2, 2),)
    # m = LinearRelu().eval()

    import torchvision
    example_inputs = (torch.randn(1, 3, 224, 224),)
    m = torchvision.models.resnet50().eval()

    # example_inputs = (torch.randn(1, 3, 16, 16),)
    # m = Mod2().eval()

    r = m(*example_inputs)
    print(r.size())

    # exit(-1)
    original_model = copy.deepcopy(m)

    import torch.ao.quantization._pt2e.quantizer.qnnpack_quantizer as qq
    import torch.ao.quantization._pt2e.quantizer.x86_inductor_quantizer as xiq
    from torch.ao.quantization._pt2e.quantizer import  X86InductorQuantizer
    quantizer = X86InductorQuantizer()
    operator_config = xiq.get_default_x86_inductor_quantization_config(is_dynamic=True)
    # operator_config = qq.get_symmetric_quantization_config(is_per_channel=True)
    quantizer.set_global(operator_config)

    with torch.no_grad():
        # program capture
        export_model, guards = torchdynamo.export(
            m,
            *copy.deepcopy(example_inputs),
            aten_graph=True,
        )

        # original_model = copy.deepcopy(m)
        print("export_model is: {}".format(export_model), flush=True)

        prepared_model = prepare_pt2e_quantizer(export_model, quantizer)
        print("prepared_model is: {}".format(prepared_model), flush=True)
        prepared_model(*example_inputs)
        # print("prepared_model is: {}".format(prepared_model), flush=True)
        quantized_model = convert_pt2e(prepared_model)
        print("quantized_model is: {}".format(quantized_model), flush=True)

        from torch.fx.passes.graph_drawer import FxGraphDrawer
        g = FxGraphDrawer(quantized_model, "resnet50")
        g.get_dot_graph().write_svg("./nrn50_dynamic_quant.svg")

        # for i in range(3):
        #     quantized_model(*example_inputs)

        def print_size_of_model(model):
            torch.save(model.state_dict(), "temp.p")
            print('Size (MB):', os.path.getsize("temp.p")/1e6)
            os.remove('temp.p')

        print_size_of_model(export_model)
        print_size_of_model(quantized_model)

if __name__ == "__main__":
    # Public Branch:
    # PyTorch: b2ef4c942779bc656cc61ff37659d927914b950e
    # PT2E not support dynamic quant path yet.

    # Test Branch: 
    # PyTorch: leslie/test_pt2e_dynamic_quant 5df926016c777e279114fadc2e194fbbb44f45bb
    # test_qnnpack_quantizer()
    test_x86inductor_quantizer()
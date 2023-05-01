import torch
from torch.ao.quantization import (
    get_default_qconfig,
    default_qconfig,
    default_dynamic_qconfig,
    per_channel_dynamic_qconfig,
    QConfigMapping,
)
from torch.ao.quantization.quantize_fx import prepare_fx, prepare_qat_fx, convert_to_reference_fx, convert_fx
import intel_extension_for_pytorch as ipex

class Mod(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            # in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1
            in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=1
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        #return self.conv(x)
        x = self.conv(x)
        return self.relu(x)

class LinearRelu(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 5)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.linear(x)
        #x = self.linear(x)
        #return self.relu(x)

def test_ipex_dynamic_quant():
    # import torchvision
    # m = torchvision.models.resnet50().eval()
    # Refer to the doc
    # https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/features/int8_overview.html
    # Dynamic quant doesn't support conv
    m = LinearRelu().eval()

    # example_inputs = (torch.randn(1, 3, 224, 224),)
    example_inputs = (torch.randn(5, 5),)
    import intel_extension_for_pytorch as ipex
    from intel_extension_for_pytorch.quantization import prepare, convert

    dynamic_qconfig = ipex.quantization.default_dynamic_qconfig
    # equal to 
    # QConfig(activation=PlaceholderObserver.with_args(dtype=torch.float, compute_dtype=torch.quint8),
    #         weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))

    print("---- start the prepare ----", flush=True)
    prepared_model = prepare(m, dynamic_qconfig, example_inputs=example_inputs)
    print("---- finish the prepare ----", flush=True)


    # make sure the example_inputs's size is same as the real input's size
    print("---- start the convert ----", flush=True)
    convert_model = convert(prepared_model)
    print("---- finish the convert ----", flush=True)
    # Optional: convert the model to traced model
    #with torch.no_grad():
    #    traced_model = torch.jit.trace(convert_model, example_input)
    #    traced_model = torch.jit.freeze(traced_model)

    # or save the model to deploy
    # traced_model.save("quantized_model.pt")
    # quantized_model = torch.jit.load("quantized_model.pt")
    # quantized_model = torch.jit.freeze(quantized_model.eval())
    # ...
    # for inference
    for i in range(3):
        print("---- step: {}".format(i), flush=True)
        y = convert_model(*example_inputs)

    # print("convert_model is: {}".format(convert_model))

if __name__ == "__main__":
    test_ipex_dynamic_quant()
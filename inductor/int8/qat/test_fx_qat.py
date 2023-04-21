# qconfig = get_default_qconfig("x86")
# qconfig_mapping = QConfigMapping().set_global(qconfig)
# m = prepare_qat_fx(model, qconfig_mapping, example_inputs=example_inputs)
# m(*example_inputs)
# m = convert_fx(m)
# torch.jit.trace(m)

import torch
from torch.ao.quantization import (
    get_default_qconfig,
    QConfigMapping,
)
from torch.ao.quantization.quantize_fx import prepare_fx, prepare_qat_fx, convert_to_reference_fx, convert_fx


class Mod(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            # in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        return self.relu(x)

def ipex_test():
    pass
def test():
    torch.backends.quantized.engine = "x86"
    # example_inputs = (torch.randn(1, 1, 224, 224),)
    example_inputs = (torch.randn(1, 3, 16, 16).to(memory_format=torch.channels_last),)
    m = Mod()
    m(*example_inputs)

    qconfig = get_default_qconfig("x86")
    qconfig_mapping = QConfigMapping().set_global(qconfig)
    m = prepare_qat_fx(m, qconfig_mapping, example_inputs=example_inputs)
    print("prepare qat model is: {}".format(m), flush=True)
    # QAT
    for i in range(3):
        m(torch.randn(1, 3, 16, 16).to(memory_format=torch.channels_last))
    m = convert_fx(m)
    print("converted model is: {}".format(m), flush=True)
    
    jit_traced_model = torch.jit.trace(m, example_inputs).eval()
    jit_traced_model = torch.jit.freeze(jit_traced_model)
    for i in range(3):
        jit_traced_model(*example_inputs)
    
    print(jit_traced_model.graph_for(*example_inputs), flush=True)

if __name__ == "__main__":
    test()

import torch
from torch.ao.quantization import (
    get_default_qconfig,
    default_qconfig,
    default_dynamic_qconfig,
    per_channel_dynamic_qconfig,
    QConfigMapping,
)
from torch.ao.quantization.quantize_fx import prepare_fx, prepare_qat_fx, convert_to_reference_fx, convert_fx

class Mod(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            # in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1
            in_channels=112, out_channels=1024, kernel_size=5, stride=1, padding=1
        )
        #self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.conv(x)
        #x = self.conv(x)
        #return self.relu(x)

class LinearRelu(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 5)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.linear(x)
        #x = self.linear(x)
        #return self.relu(x)

def test():
    import copy, os
    import torch.nn as nn
    import torch.ao.nn.intrinsic.quantized.dynamic as nniqd

    # torch.backends.quantized.engine = "x86"
    # example_inputs = (torch.randn(1, 1, 224, 224),)
    #example_inputs = (torch.randn(1, 112, 16, 16).to(memory_format=torch.channels_last),)
    example_inputs = (torch.randn(5, 5),)
    m = LinearRelu().eval()
    model = copy.deepcopy(m)
    m(*example_inputs)

    # # qconfig = get_default_qconfig("x86")
    # qconfig = default_dynamic_qconfig
    # qconfig_mapping = QConfigMapping().set_global(qconfig)

    global_qconfig = default_qconfig
    object_type_qconfig = per_channel_dynamic_qconfig

    # qconfig_mapping = {
    #     "": object_type_qconfig,
    #     "object_type": [(nn.Conv2d, object_type_qconfig), (nn.ReLU, object_type_qconfig),]}

    qconfig_mapping = QConfigMapping().set_global(torch.ao.quantization.default_dynamic_qconfig)

    m = prepare_fx(m, qconfig_mapping, example_inputs=example_inputs)
    print("prepare dynamic model is: {}".format(m), flush=True)
    # No need of calibration for dynamic quant
    quantized_model = convert_fx(m)
    print("converted dynamic model is: {}".format(m), flush=True)

    quantized_model(*example_inputs)

    for node in quantized_model.graph.nodes:
        print(type(node.target))
        #print(node == ns.call_module(nniqd.LinearReLU))
        print(node.target)
        print(type(quantized_model.linear))
    
    # jit_traced_model = torch.jit.trace(m, example_inputs).eval()
    # jit_traced_model = torch.jit.freeze(jit_traced_model)
    # for i in range(3):
    #     jit_traced_model(*example_inputs)
    
    # print(jit_traced_model.graph_for(*example_inputs), flush=True)

    def print_size_of_model(model):
        torch.save(model.state_dict(), "temp.p")
        print('Size (MB):', os.path.getsize("temp.p")/1e6)
        os.remove('temp.p')

    print_size_of_model(model)
    print_size_of_model(quantized_model)

if __name__ == "__main__":
    test()
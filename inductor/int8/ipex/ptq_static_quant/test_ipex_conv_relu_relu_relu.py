import torch
import torch.nn as nn
import intel_extension_for_pytorch as ipex
from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, QConfig
import torch._dynamo as torchdynamo
from torch.ao.quantization._quantize_pt2e import (
    prepare_pt2e_quantizer,
    convert_pt2e
)
import copy

class Mod(nn.Module):
    def __init__(self, ):
        super(Mod, self).__init__()
        self.conv = torch.nn.Conv2d(2, 2, 1)

    def forward(self, x):
        x = self.conv(x)
        x2 = torch.relu(x)
        x3 = torch.relu(x2)
        x4 = x3.relu()
        return x4

class Mod2(nn.Module):
    def __init__(self, ):
        super(Mod2, self).__init__()

    def forward(self, x):
        x = x.relu()
        x = x.relu()
        return x

def test_ipex():
    # model = Mod().eval()
    model = Mod2().eval()
    with torch.no_grad():
        x = [torch.rand(1, 2, 14, 14),]
        y = model(*x)
        default_static_qconfig = QConfig(
            activation= MinMaxObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.quint8),
            weight= PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))
        model = ipex.quantization.prepare(model, default_static_qconfig, x, inplace=False) 
        # do calibration
        y = model(*x)
        # jit trace to insert quant/dequant
        convert_model = ipex.quantization.convert(model, inplace=False)
        traced_model = torch.jit.trace(convert_model, x)
        traced_model = torch.jit.freeze(traced_model)
        # Warm up
        for _ in range(3):
            traced_model(*x)
        print(traced_model.graph_for(*x))

def test_ipex_quantization_flow2():
    model = Mod().eval()
    # model = Mod2().eval()
    with torch.no_grad():
        x = [torch.rand(1, 2, 14, 14),]
        y = model(*x)
        exported_model, guards = torchdynamo.export(
            model,
            *copy.deepcopy(x),
            aten_graph=True
        )
        exported_model = exported_model.eval()

        quantizer = ipex.quantization.IPEXQuantizer()
        quantizer.set_global(ipex.quantization.get_default_ipex_quantization_config())
        prepared_model = prepare_pt2e_quantizer(exported_model, quantizer)
        # do calibration
        y = prepared_model(*x)
        convert_model = convert_pt2e(prepared_model)
        traced_model = torch.jit.trace(convert_model, x)
        traced_model = torch.jit.freeze(traced_model)
        # Warm up
        for _ in range(3):
            traced_model(*x)
        print(traced_model.graph_for(*x))

if __name__ == "__main__":
    # test_ipex()
    test_ipex_quantization_flow2()

import torch
import intel_extension_for_pytorch as ipex
from torch.ao.quantization import (
    MinMaxObserver,
    PerChannelMinMaxObserver,
    QConfig,
)
import torch.nn as nn
from utils_vis import make_dot, draw

default_weight_observer = PerChannelMinMaxObserver.with_args(
    dtype=torch.qint8, qscheme=torch.per_channel_symmetric
)

qconfig = QConfig(
    activation=MinMaxObserver.with_args(
        qscheme=torch.per_tensor_affine, dtype=torch.quint8
    ),
    weight=default_weight_observer,
)

def test_linear_int8_in_bf16_out():
    class M(nn.Module):
        def __init__(self):
            super(M, self).__init__()
            self.linear1 = nn.Linear(15, 20, bias=True)
            self.dropout = nn.Dropout()
            self.linear2 = nn.Linear(15, 20, bias=True)

        def forward(self, x, y):
            x = self.linear1(x)
            x = self.dropout(x)
            z = self.linear2(y) + x
            return z

    x = torch.randn(2, 15)
    y = torch.randn(2, 15)
    model = M()
    with torch.no_grad(), torch._jit_internal._disable_emit_hooks():
        ipex.nn.utils._model_convert.replace_dropout_with_identity(model)
        model = ipex.quantization.prepare(
            model, qconfig, (x, y), inplace=False,
        )
        # Calibration
        model(x, y)
        with torch.cpu.amp.autocast():
            convert_model = ipex.quantization.convert(
                model, inplace=False
            )
            traced_model = torch.jit.trace(convert_model, (x, y)).eval()
            traced_model = torch.jit.freeze(traced_model)
        
        for i in range(3):
            traced_model(x, y)
        
        print(traced_model.graph_for(x, y))
        fwd_graph = traced_model.graph_for(x, y)
        draw(fwd_graph).render("int-mixed-bf16")

if __name__ == "__main__":
    # ipex._C.set_llga_fp32_bf16_enabled(False)
    test_linear_int8_in_bf16_out()

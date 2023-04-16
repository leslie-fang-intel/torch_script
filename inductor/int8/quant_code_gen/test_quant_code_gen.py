import copy
from torch import _dynamo, _inductor
from torch._inductor import config
import logging
import torch
import torch._dynamo as torchdynamo

from torch.ao.quantization.backend_config._inductor_pt2e import get_inductor_pt2e_backend_config
from torch.ao.quantization.quantize_fx import prepare_fx, convert_to_reference_fx
from torch.ao.quantization._quantize_pt2e import prepare_pt2e, convert_pt2e

from torch.ao.quantization import (
    get_default_qconfig,
    QConfigMapping,
)

from torch._inductor.compile_fx import compile_fx

if __name__ == "__main__":
    # torch._dynamo.config.log_level = logging.DEBUG
    torch._dynamo.config.verbose = True
    torch._inductor.config.trace.enabled = True
    torch._inductor.config.debug = True

    # class Mod(torch.nn.Module):
    #     def __init__(self) -> None:
    #         super().__init__()
    #         self.relu = torch.nn.ReLU()

    #     def forward(self, x):
    #         return self.relu(x + x)
    class Mod(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(
                in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
            )
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            x = self.conv(x)
            return self.relu(x)

    torch.backends.quantized.engine = "x86"
    example_inputs = (torch.randn(1, 3, 224, 224),)
    m = Mod().eval()
    # program capture
    m, guards = torchdynamo.export(
        m,
        *copy.deepcopy(example_inputs),
        aten_graph=True,
        tracing_mode="real",
    )

    print("model after torchdynamo export is: {}".format(m), flush=True)

    backend_config = get_inductor_pt2e_backend_config()
    qconfig = get_default_qconfig("x86")
    qconfig_mapping = QConfigMapping().set_global(qconfig)
    before_fusion_result = m(*example_inputs)

    m = prepare_pt2e(m, qconfig_mapping, example_inputs, backend_config)
    after_prepare_result = m(*example_inputs)

    print("model after prepare_pt2e is: {}".format(m), flush=True)

    m = convert_pt2e(m)
    after_quant_result = m(*example_inputs)

    print("model after convert_pt2e is: {}".format(m), flush=True)

    # A few ops in EXIR are not supported. Set nopython=False to make it work
    run = torch._dynamo.optimize(compile_fx, nopython=False)(m)
    # first run
    inductor_result = run(*example_inputs)

    module_result = m(*example_inputs)
    torch.allclose(inductor_result, module_result, rtol=1e-05, atol=1e-08)

    # second run
    inductor_result = run(*example_inputs)

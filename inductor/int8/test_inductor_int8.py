
import torch
import torch._dynamo as torchdynamo
from torch.ao.quantization import (
    get_default_qconfig,
    QConfigMapping,
)
from torch.testing._internal.common_quantized import (
    override_quantized_engine,
)
from torch.ao.quantization.backend_config import (
    get_qnnpack_backend_config,
)
from torch.ao.quantization.backend_config._qnnpack_pt2e import get_qnnpack_pt2e_backend_config
from torch.ao.quantization.backend_config._inductor_pt2e import get_inductor_pt2e_backend_config
from torch.ao.quantization.quantize_fx import prepare_fx, convert_to_reference_fx
from torch.ao.quantization._quantize_pt2e import prepare_pt2e, convert_pt2e
from torch.ao.ns.fx.utils import (
    compute_sqnr,
)
from torch._inductor.compile_fx import compile_fx

def test_inductor_int8_relu():
    import copy
    from torch import _dynamo, _inductor
    from torch._inductor import config
    import logging

    torch._dynamo.config.log_level = logging.DEBUG
    torch._dynamo.config.verbose = True
    torch._inductor.config.trace.enabled = True
    torch._inductor.config.debug = True

    class Mod(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            return self.relu(x + x)

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
    # self.assertEqual(inductor_result, module_result)
    torch.allclose(inductor_result, module_result, rtol=1e-05, atol=1e-08)

    # second run
    inductor_result = run(*example_inputs)


def test_inductor_int8_conv_relu():
    import copy
    from torch import _dynamo, _inductor
    from torch._inductor import config
    import logging
    import numpy as np
    import random

    local_seed = 2023
    torch.manual_seed(local_seed) # Set PyTorch seed
    np.random.seed(seed=local_seed) # Set Numpy seed
    random.seed(local_seed) # Set the Python seed

    torch._dynamo.config.log_level = logging.DEBUG
    torch._dynamo.config.verbose = True
    torch._inductor.config.trace.enabled = True
    torch._inductor.config.debug = True

    class Mod(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(
                in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1
            )
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            x = self.conv(x)
            return self.relu(x)

    torch.backends.quantized.engine = "x86"
    example_inputs = (torch.randn(1, 1, 224, 224),)
    m = Mod().eval()
    # program capture
    m, guards = torchdynamo.export(
        m,
        *copy.deepcopy(example_inputs),
        aten_graph=True,
        tracing_mode="real",
    )

    m = m.eval()
    print("model after torchdynamo export is: {}".format(m), flush=True)

    backend_config = get_inductor_pt2e_backend_config()
    qconfig = get_default_qconfig("x86")
    qconfig_mapping = QConfigMapping().set_global(qconfig)
    before_fusion_result = m(*example_inputs)

    m = prepare_pt2e(m, qconfig_mapping, example_inputs, backend_config)
    after_prepare_result = m(*example_inputs)
    print("model after prepare_pt2e is: {}".format(m), flush=True)

    print("check the result after prepare: {}".format(
        torch.allclose(before_fusion_result, after_prepare_result)),
        flush=True)

    m = convert_pt2e(m)
    after_quant_result = m(*example_inputs)
    print("model after convert_pt2e is: {}".format(m), flush=True)

    print("check the result after convert: {}".format(
        torch.allclose(before_fusion_result, after_quant_result, rtol=5e-02, atol=5e-02)),
        flush=True)

    # A few ops in EXIR are not supported. Set nopython=False to make it work
    run = torch._dynamo.optimize(compile_fx, nopython=False)(m)

    # first run
    print("start the first run", flush=True)
    inductor_result = run(*example_inputs)

    print("example_inputs is: {}".format(example_inputs[0]), flush=True)
    print("before_fusion_result is: {}".format(before_fusion_result), flush=True)
    print("after_quant_result is: {}".format(after_quant_result), flush=True)
    print("inductor first run result is: {}".format(inductor_result), flush=True)

    # second run
    print("start the second run", flush=True)
    inductor_result = run(*example_inputs)

    print("inductor second run result is: {}".format(inductor_result), flush=True)

    # np.testing.assert_array_almost_equal(res_ref.cpu().numpy(),
    # res_quantized.cpu().numpy(), decimal=2)

def test_inductor_int8_conv_relu_v2():
    import copy
    from torch import _dynamo, _inductor
    from torch._inductor import config
    import logging
    import numpy as np
    import random

    torch._dynamo.config.log_level = logging.DEBUG
    torch._dynamo.config.verbose = True
    torch._inductor.config.trace.enabled = True
    torch._inductor.config.debug = True

    class Mod(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(
                in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1
            )
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            x = self.conv(x)
            return self.relu(x)

    torch.backends.quantized.engine = "x86"
    example_inputs = (torch.randn(1, 1, 224, 224),)
    m = Mod().eval()
    # program capture
    m, guards = torchdynamo.export(
        m,
        *copy.deepcopy(example_inputs),
        aten_graph=True,
        tracing_mode="real",
    )

    m = m.eval()
    backend_config = get_inductor_pt2e_backend_config()
    qconfig = get_default_qconfig("x86")
    qconfig_mapping = QConfigMapping().set_global(qconfig)
    before_fusion_result = m(*example_inputs)

    m = prepare_pt2e(m, qconfig_mapping, example_inputs, backend_config)
    m = convert_pt2e(m)
    # A few ops in EXIR are not supported. Set nopython=False to make it work
    run = torch._dynamo.optimize(compile_fx, nopython=False)(m)

    # first run
    inductor_result = run(*example_inputs)

    # second run
    inductor_result = run(*example_inputs)

if __name__ == "__main__":
    # test_inductor_int8_relu()
    test_inductor_int8_conv_relu_v2()

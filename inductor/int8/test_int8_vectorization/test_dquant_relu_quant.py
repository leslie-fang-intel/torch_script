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
from torch.ao.quantization.backend_config._x86_inductor_pt2e import get_x86_inductor_pt2e_backend_config
from torch.ao.quantization.quantize_fx import prepare_fx, convert_to_reference_fx
from torch.ao.quantization._quantize_pt2e import prepare_pt2e, convert_pt2e
from torch.ao.ns.fx.utils import (
    compute_sqnr,
)
from torch._inductor.compile_fx import compile_fx

from torch._decomp import get_decompositions
from torch.fx.experimental.proxy_tensor import make_fx

quant_decomp = get_decompositions(
    [
        torch.ops.quantized_decomposed.quantize_per_tensor,
        torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
        torch.ops.quantized_decomposed.dequantize_per_tensor,
        torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
    ]
)

def test_quant_dequant_relu_quant_dequant_model():
    torch._inductor.config.cpp.enable_kernel_profile = True
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

    # torch._dynamo.config.log_level = logging.DEBUG
    torch._dynamo.config.verbose = True
    torch._inductor.config.trace.enabled = True
    torch._inductor.config.debug = True

    class Mod(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # self.conv = torch.nn.Conv2d(
            #     # in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1
            #     in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
            # )
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            return self.relu(x)
        torch.backends.quantized.engine = "x86"
    # example_inputs = (torch.randn(1, 1, 224, 224),)
    example_inputs = (torch.randn(1, 3, 16, 16).to(memory_format=torch.channels_last),)
    m = Mod().eval()
    # program capture
    
    
    tracing_mode = "real"
    #tracing_mode = "symbolic"
    m, guards = torchdynamo.export(
        m,
        *copy.deepcopy(example_inputs),
        aten_graph=True,
        tracing_mode=tracing_mode,
    )

    m = m.eval()
    print("model after torchdynamo export is: {}".format(m), flush=True)
    print("guards is: {}".format(guards), flush=True)

    # m(torch.randn(2, 3, 16, 16).to(memory_format=torch.channels_last))
    # exit(-1)

    backend_config = get_x86_inductor_pt2e_backend_config()
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

    m = make_fx(m, decomposition_table=quant_decomp)(*copy.deepcopy(example_inputs))

    print("model after convert_pt2e is: {}".format(m), flush=True)

    print("check the result after convert: {}".format(
        torch.allclose(before_fusion_result, after_quant_result, rtol=5e-02, atol=5e-02)),
        flush=True)

    run = compile_fx(m, example_inputs)

    print("start the first run", flush=True)
    inductor_result = run(*example_inputs)

    print("start the second run", flush=True)
    inductor_result = run(*example_inputs)

def test_dequant_relu_quant_model():
    torch._inductor.config.cpp.enable_kernel_profile = True
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

    # torch._dynamo.config.log_level = logging.DEBUG
    torch._dynamo.config.verbose = True
    torch._inductor.config.trace.enabled = True
    torch._inductor.config.debug = True

    class Mod(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            x = torch.ops.quantized_decomposed.dequantize_per_tensor(x, 1.0 / 19.228647416108554, 62, 0, 127, torch.uint8)
            y = self.relu(x)
            y2 = torch.ops.quantized_decomposed.quantize_per_tensor(y, input_scale, input_zp, 0, 127, torch.uint8)
            return y2
    
    torch.backends.quantized.engine = "x86"
    # example_inputs = (torch.randn(1, 1, 224, 224),)

    input = torch.randn(1, 3, 16, 16).to(memory_format=torch.channels_last)
    input_scale = 1.0 / 19.228647416108554
    input_zp = 62
    q_input = torch.ops.quantized_decomposed.quantize_per_tensor(input, input_scale, input_zp, 0, 127, torch.uint8)
    example_inputs = (q_input,)
    m = Mod().eval()
    # program capture
    
    
    tracing_mode = "real"
    #tracing_mode = "symbolic"
    m, guards = torchdynamo.export(
        m,
        *copy.deepcopy(example_inputs),
        aten_graph=True,
        tracing_mode=tracing_mode,
    )

    m = m.eval()
    print("model after torchdynamo export is: {}".format(m), flush=True)
    print("guards is: {}".format(guards), flush=True)

    # m(torch.randn(2, 3, 16, 16).to(memory_format=torch.channels_last))
    # exit(-1)

    # backend_config = get_x86_inductor_pt2e_backend_config()
    # qconfig = get_default_qconfig("x86")
    # qconfig_mapping = QConfigMapping().set_global(qconfig)
    # before_fusion_result = m(*example_inputs)

    # m = prepare_pt2e(m, qconfig_mapping, example_inputs, backend_config)
    # after_prepare_result = m(*example_inputs)
    # print("model after prepare_pt2e is: {}".format(m), flush=True)

    # print("check the result after prepare: {}".format(
    #     torch.allclose(before_fusion_result, after_prepare_result)),
    #     flush=True)
    
    # m = convert_pt2e(m)
    # after_quant_result = m(*example_inputs)

    m = make_fx(m, decomposition_table=quant_decomp)(*copy.deepcopy(example_inputs))

    print("model after convert_pt2e is: {}".format(m), flush=True)

    # print("check the result after convert: {}".format(
    #     torch.allclose(before_fusion_result, after_quant_result, rtol=5e-02, atol=5e-02)),
    #     flush=True)

    run = compile_fx(m, example_inputs)

    print("start the first run", flush=True)
    inductor_result = run(*example_inputs)

    print("start the second run", flush=True)
    inductor_result = run(*example_inputs)

if __name__ == "__main__":
    #test_quant_dequant_relu_quant_dequant_model()
    test_dequant_relu_quant_model()


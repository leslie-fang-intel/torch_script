
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

def test_inductor_int8_relu():
    import copy
    from torch import _dynamo, _inductor
    from torch._inductor import config
    import logging

    # torch._dynamo.config.log_level = logging.DEBUG
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

    run = compile_fx_quantization(m, example_inputs)

    # first run
    inductor_result = run(*example_inputs)

    module_result = m(*example_inputs)
    # self.assertEqual(inductor_result, module_result)
    torch.allclose(inductor_result[0], module_result[0], rtol=1e-05, atol=1e-08)

    # second run
    inductor_result = run(*example_inputs)


def test_inductor_int8_conv_relu():
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
            self.conv = torch.nn.Conv2d(
                # in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1
                in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
            )
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            x = self.conv(x)
            return self.relu(x)

    with torch.no_grad():
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

        # m = make_fx(m, decomposition_table=quant_decomp)(*copy.deepcopy(example_inputs))

        after_quant_result = m(*example_inputs)
        print("model after convert_pt2e is: {}".format(m), flush=True)

        print("check the result after convert: {}".format(
            torch.allclose(before_fusion_result, after_quant_result, rtol=5e-02, atol=5e-02)),
            flush=True)

        run = compile_fx(m, example_inputs)

        print("start the first run", flush=True)
        inductor_result = run(*example_inputs)

        print("start the second run", flush=True)
        inductor_result = run(*example_inputs)

        print(torch.allclose(before_fusion_result, inductor_result, rtol=5e-02, atol=5e-02), flush=True)

def test_inductor_int8_conv_relu_v2():
    import copy
    from torch import _dynamo, _inductor
    from torch._inductor import config
    import logging
    import numpy as np
    import random

    # torch._dynamo.config.log_level = logging.DEBUG
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
    # Step1: program capture with EXIR
    m, guards = torchdynamo.export(
        m,
        *copy.deepcopy(example_inputs),
        aten_graph=True,
        tracing_mode="real",
    )
    print(m)
    # Step2: Prepare phase to insert observer for calibration
    # TODO: Use QTagger in the future
    backend_config = get_inductor_pt2e_backend_config()
    qconfig_mapping = QConfigMapping().set_global(get_default_qconfig("x86"))
    m = prepare_pt2e(m, qconfig_mapping, example_inputs, backend_config)
    # Step3: Convert to reference quantized model
    m = convert_pt2e(m)
    # Step4: Lowering to inductor: Op fusion; External call of conv/matmul; Decompose and Code-Gen.
    run = compile_fx(m, example_inputs)


    #run = compile_fx_quantization(m, example_inputs)

    # first run
    inductor_result = run(*example_inputs)

    # second run
    inductor_result = run(*example_inputs)

def test_op():
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
    
    m = Mod().eval()
    weight = m.conv.weight
    bias = m.conv.bias
    in_channels = 3
    out_channels = 16
    kernel_size = 3
    stride = (1, 1)
    padding = (1, 1)
    dilation = (1, 1)
    x = torch.randn(1, 3, 16, 16).to(memory_format=torch.channels_last)
    ref_result = torch.ops.aten.convolution.default(x, weight, bias, stride, padding, dilation, False, padding, 1)

    weight_scales = [0.00141001,
                     0.00135255,
                     0.00147836,
                     0.00141083,
                     0.00150533,
                     0.00147467,
                     0.00145911,
                     0.00150878,
                     0.00137985,
                     0.00150667,
                     0.00146871,
                     0.00128621,
                     0.00146854,
                     0.0013474,
                     0.00150807,
                     0.00150064]
    weight_scales = torch.DoubleTensor(weight_scales)
    weight_zero_points = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]
    weight_zero_point = torch.IntTensor(weight_zero_points)
    x_scale = 0.05
    x_zp = 26
    quant_weight = torch.ops.quantized_decomposed.quantize_per_channel(
        weight,
        weight_scales,
        weight_zero_point,
        0,
        -128,
        127,
        torch.int8
        )
    packed_w, packed_bias = torch.ops.quantized.conv_prepack_cpu_tensor(quant_weight,
                                                        weight_scales,
                                                        x.size(),
                                                        x_scale, # x_scale
                                                        x_zp, # x_zp
                                                        bias,
                                                        stride,
                                                        padding,
                                                        dilation,
                                                        1)
    result2 = torch.ops.quantized.conv_prepacked_weight(
        x,
        weight,
        bias,
        packed_w,
        weight_scales,
        weight_zero_point,
        0,
        packed_bias,
        stride,
        padding,
        dilation,
        False,
        padding,
        1,
        torch.tensor(x_scale),
        torch.tensor(x_zp)
    )
    

    print("ref_result is: {}".format(ref_result), flush=True)
    print("result2 is: {}".format(result2), flush=True)
    print(torch.allclose(ref_result, result2, rtol=5e-02, atol=5e-02), flush=True)

if __name__ == "__main__":
    #test_inductor_int8_relu()
    test_inductor_int8_conv_relu()
    
    # exit(0)

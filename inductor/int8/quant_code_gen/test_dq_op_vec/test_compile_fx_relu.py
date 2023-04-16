import torch
from torch import _dynamo, _inductor
from torch._inductor import config
import logging
import numpy as np
import random
from torch._inductor import codecache, config, metrics, test_operators
from torch.fx.experimental.proxy_tensor import make_fx
from torch._inductor.compile_fx import (
    compile_fx,
    compile_fx_inner,
    complex_memory_overlap,
)
from torch.ao.quantization._pt2e.quantizer import X86InductorQuantizer
from torch.ao.quantization._quantize_pt2e import prepare_pt2e, convert_pt2e, prepare_pt2e_quantizer
import torch._dynamo as torchdynamo
import copy

# torch._dynamo.config.log_level = logging.DEBUG
torch._dynamo.config.verbose = True
torch._inductor.config.trace.enabled = True
torch._inductor.config.trace.debug_log = True
torch._inductor.config.debug = True

local_seed = 2023
torch.manual_seed(local_seed) # Set PyTorch seed
np.random.seed(seed=local_seed) # Set Numpy seed
random.seed(local_seed) # Set the Python seed

relu = torch.nn.ReLU()

def test1():
    class M(torch.nn.Module):
        def __init__(self, ):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 16, (3, 3), stride=(1, 1), padding=(1, 1))
            self.relu = torch.nn.ReLU()
            self.maxpool = torch.nn.MaxPool2d(3, stride=2)
            self.relu2 = torch.nn.ReLU()

        def forward(self, x):
            # return self.maxpool(self.conv1(x))
            return self.relu2(self.relu(self.conv1(x)))
            # return x + x

    m = M().to(memory_format=torch.channels_last).eval()
    # **TODO** Also test channel_last format
    example_inputs = (torch.randn(2, 3, 224, 224).contiguous(memory_format=torch.channels_last),)
    # program capture
    # **TODO** Add testcase for tracing_mode="symbolic" after fix issue:
    # https://github.com/pytorch/pytorch/issues/96274
    export_module, guards = torchdynamo.export(
        m,
        *copy.deepcopy(example_inputs),
        aten_graph=True,
        tracing_mode="real",
    )

    print("m after torchdynamo export is: {}".format(export_module), flush=True)
    import torch.ao.quantization._pt2e.quantizer.x86_inductor_quantizer as xiq
    quantizer = X86InductorQuantizer()
    operator_spec = xiq.get_default_x86_inductor_operator_spec()
    quantizer.set_global(operator_spec)

    m = prepare_pt2e_quantizer(export_module, quantizer)
    print("m after prepare_pt2e_quantizer is: {}".format(m), flush=True)
    after_prepare_result = m(*example_inputs)
    m = convert_pt2e(m)
    print("m after convert_pt2e is: {}".format(m), flush=True)

    # for node in m.graph.nodes:
    #     print("node is:{}, target is:{}".format(node, node.target), flush=True)
    #     if node.target == torch.ops.quantized_decomposed.dequantize_per_tensor:
    #         print("--- hit -----", flush=True)
    #         for arg in node.args:
    #             print("arg is: {}, type(args) is:{}".format(arg, type(arg)), flush=True)
    #         # scale
    #         scale_node = node.args[1]
    #         print("scale_node.op is: {}, scale_node.target is: {}".format(scale_node.op, scale_node.target), flush=True)
    #         assert (scale_node.op == "get_attr")
    #         scale_scalar_tensor = getattr(m, scale_node.target)
    #         print("scale_scalar_tensor is: {}".format(scale_scalar_tensor), flush=True)
    #         print("scale_scalar_tensor.item() is: {}".format(scale_scalar_tensor.item()), flush=True)
    #         print("type(scale_scalar_tensor.item()) is: {}".format(type(scale_scalar_tensor.item())), flush=True)
    #         scale_scalar = scale_scalar_tensor.item()
    #         node.update_arg(1, scale_scalar)

    #         # zp
    #         zp_node = node.args[2]
    #         zp_scalar_tensor = getattr(m, zp_node.target)
    #         zp_scalar = zp_scalar_tensor.item()
    #         node.update_arg(2, zp_scalar)
    #     elif node.target == torch.ops.quantized_decomposed.quantize_per_tensor:
    #         # scale
    #         scale_node = node.args[1]
    #         assert (scale_node.op == "get_attr")
    #         scale_scalar_tensor = getattr(m, scale_node.target)
    #         scale_scalar = scale_scalar_tensor.item()
    #         node.update_arg(1, scale_scalar)

    #         # zp
    #         zp_node = node.args[2]
    #         zp_scalar_tensor = getattr(m, zp_node.target)
    #         zp_scalar = zp_scalar_tensor.item()
    #         node.update_arg(2, zp_scalar)            
              
    
    # m.graph.eliminate_dead_code()
    # m.graph.lint()
    # m.recompile()


    print("print again", flush=True)
    for node in m.graph.nodes:
        print("node is:{}, target is:{}".format(node, node.target), flush=True)
        if node.target == torch.ops.quantized_decomposed.dequantize_per_tensor:
            print("--- hit -----", flush=True)
            for arg in node.args:
                print("arg is: {}, type(args) is:{}".format(arg, type(arg)), flush=True)

    opt_fn = compile_fx(m, example_inputs)

    print("---- first run ----", flush=True)
    inductor_res = opt_fn(*example_inputs)

    print("---- second run ----", flush=True)
    inductor_res = opt_fn(*example_inputs)

    return

    zero_point = 100
    scale = 0.001

    # zero_point = torch.tensor(100)
    # scale = torch.tensor(0.001)

    print("x is: {}".format(x), flush=True)

    opt_fn = torch._dynamo.optimize("inductor")(fn)
    
    # traced = make_fx(fn)(x, zero_point, scale)
    # print("traced graph is: {}".format(traced), flush=True)
    # opt_fn = compile_fx(traced, [x, zero_point, scale])

    opt_fn(x, zero_point, scale)

    real_out = fn(x, zero_point, scale)

    # exit(-1)

    compiled_out = opt_fn(x, zero_point, scale)
    print("real_out is: {}".format(real_out), flush=True)
    print("compiled_out is: {}".format(compiled_out), flush=True)
    tol = 0.0001
    print(torch.allclose(real_out, compiled_out, atol=tol, rtol=tol))
    assert torch.allclose(real_out, compiled_out, atol=tol, rtol=tol), "Fail to compare result of real_out and compiled_out"



def test2():
    def fn(x, scale, zero_point, use_dequant, use_quant, use_decomposed):
        # For quantized_decomposed.dequantize_per_tensor
        # Refer to torch/ao/quantization/fx/_decomposed.py
        if use_dequant:
            if use_decomposed:
                x = (x.to(torch.float32) - zero_point) * scale
            else:
                x = torch.ops.quantized_decomposed.dequantize_per_tensor(x, scale, zero_point, 0, 255, torch.uint8)

        x = torch.relu(x)

        # For quantized_decomposed.quantize_per_tensor
        # Refer to torch/ao/quantization/fx/_decomposed.py
        if use_quant:
            if use_decomposed:
                inv_scale = 1.0 / scale
                x = torch.clamp(torch.round(x * inv_scale) + zero_point, 0, 255).to(
                    torch.uint8
                )
            else:
                x = torch.ops.quantized_decomposed.quantize_per_tensor(x, scale, zero_point, 0, 255, torch.uint8)
        return x

    use_dequant_list = [False, True]
    use_quant_list = [False, True]
    use_decomposed_list = [False, True]


    import logging
    # torch._dynamo.config.log_level = logging.DEBUG
    torch._dynamo.config.verbose = True
    torch._inductor.config.trace.enabled = True
    torch._inductor.config.trace.debug_log = True
    torch._inductor.config.debug = True

    use_dequant_list = [True]
    use_quant_list = [True]
    use_decomposed_list = [False]
    import itertools
    for use_dequant, use_quant, use_decomposed in itertools.product(
        use_dequant_list, use_quant_list, use_decomposed_list
    ):
        x = torch.clamp(
            torch.randn((1, 7, 7, 9), dtype=torch.float32) * 100, 0, 255
        )
        if use_dequant:
            x = x.to(torch.uint8)
        zero_point = 100
        scale = 0.01
        if not use_decomposed:
            zero_point = torch.tensor(zero_point, dtype=torch.int32)
            scale = torch.tensor(scale)
        with config.patch({"cpp.simdlen": None}):
            torch._dynamo.reset()
            metrics.reset()
            opt_fn = torch._dynamo.optimize("inductor")(fn)
            opt_fn(x, scale, zero_point, use_dequant, use_quant, use_decomposed)

            real_out = fn(x, scale, zero_point, use_dequant, use_quant, use_decomposed)
            compiled_out = opt_fn(x, scale, zero_point, use_dequant, use_quant, use_decomposed)


if __name__ == "__main__":
    simdlen = None # Default, on this system is avx512 version
    simdlen = 1 # scalar version
    simdlen = 255 # scalar version
    simdlen = 256 # Unspported avx2 version
    #simdlen = 257 # scalar version
    #simdlen = 512 # avx512 version
    #simdlen = 513 # scalar version

    simdlens = [None, 1, 255, 256, 257, 512, 513]

    simdlens = [None]

    for simdlen in simdlens:
        with config.patch({"cpp.simdlen": simdlen}):
            torch._dynamo.reset()
            metrics.reset()
            # test1()
            test2()
            # print("simdlen is: {}".format(simdlen), flush=True)
            # if simdlen in [None, 256, 512]:
            #     assert metrics.generated_cpp_vec_kernel_count >= 1
            # print("metrics.generated_cpp_vec_kernel_count is: {}".format(metrics.generated_cpp_vec_kernel_count), flush=True)


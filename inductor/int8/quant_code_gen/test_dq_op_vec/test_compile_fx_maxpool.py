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

torch._dynamo.config.log_level = logging.DEBUG
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
            self.conv1 = torch.nn.Conv2d(3, 64, (7, 7), stride=(2, 2), padding=(3, 3))
            #self.conv2 = torch.nn.Conv2d(3, 16, (3, 3), stride=(1, 1), padding=(1, 1))
            self.relu = torch.nn.ReLU()
            self.maxpool = torch.nn.MaxPool2d((3, 3), stride=(2, 2), padding=(1, 1))
            self.conv2 = torch.nn.Conv2d(64, 64, (3, 3), stride=(1, 1), padding=(1, 1))
            self.relu2 = torch.nn.ReLU()

        def forward(self, x):
            return self.conv2(self.maxpool(self.conv1(x)))
            # return self.conv2(self.relu(self.conv1(x)))

            # print("x.size() is:{}".format(x.size()), flush=True)
            # t1 = self.conv1(x)
            # print("t1.size() is:{}".format(t1.size()), flush=True)
            # #t2 = self.maxpool(t1)
            # t2 = self.relu(t1)
            # print("t2.size() is:{}".format(t2.size()), flush=True)
            # t3 = self.conv2(t2)
            # print("t3.size() is:{}".format(t3.size()), flush=True)
            # return t3

        


            # return self.relu2(self.relu(self.conv1(x)))
            # return x + x

    m = M().to(memory_format=torch.channels_last).eval()
    # **TODO** Also test channel_last format
    example_inputs = (torch.randn(116, 3, 224, 224).contiguous(memory_format=torch.channels_last),)
    

    # m(*example_inputs)
    # return
    
    
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
    compiled_out = opt_fn(*example_inputs)

    print("---- Finish the testing ----", flush=True)




    # zero_point = 100
    # scale = 0.001

    # # zero_point = torch.tensor(100)
    # # scale = torch.tensor(0.001)

    # print("x is: {}".format(x), flush=True)

    # opt_fn = torch._dynamo.optimize("inductor")(fn)
    
    # # traced = make_fx(fn)(x, zero_point, scale)
    # # print("traced graph is: {}".format(traced), flush=True)
    # # opt_fn = compile_fx(traced, [x, zero_point, scale])

    # opt_fn(x, zero_point, scale)

    # real_out = fn(x, zero_point, scale)

    # # exit(-1)

    # compiled_out = opt_fn(x, zero_point, scale)
    # print("real_out is: {}".format(real_out), flush=True)
    # print("compiled_out is: {}".format(compiled_out), flush=True)
    # tol = 0.0001
    # print(torch.allclose(real_out, compiled_out, atol=tol, rtol=tol))
    # assert torch.allclose(real_out, compiled_out, atol=tol, rtol=tol), "Fail to compare result of real_out and compiled_out"


# maxpool = torch.nn.MaxPool2d(kernel_size, stride)
def test2():
    def fn(x, scale, zero_point, use_dequant, use_quant, use_decomposed):
        # For quantized_decomposed.dequantize_per_tensor
        # Refer to torch/ao/quantization/fx/_decomposed.py
        if use_dequant:
            if use_decomposed:
                x = (x.to(torch.float32) - zero_point) * scale
            else:
                x = torch.ops.quantized_decomposed.dequantize_per_tensor(x, scale, zero_point, 0, 255, torch.uint8)

        # x = torch.relu(x)
        max_pool2d_with_indices_default = torch.ops.aten.max_pool2d_with_indices.default(x, [3, 3], [2, 2], [1, 1])
        x = max_pool2d_with_indices_default[0]

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
    torch._dynamo.config.log_level = logging.DEBUG
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
            torch.randn((116, 64, 112, 112), dtype=torch.float32) * 100, 0, 255
        )
        if use_dequant:
            x = x.to(torch.uint8).to(memory_format=torch.channels_last)
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
            test1()
            # test2()
            
            # print("simdlen is: {}".format(simdlen), flush=True)
            # if simdlen in [None, 256, 512]:
            #     assert metrics.generated_cpp_vec_kernel_count >= 1
            # print("metrics.generated_cpp_vec_kernel_count is: {}".format(metrics.generated_cpp_vec_kernel_count), flush=True)


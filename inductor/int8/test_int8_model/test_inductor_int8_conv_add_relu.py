
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
from torch._inductor.compile_fx import compile_fx, compile_fx_quantization

def test_inductor_int8_conv_add_relu():
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
        def __init__(self, inplace_add=False, inplace_relu=False) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(
                # in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1
                in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv3 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
            self.relu = torch.nn.ReLU(inplace=inplace_relu)
            self.inplace_add = inplace_add

        def forward(self, x):
            if not self.inplace_add:
                x1 = self.conv(x)
                return self.relu(self.conv2(x1) + self.conv3(x1))
            else:
                x1 = self.conv(x)
                accum = self.conv2(x1)
                accum += self.conv3(x1)
                return self.relu(accum)
            #return self.conv2(x1)

    torch.backends.quantized.engine = "x86"
    # example_inputs = (torch.randn(1, 1, 224, 224),)
    example_inputs = (torch.randn(1, 3, 16, 16),)
    #m = Mod(inplace_add=True).eval()
    #for inplace_add in [True, False]:
    import itertools
    inplace_add_inplace_relu_optioins = itertools.product(
        [True, False],  # inplace add
        [True, False],  # inplace relu
    )

    for inplace_add, inplace_relu in inplace_add_inplace_relu_optioins:
        m = Mod(inplace_add=inplace_add, inplace_relu=inplace_relu).eval()

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

        # Draw Graph
        from torch.fx.passes.graph_drawer import FxGraphDrawer
        g = FxGraphDrawer(m, "resnet18")
        g.get_dot_graph().write_svg("prepare.svg")    

        print("check the result after prepare: {}".format(
            torch.allclose(before_fusion_result, after_prepare_result)),
            flush=True)

        #return
        #exit(0)

        m = convert_pt2e(m)
        after_quant_result = m(*example_inputs)
        print("model after convert_pt2e is: {}".format(m), flush=True)
        g = FxGraphDrawer(m, "resnet18")
        g.get_dot_graph().write_svg("convert.svg")  

        print("check the result after convert: {}".format(
            torch.allclose(before_fusion_result, after_quant_result, rtol=5e-02, atol=5e-02)),
            flush=True)

        # exit(-1)
        # A few ops in EXIR are not supported. Set nopython=False to make it work
        # run = torch._dynamo.optimize(compile_fx, nopython=False)(m)
        # run = compile_fx_quantization(m, example_inputs, pytree_unflatten = True)
        #for pytree_unflatten in [True, False]:
        for pytree_unflatten in [False]:
            copy_m = copy.deepcopy(m)
            run = compile_fx_quantization(copy_m, example_inputs, pytree_unflatten = pytree_unflatten)

            # first run
            print("start the first run", flush=True)
            inductor_result = run(*example_inputs)


            # second run
            print("start the second run", flush=True)
            # import pdb;pdb.set_trace()
            inductor_result = run(*example_inputs)

            # print("inductor second run result is: {}".format(inductor_result), flush=True)

            # np.testing.assert_array_almost_equal(res_ref.cpu().numpy(),
            # res_quantized.cpu().numpy(), decimal=2)
            print(type(inductor_result), flush=True)
            print(type(after_quant_result), flush=True)
            print(torch.allclose(inductor_result if pytree_unflatten else inductor_result[0], after_quant_result, rtol=1e-05, atol=1e-08), flush=True)
    print("Finish the test", flush=True)


if __name__ == "__main__":
    test_inductor_int8_conv_add_relu()

import torch
import torch._dynamo as torchdynamo
import torchvision.models as models
import copy
from torch._inductor.compile_fx import compile_fx

import logging
torch._dynamo.config.log_level = logging.DEBUG
torch._dynamo.config.verbose = True
torch._inductor.config.trace.enabled = True
torch._inductor.config.debug = True

class M(torch.nn.Module):
    def __init__(self, use_relu: bool = False, inplace_relu: bool = False):
        super().__init__()
        self.use_relu = use_relu
        self.conv1 = torch.nn.Conv2d(3, 6, (2, 2), stride=(1, 1), padding=(1, 1))
        self.relu = torch.nn.ReLU(inplace=inplace_relu)

    def forward(self, x):
        x = self.conv1(x)
        return self.relu(x) if self.use_relu else x

def test_fp32():
    model = models.__dict__["resnet50"](pretrained=True).eval()
    #tracing_mode = "real"
    tracing_mode = "symbolic"
    example_inputs = (torch.randn(116, 3, 224, 224).to(memory_format=torch.channels_last),)
    m, guards = torchdynamo.export(
        model,
        *copy.deepcopy(example_inputs),
        aten_graph=True,
        tracing_mode=tracing_mode,
    )
    print("model after dynamoexport is: {}".format(m), flush=True)
    with torch.no_grad():
        base_res = m(*example_inputs)
        base_res = m(*example_inputs)
        print("base_res.size() is: {}".format(base_res.size()), flush=True)
        input2 = torch.randn(200, 3, 224, 224).contiguous(memory_format=torch.channels_last)
        base_res2 = m(input2)
        print("base_res2.size() is: {}".format(base_res2.size()), flush=True)
        input3 = torch.randn(2, 3, 224, 224).contiguous(memory_format=torch.channels_last)
        base_res3 = m(input3)
        print("base_res3.size() is: {}".format(base_res3.size()), flush=True)

        run = compile_fx(m, example_inputs)

        inductor_res = run(*example_inputs)
        inductor_res = run(*example_inputs)
        print("inductor_res.size() is: {}".format(inductor_res.size()), flush=True)
        input2 = torch.randn(200, 3, 224, 224).contiguous(memory_format=torch.channels_last)
        inductor_res2 = run(input2)
        print("inductor_res2.size() is: {}".format(inductor_res2.size()), flush=True)
        input3 = torch.randn(2, 3, 224, 224).contiguous(memory_format=torch.channels_last)
        inductor_res3 = run(input3)
        print("inductor_res3.size() is: {}".format(inductor_res3.size()), flush=True)

def test_fp32_torch_compile():
    model = models.__dict__["resnet50"](pretrained=True).eval()
    example_inputs = (torch.randn(116, 3, 224, 224).to(memory_format=torch.channels_last),)
    with torch.no_grad():
        import pdb;pdb.set_trace()
        run = torch.compile(model, fullgraph=True, dynamic=True)
        inductor_res = run(*example_inputs)
        inductor_res = run(*example_inputs)
        print("inductor_res.size() is: {}".format(inductor_res.size()), flush=True)
        input2 = torch.randn(200, 3, 224, 224).contiguous(memory_format=torch.channels_last)
        inductor_res2 = run(input2)
        print("inductor_res2.size() is: {}".format(inductor_res2.size()), flush=True)
        input3 = torch.randn(2, 3, 224, 224).contiguous(memory_format=torch.channels_last)
        inductor_res3 = run(input3)
        print("inductor_res3.size() is: {}".format(inductor_res3.size()), flush=True)

def test_int8():
    from torch.ao.quantization._quantize_pt2e import prepare_pt2e, convert_pt2e
    from torch.ao.quantization.backend_config._inductor_pt2e import get_inductor_pt2e_backend_config
    from torch.ao.quantization import (
        get_default_qconfig,
        QConfigMapping,
    )
   
    # model = models.__dict__["resnet50"](pretrained=True).eval()
    model = M(False, False).eval()

    tracing_mode = "real"
    #tracing_mode = "symbolic"
    example_inputs = (torch.randn(116, 3, 224, 224).to(memory_format=torch.channels_last),)
    m, guards = torchdynamo.export(
        model,
        *copy.deepcopy(example_inputs),
        aten_graph=True,
        tracing_mode=tracing_mode,
    )

    # print("export_real_rn50 graph is:{}".format(m), flush=True)
    # from torch.fx.passes.graph_drawer import FxGraphDrawer
    # g = FxGraphDrawer(m, "resnet50")
    # g.get_dot_graph().write_svg("./export_real_rn50.svg")
   
    # # print("export_symbolic_rn50 graph is:{}".format(m), flush=True)
    # # from torch.fx.passes.graph_drawer import FxGraphDrawer
    # # g = FxGraphDrawer(m, "resnet50")
    # # g.get_dot_graph().write_svg("./export_symbolic_rn50.svg")
    # exit(-1)

    with torch.no_grad():
        base_res = m(*example_inputs)
        base_res = m(*example_inputs)
        print("base_res.size() is: {}".format(base_res.size()), flush=True)
        input2 = torch.randn(200, 3, 224, 224).contiguous(memory_format=torch.channels_last)
        base_res2 = m(input2)
        print("base_res2.size() is: {}".format(base_res2.size()), flush=True)
        input3 = torch.randn(2, 3, 224, 224).contiguous(memory_format=torch.channels_last)
        base_res3 = m(input3)
        print("base_res3.size() is: {}".format(base_res3.size()), flush=True)

        torch.backends.quantized.engine = "x86"
        backend_config = get_inductor_pt2e_backend_config()
        qconfig = get_default_qconfig("x86")
        qconfig_mapping = QConfigMapping().set_global(qconfig)
        prepared_model = prepare_pt2e(m, qconfig_mapping, example_inputs, backend_config=backend_config)

        # Must use dummy input to init, otherwise conv weight scale dim size is not initlized correctly
        prepared_model(*example_inputs)

        converted_model = convert_pt2e(prepared_model)

        # from torch.fx.passes.graph_drawer import FxGraphDrawer
        # g = FxGraphDrawer(prepared_model, "resnet50")
        # g.get_dot_graph().write_svg("/home/lesliefang/pytorch_1_7_1/torch_inductor/rn50_prepare.svg")

        from torch._inductor.compile_fx import compile_fx_quantization
        run = compile_fx_quantization(converted_model, example_inputs)
        
        # run = compile_fx(converted_model, example_inputs)

        inductor_res = run(*example_inputs)
        inductor_res = run(*example_inputs)
        print("inductor_res.size() is: {}".format(inductor_res[0].size()), flush=True)
        input2 = torch.randn(116, 3, 224, 224).contiguous(memory_format=torch.channels_last)
        inductor_res2 = run(input2)
        print("inductor_res2.size() is: {}".format(inductor_res2[0].size()), flush=True)
        input3 = torch.randn(116, 3, 224, 224).contiguous(memory_format=torch.channels_last)
        inductor_res3 = run(input3)
        print("inductor_res3.size() is: {}".format(inductor_res3[0].size()), flush=True)

def test_int8_torch_compile():
    from torch.ao.quantization._quantize_pt2e import prepare_pt2e, convert_pt2e
    from torch.ao.quantization.backend_config._inductor_pt2e import get_inductor_pt2e_backend_config
    from torch.ao.quantization import (
        get_default_qconfig,
        QConfigMapping,
    )
   
    # model = models.__dict__["resnet50"](pretrained=True).eval()
    model = M(False, False).eval()

    tracing_mode = "real"
    #tracing_mode = "symbolic"
    example_inputs = (torch.randn(116, 3, 224, 224).to(memory_format=torch.channels_last),)
    m, guards = torchdynamo.export(
        model,
        *copy.deepcopy(example_inputs),
        aten_graph=True,
        tracing_mode=tracing_mode,
    )

    # print("export_real_rn50 graph is:{}".format(m), flush=True)
    # from torch.fx.passes.graph_drawer import FxGraphDrawer
    # g = FxGraphDrawer(m, "resnet50")
    # g.get_dot_graph().write_svg("./export_real_rn50.svg")
   
    # # print("export_symbolic_rn50 graph is:{}".format(m), flush=True)
    # # from torch.fx.passes.graph_drawer import FxGraphDrawer
    # # g = FxGraphDrawer(m, "resnet50")
    # # g.get_dot_graph().write_svg("./export_symbolic_rn50.svg")
    # exit(-1)

    with torch.no_grad():
        base_res = m(*example_inputs)
        base_res = m(*example_inputs)
        print("base_res.size() is: {}".format(base_res.size()), flush=True)
        input2 = torch.randn(200, 3, 224, 224).contiguous(memory_format=torch.channels_last)
        base_res2 = m(input2)
        print("base_res2.size() is: {}".format(base_res2.size()), flush=True)
        input3 = torch.randn(2, 3, 224, 224).contiguous(memory_format=torch.channels_last)
        base_res3 = m(input3)
        print("base_res3.size() is: {}".format(base_res3.size()), flush=True)

        torch.backends.quantized.engine = "x86"
        backend_config = get_inductor_pt2e_backend_config()
        qconfig = get_default_qconfig("x86")
        qconfig_mapping = QConfigMapping().set_global(qconfig)
        prepared_model = prepare_pt2e(m, qconfig_mapping, example_inputs, backend_config=backend_config)

        # Must use dummy input to init, otherwise conv weight scale dim size is not initlized correctly
        prepared_model(*example_inputs)

        converted_model = convert_pt2e(prepared_model)

        # from torch.fx.passes.graph_drawer import FxGraphDrawer
        # g = FxGraphDrawer(prepared_model, "resnet50")
        # g.get_dot_graph().write_svg("/home/lesliefang/pytorch_1_7_1/torch_inductor/rn50_prepare.svg")

        # from torch._inductor.compile_fx import compile_fx_quantization
        # run = compile_fx_quantization(converted_model, example_inputs)
        
        run = compile_fx(converted_model, example_inputs)

        inductor_res = run(*example_inputs)
        inductor_res = run(*example_inputs)
        print("inductor_res.size() is: {}".format(inductor_res.size()), flush=True)
        input2 = torch.randn(116, 3, 224, 224).contiguous(memory_format=torch.channels_last)
        inductor_res2 = run(input2)
        print("inductor_res2.size() is: {}".format(inductor_res2.size()), flush=True)
        input3 = torch.randn(116, 3, 224, 224).contiguous(memory_format=torch.channels_last)
        inductor_res3 = run(input3)
        print("inductor_res3.size() is: {}".format(inductor_res3.size()), flush=True)

if __name__ == "__main__":
    # test_fp32()
    # test_fp32_torch_compile()
    
    # test_int8()
    test_int8_torch_compile()

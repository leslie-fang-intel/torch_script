import torch
import torchvision
import torch._dynamo as torchdynamo
import copy
from torch.ao.quantization._pt2e.quantizer import (
    OperatorConfig,
    QNNPackQuantizer,
    Quantizer,
    X86InductorQuantizer,
)
from torch.ao.quantization._quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e_quantizer,
    prepare_qat_pt2e_quantizer,
)
from utils_vis import make_dot, draw

def draw_graph(model,data,graph_name):
    model.eval()
    with torch.no_grad():
        traced_model = torch.jit.trace(model, data)
        traced_model = torch.jit.freeze(traced_model)
        y = traced_model(data)
        y = traced_model(data)

        graph = traced_model.graph_for(data)
        # print(graph)
        print(graph_name,traced_model.code)
        draw(graph).render(graph_name)

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)
        self.bn = torch.nn.BatchNorm2d(3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

def pytorch_pt2e_qat(model_fp, data):
    example_inputs = (data, )
    m, guards = torchdynamo.export(
        model_fp,
        *copy.deepcopy(example_inputs),
        aten_graph=True
    )

    before_fusion_result = m(*example_inputs)
    import torch.ao.quantization._pt2e.quantizer.x86_inductor_quantizer as xiq
    quantizer = X86InductorQuantizer()
    operator_config = xiq.get_default_x86_inductor_quantization_config(is_qat=True)
    quantizer.set_global(operator_config)

    # import torch.ao.quantization._pt2e.quantizer.qnnpack_quantizer as qq
    # quantizer = QNNPackQuantizer()
    # quantizer.set_global(qq.get_symmetric_quantization_config(is_per_channel=True, is_qat=True))

    # Insert Observer
    m = prepare_qat_pt2e_quantizer(m, quantizer)
    print("prepared model is: {}".format(m), flush=True)

    from torch.fx.passes.graph_drawer import FxGraphDrawer
    g = FxGraphDrawer(m, "resnet50")
    g.get_dot_graph().write_svg("./rn50_qat_pt2e_prepare.svg")

    after_prepare_result = m(*example_inputs)
    
    m = convert_pt2e(m)
    print("converted model is: {}".format(m), flush=True)

    m.eval()
    with torch.no_grad():
        traced_model = torch.jit.trace(m, data, check_trace=False)
        traced_model = torch.jit.freeze(traced_model)
        y = traced_model(data)
        y = traced_model(data)

        graph = traced_model.graph_for(data)
        print(graph, flush=True)
        # print(graph_name,traced_model.code)
        graph_name = "./pt2e_qat_rn50_jit.svg"
        draw(graph).render(graph_name)

if __name__ == "__main__":
    data = torch.randn(1, 3, 224, 224)
    model_fp = torchvision.models.resnet50(pretrained=True)
    # model_fp = M()

    print("--------------PyTorch 2.0 QAT -----------")
    pytorch_pt2e_qat(model_fp, data)   

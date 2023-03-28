import torch
import torch._dynamo as torchdynamo
from torch.ao.quantization import (
    get_default_qconfig,
    QConfigMapping,
)
from torch.ao.quantization._quantize_pt2e import prepare_pt2e, convert_pt2e

class Mod(torch.nn.Module):
    def __init__(self,) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.maxpool = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = torch.nn.ReLU()

        self.conv4 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        x1 = self.conv(x)
        res1 = self.relu(self.conv2(x1) + self.conv3(x1))
        res2 = self.relu2(self.conv4(res1) + res1)
        return res2

def test_issue_example():
    from torch.ao.quantization.backend_config import (
        BackendConfig,
        DTypeConfig,
        ObservationType,
        BackendPatternConfig,
    )
    from torch.ao.quantization.utils import MatchAllNode
    import itertools
    import copy
    weighted_op_quint8_dtype_config = DTypeConfig(
        input_dtype=torch.quint8,
        output_dtype=torch.quint8,
        weight_dtype=torch.qint8,
        bias_dtype=torch.float,
    )
    def get_conv_configs():
        conv_configs = []
        observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
        dtype_configs = [weighted_op_quint8_dtype_config]
        conv_configs.append(
            BackendPatternConfig(torch.ops.aten.convolution.default)
            .set_observation_type(observation_type)  # noqa: E131
            .set_dtype_configs(dtype_configs)
            ._set_input_type_to_index({"weight": 1, "bias": 2})
        )
        # Conv add ReLU case
        def _conv_add_relu_root_node_getter_left(pattern):
            relu, add_pattern = pattern
            _, conv, _ = add_pattern
            return conv
        def _conv_add_relu_extra_inputs_getter_left(pattern):
            """ get inputs pattern for extra inputs, inputs for root node
            are assumed to be copied over from root node to the fused node
            """
            relu, add_pattern = pattern
            _, conv, extra_input = add_pattern
            return [extra_input]
        
        conv_add_relu_optioins = itertools.product(
            [torch.ops.aten.add.Tensor, torch.ops.aten.add_.Tensor],  # add op
            [torch.ops.aten.relu.default, torch.ops.aten.relu_.default],  # relu op
        )
        for add_op, relu_op in conv_add_relu_optioins:
            conv_configs.append(
                BackendPatternConfig()
                    ._set_pattern_complex_format((relu_op, (add_op, torch.ops.aten.convolution.default, MatchAllNode)))  # noqa: E131
                    .set_observation_type(observation_type)
                    .set_dtype_configs(dtype_configs)
                    ._set_input_type_to_index({"weight": 1, "bias": 2})
                    ._set_root_node_getter(_conv_add_relu_root_node_getter_left)
                    ._set_extra_inputs_getter(_conv_add_relu_extra_inputs_getter_left)
            )
        return conv_configs

    def get_inductor_pt2e_backend_config():
        return (
            BackendConfig("inductor_pytorch_2.0_export")
            .set_backend_pattern_configs(get_conv_configs())
        )
    torch.backends.quantized.engine = "x86"
    example_inputs = (torch.randn(1, 3, 16, 16),)
    m = Mod().eval()

    m(*example_inputs)
    # return
    # import torchvision.models as models
    # m = models.__dict__["resnet50"](pretrained=True).eval()
    
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

    from torch.fx.passes.graph_drawer import FxGraphDrawer
    g = FxGraphDrawer(m, "conv_add_relu")
    g.get_dot_graph().write_svg("prepare_serials_conv_add_relu_graph.svg")
    # g = FxGraphDrawer(m, "resnet50")
    # g.get_dot_graph().write_svg("prepare_rn50_graph.svg")

def test():
    from torch.ao.quantization.backend_config import (
        BackendConfig,
        DTypeConfig,
        ObservationType,
        BackendPatternConfig,
    )
    from torch.ao.quantization.utils import MatchAllNode
    import itertools
    import copy
    weighted_op_quint8_dtype_config = DTypeConfig(
        input_dtype=torch.quint8,
        output_dtype=torch.quint8,
        weight_dtype=torch.qint8,
        bias_dtype=torch.float,
    )
    from torch.ao.quantization.backend_config._inductor_pt2e import get_inductor_pt2e_backend_config
    torch.backends.quantized.engine = "x86"
    example_inputs = (torch.randn(1, 3, 16, 16),)
    m = Mod().eval()

    # import torchvision.models as models
    # m = models.__dict__["resnet50"](pretrained=True).eval()
    
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

    from torch.fx.passes.graph_drawer import FxGraphDrawer
    g = FxGraphDrawer(m, "conv_add_relu")
    g.get_dot_graph().write_svg("prepare_conv_add_relu_graph.svg")
    # g = FxGraphDrawer(m, "resnet50")
    # g.get_dot_graph().write_svg("prepare_rn50_graph.svg")

if __name__ == "__main__":
    #test_issue_example()
    test()

import torch
import torch._dynamo as torchdynamo
import copy
import torchvision
from torch.ao.quantization import (
    get_default_qconfig,
    QConfigMapping,
)
from torch.ao.quantization.backend_config import (
    get_qnnpack_backend_config,
)
from torch.ao.quantization.backend_config._qnnpack_pt2e import get_qnnpack_pt2e_backend_config
from torch.ao.quantization.quantize_fx import prepare_fx, convert_to_reference_fx
from torch.ao.quantization._quantize_pt2e import prepare_pt2e, convert_pt2e
from torch.ao.ns.fx.utils import (
    compute_sqnr,
)

class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv = torch.nn.Conv2d(3, 64, (3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        return self.relu(self.conv(x))

def test():
    torch.backends.quantized.engine = "qnnpack"
    example_inputs = (torch.randn(1, 3, 224, 224),)
    
    # m = torchvision.models.resnet18().eval()
    m = SimpleNet().eval()

    m_copy = copy.deepcopy(m)

    print("New qnnpack_pt2e backend", flush=True)
    # program capture
    m, guards = torchdynamo.export(
        m,
        *copy.deepcopy(example_inputs),
        aten_graph=True,
        tracing_mode="real",
    )

    backend_config = get_qnnpack_pt2e_backend_config()
    # TODO: define qconfig_mapping specifically for executorch
    qconfig = get_default_qconfig("qnnpack")
    qconfig_mapping = QConfigMapping().set_global(qconfig)
    before_fusion_result = m(*example_inputs)

    m = prepare_pt2e(m, qconfig_mapping, example_inputs, backend_config)

    print("model after prepare is: {}".format(m), flush=True)
    # checking that we inserted observers correctly for maxpool operator (input and
    # output share observer instance)
    # self.assertEqual(id(m.activation_post_process_3), id(m.activation_post_process_2))
    after_prepare_result = m(*example_inputs)
    m = convert_pt2e(m)

    print("model after convert is: {}".format(m), flush=True)

    after_quant_result = m(*example_inputs)

    # Step2
    print("Original fx qnnpack backend", flush=True)

    # comparing with existing fx graph mode quantization reference flow
    backend_config = get_qnnpack_backend_config()
    m_fx = prepare_fx(m_copy, qconfig_mapping, example_inputs, backend_config=backend_config)
    after_prepare_result_fx = m_fx(*example_inputs)
    m_fx = convert_to_reference_fx(m_fx, backend_config=backend_config)

    # print("model after fx convert is: {}".format(m_fx), flush=True)
    after_quant_result_fx = m_fx(*example_inputs)

    # # the result matches exactly after prepare
    # self.assertEqual(after_prepare_result, after_prepare_result_fx)
    # self.assertEqual(compute_sqnr(after_prepare_result, after_prepare_result_fx), torch.tensor(float("inf")))
    # # there are slight differences after convert due to different implementations
    # # of quant/dequant
    # self.assertTrue(torch.max(after_quant_result - after_quant_result_fx) < 1e-1)
    # self.assertTrue(compute_sqnr(after_quant_result, after_quant_result_fx) > 35)

if __name__ == "__main__":
    test()
import torch
import torch.nn as nn
import torch._dynamo as torchdynamo
import copy
from torch.ao.quantization import (
    get_default_qconfig,
    observer,
    QConfigMapping,
    default_per_channel_symmetric_qnnpack_qconfig,
)
from torch.ao.quantization.backend_config._x86_inductor_pt2e import (
    get_x86_inductor_pt2e_backend_config,
)
from torch.ao.quantization._quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
)
from torch._inductor.compile_fx import compile_fx
import torch.ao.quantization._pt2e.quantizer.qnnpack_quantizer as qq
from torch.ao.quantization._pt2e.quantizer import (
    OperatorConfig,
    QNNPackQuantizer,
    Quantizer,
)
from torch.ao.quantization._quantize_pt2e import (
    convert_pt2e,
    _convert_to_reference_decomposed_fx,
    prepare_pt2e_quantizer,
    prepare_qat_pt2e_quantizer,
)

torch._dynamo.config.verbose = True
torch._inductor.config.trace.enabled = True
torch._inductor.config.debug = True


def test_rn50():
    import torchvision

    example_inputs = (torch.randn(1, 3, 224, 224),)
    m = torchvision.models.resnet50().eval()
    m_copy = copy.deepcopy(m)

    export_model, guards = torchdynamo.export(
        m,
        *copy.deepcopy(example_inputs),
        aten_graph=True
    )

    from torch.ao.quantization._pt2e.quantizer import X86InductorQuantizer
    import torch.ao.quantization._pt2e.quantizer.x86_inductor_quantizer as xiq
    quantizer = X86InductorQuantizer()
    operator_config = xiq.get_default_x86_inductor_quantization_config()
    quantizer.set_global(operator_config)

    with torch.no_grad():
        prepare_model = prepare_pt2e_quantizer(export_model, quantizer)
        print("prepared model is: {}".format(prepare_model), flush=True)
        prepare_model(*example_inputs)

        convert_model = convert_pt2e(prepare_model)
        print("converted model is: {}".format(convert_model), flush=True)

        convert_model.eval()

        compiler_model = torch.compile(convert_model)

        print("start the first run", flush=True)
        compiler_model(*example_inputs)

        print("start the second run", flush=True)
        compiler_model(*example_inputs)



if __name__ == "__main__":
    test_rn50()

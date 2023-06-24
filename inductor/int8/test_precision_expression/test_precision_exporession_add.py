    
import torch
from torch.ao.quantization._pt2e.quantizer import (
    ComposableQuantizer,
    DerivedQuantizationSpec,
    EmbeddingQuantizer,
    FixedQParamsQuantizationSpec,
    OperatorConfig,
    QNNPackQuantizer,
    QuantizationAnnotation,
    QuantizationSpec,
    Quantizer,
    SharedQuantizationSpec,
)
import copy
import torch._dynamo as torchdynamo
from torch.ao.quantization._quantize_pt2e import (
    _convert_to_reference_decomposed_fx,
    convert_pt2e,
    prepare_pt2e_quantizer,
    prepare_qat_pt2e_quantizer,
)

torch._dynamo.config.verbose = True
torch._inductor.config.trace.enabled = True
torch._inductor.config.trace.debug_log = True
torch._inductor.config.debug = True
torch._inductor.config.verbose_progress = True
from torch._inductor.compile_fx import compile_fx

import numpy as np
import random

local_seed = 2023
torch.manual_seed(local_seed) # Set PyTorch seed
np.random.seed(seed=local_seed) # Set Numpy seed
random.seed(local_seed) # Set the Python seed

def test():
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return x + y

    import torch.ao.quantization._pt2e.quantizer.qnnpack_quantizer as qq

    quantizer = QNNPackQuantizer()
    operator_config = qq.get_symmetric_quantization_config(is_per_channel=True)
    quantizer.set_global(operator_config)
    m_eager = M().eval()

    example_inputs = (torch.randn(1, 3, 32), torch.randn(1, 3, 32),)
    # program capture
    m = m_eager
    m, guards = torchdynamo.export(
        m,
        *copy.deepcopy(example_inputs),
        aten_graph=True,
    )

    m = prepare_pt2e_quantizer(m, quantizer)
    # Calibrate
    m(*example_inputs)
    m_ref = copy.deepcopy(m)
    m_ref = convert_pt2e(m_ref, use_reference_representation=False)
    m = convert_pt2e(m, use_reference_representation=True)
    print("m_ref:", m_ref)
    print("m:", m)

    # optimized_model = compile_fx(m_ref, example_inputs)
    # print("first run", flush=True)
    # optimized_model(*example_inputs)
    # print("second run", flush=True)
    # res = optimized_model(*example_inputs)

    # pt2_quant_output_ref = m_ref(*example_inputs)

    # print("res is: {}".format(res), flush=True)
    # print("pt2_quant_output_ref is: {}".format(pt2_quant_output_ref), flush=True)
    # print(torch.allclose(res, pt2_quant_output_ref[0], rtol=0.01, atol=0.01), flush=True)



    optimized_model = compile_fx(m, example_inputs)
    print("first run", flush=True)
    optimized_model(*example_inputs)
    print("second run", flush=True)
    res = optimized_model(*example_inputs)

    pt2_quant_output = m(*example_inputs)

    print("res is: {}".format(res), flush=True)
    print("pt2_quant_output is: {}".format(pt2_quant_output), flush=True)
    print(torch.allclose(res, pt2_quant_output[0], rtol=0.01, atol=0.01), flush=True)
        

if __name__ == "__main__":
    test()

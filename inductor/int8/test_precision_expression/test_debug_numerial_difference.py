    
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
import torch.utils._pytree as pytree
import torch.fx._pytree as fx_pytree
from torch.utils._pytree import (
    tree_flatten,
    tree_map,
    tree_unflatten,
    TreeSpec,
    LeafSpec,
    pytree_to_str,
    str_to_pytree,
)

local_seed = 2023
torch.manual_seed(local_seed) # Set PyTorch seed
np.random.seed(seed=local_seed) # Set Numpy seed
random.seed(local_seed) # Set the Python seed

def test():
    class M(torch.nn.Module):
        def __init__(self,):
            super().__init__()

        def forward(self, x, y):
            # arg0, arg1, = fx_pytree.tree_flatten_spec(([x, y], {}), self._in_spec)
            arg0, arg1, = x, y
            quantize_per_tensor_default = torch.ops.quantized_decomposed.quantize_per_tensor(arg0, 0.02003643848001957, 130, 0, 255, torch.uint8);  arg0 = None
            quantize_per_tensor_default_1 = torch.ops.quantized_decomposed.quantize_per_tensor(arg1, 0.021639926359057426, 138, 0, 255, torch.uint8);  arg1 = None
            _to_copy_default = torch.ops.aten._to_copy.default(quantize_per_tensor_default, dtype = torch.int32);  quantize_per_tensor_default = None
            _to_copy_default_1 = torch.ops.aten._to_copy.default(quantize_per_tensor_default_1, dtype = torch.int32);  quantize_per_tensor_default_1 = None
            div_tensor = torch.ops.aten.div.Tensor(0.02003643848001957, 0.03313786908984184)
            sub_tensor = torch.ops.aten.sub.Tensor(_to_copy_default, 130);  _to_copy_default = None
            mul_tensor = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor);  div_tensor = sub_tensor = None
            _to_copy_default_2 = torch.ops.aten._to_copy.default(mul_tensor, dtype = torch.int32);  mul_tensor = None
            div_tensor_1 = torch.ops.aten.div.Tensor(0.021639926359057426, 0.03313786908984184)
            sub_tensor_1 = torch.ops.aten.sub.Tensor(_to_copy_default_1, 138);  _to_copy_default_1 = None
            mul_tensor_1 = torch.ops.aten.mul.Tensor(div_tensor_1, sub_tensor_1);  div_tensor_1 = sub_tensor_1 = None
            _to_copy_default_3 = torch.ops.aten._to_copy.default(mul_tensor_1, dtype = torch.int32);  mul_tensor_1 = None
            add_tensor_1 = torch.ops.aten.add.Tensor(_to_copy_default_2, _to_copy_default_3);  _to_copy_default_2 = _to_copy_default_3 = None
            add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, 133);  add_tensor_1 = None
            clamp_default = torch.ops.aten.clamp.default(add_tensor_2, 0, 255);  add_tensor_2 = None
            _to_copy_default_4 = torch.ops.aten._to_copy.default(clamp_default, dtype = torch.uint8);  clamp_default = None
            dequantize_per_tensor_default_2 = torch.ops.quantized_decomposed.dequantize_per_tensor(_to_copy_default_4, 0.03313786908984184, 133, 0, 255, torch.uint8);  _to_copy_default_4 = None
            # return pytree.tree_unflatten([dequantize_per_tensor_default_2], self._out_spec)
            return (dequantize_per_tensor_default_2, )

    m = M()
    example_inputs = (torch.randn(1, 3, 32), torch.randn(1, 3, 32),)


    # Calculate reference result
    pt2_quant_output = m(*example_inputs)

    optimized_model = compile_fx(m, example_inputs)
    print("first run", flush=True)
    optimized_model(*example_inputs)
    print("second run", flush=True)
    res = optimized_model(*example_inputs)

    print("res is: {}".format(res), flush=True)
    print("pt2_quant_output is: {}".format(pt2_quant_output), flush=True)
    print(torch.allclose(res[0], pt2_quant_output[0], rtol=0.01, atol=0.01), flush=True)
        

def test_torch_to_int():
    input = torch.tensor([1.27, 1.68, -1.27, -1.68], dtype=torch.float)
    print("input is: {}".format(input), flush=True)
    input_int = input.to(dtype=torch.int32)
    print("input_int is: {}".format(input_int), flush=True)


if __name__ == "__main__":
    test()
    # test_torch_to_int()

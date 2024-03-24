import torch
from torch.library import Library, impl
import torch.ao.quantization.fx._decomposed
import numpy as np
import random
import copy

local_seed = 2024

torch.manual_seed(local_seed) # Set PyTorch seed
np.random.seed(seed=local_seed) # Set Numpy seed
random.seed(local_seed) # Set the Python seed

def qdq(input, scales, zero_points, axis, quant_min, quant_max, dtype, label):
    res = torch.ops.quantized_decomposed.test_backward(
        input, scales, zero_points, axis, quant_min, quant_max, dtype
    )
    return res

def test_eager(input, scales, zero_points, axis, quant_min, quant_max, dtype, label):
    print("--- start test_eager----", flush=True)
    res = qdq(input, scales, zero_points, axis, quant_min, quant_max, dtype, label)
    res.sum().backward()
    print("input_grad is: {}".format(input.grad), flush=True)

def test_compile(input, scales, zero_points, axis, quant_min, quant_max, dtype, label):
    print("--- start test_compile----", flush=True)
    compiled_qdq = torch.compile(qdq)
    print("---- start the forward run ----", flush=True)
    res = compiled_qdq(input, scales, zero_points, axis, quant_min, quant_max, dtype, label)
    print("---- start the backward run ----", flush=True)
    res.sum().backward()
    print("input_grad is: {}".format(input.grad), flush=True)

if __name__ == "__main__":

    input = torch.randn(1, 3, 224, 224, requires_grad=True)
    scales = torch.ones((1, 3, 224, 224,))
    zero_points = torch.zeros((1, 3, 224, 224,))
    axis = 1
    quant_min = -128
    quant_max = 127
    dtype = torch.int8
    label = torch.randn(1, 3, 224, 224)

    cinput = copy.deepcopy(input)
    cscales = copy.deepcopy(scales)
    czero_points = copy.deepcopy(zero_points)
    caxis = copy.deepcopy(axis)
    cquant_min = copy.deepcopy(quant_min)
    cquant_max = copy.deepcopy(quant_max)
    cdtype = copy.deepcopy(dtype)
    clabel = copy.deepcopy(label)

    test_eager(input, scales, zero_points, axis, quant_min, quant_max, dtype, label)  # Success
    test_compile(cinput, cscales, czero_points, caxis, cquant_min, cquant_max, cdtype, clabel)  # Failed
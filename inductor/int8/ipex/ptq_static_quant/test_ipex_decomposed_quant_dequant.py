import torch
import torch.ao.quantization.fx._decomposed
import numpy as np
import random

local_seed =2023
torch.manual_seed(local_seed) # Set PyTorch seed
np.random.seed(seed=local_seed) # Set Numpy seed
random.seed(local_seed) # Set the Python seed

def test_correctness_quant():
    input_tensor = torch.randn(1, 3, 2, 2)
    scale = 0.25
    zp = 0
    dtype = torch.uint8

    # import pdb;pdb.set_trace()
    q_tensor = torch.quantize_per_tensor(input_tensor, scale, zp, torch.quint8)
    decomposed_q_tensor = torch.ops.quantized_decomposed.quantize_per_tensor(input_tensor, scale, zp, 0, 255, torch.uint8)

    # torch._make_per_tensor_quantized_tensor(q_tensor.int_repr(), 1.2, 1)
    # print(type(q_tensor), flush=True)
    # print(type(decomposed_q_tensor), flush=True)
    # print(q_tensor, flush=True)
    print(q_tensor.int_repr(), flush=True)
    print(decomposed_q_tensor, flush=True)

def test_correctness_quant_ipex():
    import intel_extension_for_pytorch as ipex
    input_tensor = torch.randn(1, 3, 2, 2)
    scale = 0.25
    zp = 0
    dtype = torch.uint8

    # import pdb;pdb.set_trace()
    q_tensor = torch.quantize_per_tensor(input_tensor, scale, zp, torch.quint8)
    decomposed_q_tensor = torch.ops.quantized_decomposed.quantize_per_tensor(input_tensor, scale, zp, 0, 255, torch.uint8)

    # torch._make_per_tensor_quantized_tensor(q_tensor.int_repr(), 1.2, 1)
    # print(type(q_tensor), flush=True)
    # print(type(decomposed_q_tensor), flush=True)
    # print(q_tensor, flush=True)
    print(q_tensor.int_repr(), flush=True)
    print(decomposed_q_tensor, flush=True)
    print("torch.equal(tensorA, tensorB) is: {}".format(torch.equal(q_tensor.int_repr(), decomposed_q_tensor)), flush=True)

if __name__ == "__main__":
    # test_correctness_quant()
    test_correctness_quant_ipex()
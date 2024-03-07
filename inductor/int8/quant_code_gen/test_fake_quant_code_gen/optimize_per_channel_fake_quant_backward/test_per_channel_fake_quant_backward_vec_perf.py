# Test: using iomp and jemalloc
# CMD: clear && numactl -C 0-3 -m 0 python test_per_channel_fake_quant_backward_vec_perf.py
import torch
import torch.ao.quantization.fx._decomposed
import copy
import numpy as np
import random

local_seed = 2024

torch.manual_seed(local_seed) # Set PyTorch seed
np.random.seed(seed=local_seed) # Set Numpy seed
random.seed(local_seed) # Set the Python seed

def fq(input, scales, zero_points, axis, quant_min, quant_max):
    res = torch.fake_quantize_per_channel_affine(
        input, scales, zero_points, axis, quant_min, quant_max
    )
    return res

def qdq(input, scales, zero_points, axis, quant_min, quant_max):
    res = torch.ops.quantized_decomposed.fake_quant_per_channel(
        input, scales, zero_points, axis, quant_min, quant_max
    )
    return res

compiled_qdq = torch.compile(qdq)

def test_eager_aten_fake_quant(
    input, scales, zero_points, axis, quant_min, quant_max
):
    input.grad = None
    res = fq(input, scales, zero_points, axis, quant_min, quant_max)
    res.sum().backward()
    return res, input.grad

def test_eager_decomposed_fake_quant(
    input, scales, zero_points, axis, quant_min, quant_max
):
    input.grad = None
    res = qdq(input, scales, zero_points, axis, quant_min, quant_max)
    res.sum().backward()
    return res, input.grad

def test_compile_decomposed_fake_quant(
    input, scales, zero_points, axis, quant_min, quant_max
):
    input.grad = None
    res = compiled_qdq(input, scales, zero_points, axis, quant_min, quant_max)
    res.sum().backward()
    return res, input.grad

input = torch.randn(2, 3, 224, 224)
input[1, 2, 3, 4] = 257
input.requires_grad_()
scales = torch.ones((3,))
zero_points = torch.zeros((3,))
axis = 1
quant_min = -128
quant_max = 127

aten_input = copy.deepcopy(input)
compiler_input = copy.deepcopy(input)


# with torch.no_grad():
import time
warmup_step = 500
test_step = 2000

print("fake quant:")
# %timeit fq(input, scales, zero_points, axis, quant_min, quant_max, dtype)
for _ in range(warmup_step):
    test_eager_aten_fake_quant(
        aten_input, scales, zero_points, axis, quant_min, quant_max
    )
fq_start_time = time.time()
for _ in range(test_step):
    test_eager_aten_fake_quant(
        aten_input, scales, zero_points, axis, quant_min, quant_max
    )
fq_end_time = time.time()
res_aten_eager, input_grad_aten_eager = test_eager_aten_fake_quant(
    aten_input, scales, zero_points, axis, quant_min, quant_max
)

print("qdq:")
# %timeit qdq(input, scales, zero_points, axis, quant_min, quant_max, dtype)
for _ in range(warmup_step):
    test_eager_decomposed_fake_quant(
        input, scales, zero_points, axis, quant_min, quant_max
    )
qdq_start_time = time.time()
for _ in range(test_step):
    test_eager_decomposed_fake_quant(
        input, scales, zero_points, axis, quant_min, quant_max
    )
qdq_end_time = time.time()
res_decomp_eager, input_grad_decomp_eager = test_eager_decomposed_fake_quant(
    input, scales, zero_points, axis, quant_min, quant_max
)

print("compiled qdq:")
compiled_qdq = torch.compile(qdq)
# warmup
for _ in range(warmup_step):
    test_compile_decomposed_fake_quant(
        compiler_input, scales, zero_points, axis, quant_min, quant_max
    )
# %timeit compiled_qdq(input, scales, zero_points, axis, quant_min, quant_max, dtype)
compiled_qdq_start_time = time.time()
for _ in range(test_step):
    test_compile_decomposed_fake_quant(
        compiler_input, scales, zero_points, axis, quant_min, quant_max
    )
compiled_qdq_end_time = time.time()
res, input_grad = test_compile_decomposed_fake_quant(
    compiler_input, scales, zero_points, axis, quant_min, quant_max
)

# Test result tested on CPX
print("ATen fq time is: {}".format(fq_end_time-fq_start_time), flush=True)  # almost 1.6118338108062744
print("decomposed qdq time is: {}".format(qdq_end_time-qdq_start_time), flush=True)  # almost 1.2562391757965088
print("compiled decomposed qdq time is: {}".format(compiled_qdq_end_time-compiled_qdq_start_time), flush=True) # almost 1.3850011825561523

print("torch.allclose(res_aten_eager, res) is: {}".format(torch.allclose(res_aten_eager, res)), flush=True)
print("torch.allclose(input_grad_aten_eager, input_grad) is: {}".format(torch.allclose(input_grad_aten_eager, input_grad)), flush=True)
print("torch.allclose(res_decomp_eager, res) is: {}".format(torch.allclose(res_decomp_eager, res)), flush=True)
print("torch.allclose(input_grad_decomp_eager, input_grad) is: {}".format(torch.allclose(input_grad_decomp_eager, input_grad)), flush=True)
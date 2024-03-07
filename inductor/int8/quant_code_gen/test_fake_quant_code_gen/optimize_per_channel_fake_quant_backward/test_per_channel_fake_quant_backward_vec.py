# Test: using iomp and jemalloc
# CMD: clear && numactl -C 0-3 -m 0 python test_per_channel_fake_quant_vec.py
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
    compiled_qdq = torch.compile(qdq)
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

res_aten_eager, input_grad_aten_eager = test_eager_aten_fake_quant(
    aten_input, scales, zero_points, axis, quant_min, quant_max
)
res_decomp_eager, input_grad_decomp_eager = test_eager_decomposed_fake_quant(
    input, scales, zero_points, axis, quant_min, quant_max
)
res, input_grad = test_compile_decomposed_fake_quant(
    compiler_input, scales, zero_points, axis, quant_min, quant_max
)
print("torch.allclose(res_aten_eager, res) is: {}".format(torch.allclose(res_aten_eager, res)), flush=True)
print("torch.allclose(input_grad_aten_eager, input_grad) is: {}".format(torch.allclose(input_grad_aten_eager, input_grad)), flush=True)
print("torch.allclose(res_decomp_eager, res) is: {}".format(torch.allclose(res_decomp_eager, res)), flush=True)
print("torch.allclose(input_grad_decomp_eager, input_grad) is: {}".format(torch.allclose(input_grad_decomp_eager, input_grad)), flush=True)
exit(-1)

# # res = fq(input, scales, zero_points, axis, quant_min, quant_max, dtype)
# # loss_fn = torch.nn.MSELoss()
# # loss_fn(res, torch.randn(1, 3, 224, 224)).backward()
# # exit(-1)


# with torch.no_grad():
import time
warmup_step = 5
test_step = 20

print("qdq:")
# %timeit qdq(input, scales, zero_points, axis, quant_min, quant_max, dtype)
for _ in range(warmup_step):
    qdq(input, scales, zero_points, axis, quant_min, quant_max, dtype)
qdq_start_time = time.time()
for _ in range(test_step):
    qdq(input, scales, zero_points, axis, quant_min, quant_max, dtype)
qdq_end_time = time.time()
ref_res, ref_input_grad = qdq(input, scales, zero_points, axis, quant_min, quant_max, dtype)

print("fake quant:")
# %timeit fq(input, scales, zero_points, axis, quant_min, quant_max, dtype)
for _ in range(warmup_step):
    fq(input, scales, zero_points, axis, quant_min, quant_max, dtype)
fq_start_time = time.time()
for _ in range(test_step):
    fq(input, scales, zero_points, axis, quant_min, quant_max, dtype)
fq_end_time = time.time()
ref_res2, ref_input_grad2 = fq(input, scales, zero_points, axis, quant_min, quant_max, dtype)

print("compiled qdq:")
compiled_qdq = torch.compile(qdq)
# warmup
for _ in range(warmup_step):
    compiled_qdq(input, scales, zero_points, axis, quant_min, quant_max, dtype)
# %timeit compiled_qdq(input, scales, zero_points, axis, quant_min, quant_max, dtype)
compiled_qdq_start_time = time.time()
for _ in range(test_step):
    compiled_qdq(input, scales, zero_points, axis, quant_min, quant_max, dtype)
compiled_qdq_end_time = time.time()
res, input_grad = compiled_qdq(input, scales, zero_points, axis, quant_min, quant_max, dtype)

print("dqd time is: {}".format(qdq_end_time-qdq_start_time), flush=True)  # almost 1.2505955696105957
print("fq time is: {}".format(fq_end_time-fq_start_time), flush=True)  # almost 0.6014804840087891
print("compiled qdqd time is: {}".format(compiled_qdq_end_time-compiled_qdq_start_time), flush=True) # almost 0.1002035140991211

print(torch.allclose(ref_res, res), flush=True)
print(torch.allclose(ref_res2, res), flush=True)
print(torch.allclose(ref_input_grad, input_grad), flush=True)
print(torch.allclose(ref_input_grad2, input_grad), flush=True)
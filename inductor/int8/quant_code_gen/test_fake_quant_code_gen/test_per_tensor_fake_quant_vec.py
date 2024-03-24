# CMD: clear && numactl -C 0-27 -m 0 python test_per_channel_fake_quant_vec.py
import torch
import torch.ao.quantization.fx._decomposed

def qdq(input, scales, zero_points, quant_min, quant_max, dtype):
    input = torch.ops.quantized_decomposed.quantize_per_tensor(input, scales, zero_points, quant_min, quant_max, dtype)
    input = torch.ops.quantized_decomposed.dequantize_per_tensor(input, scales, zero_points, quant_min, quant_max, dtype)
    return input

def fq(input, scales, zero_points, quant_min, quant_max, dtype):
    input = torch.fake_quantize_per_tensor_affine(input, scales, zero_points, quant_min, quant_max)
    return input

device = torch.device("cpu")
input = torch.randn(1, 3, 224, 224).to(device=device)
# scales = torch.ones((3,)).to(device=device)
# zero_points = torch.zeros((3,)).to(device=device)
scales = 3.0
zero_points = 3

axis = 1

# Int8 Data type
quant_min = -128
quant_max = 127
dtype = torch.int8

## Uint8 data type
# quant_min = 0
# quant_max = 255
# dtype = torch.uint8

print("Input: ", input)
print("Scales: ", scales)
print("Zero points: ", zero_points)
print("Axis: ", axis)
print("Quant min: ", quant_min)
print("Quant max: ", quant_max)

import time
warmup_step = 10
test_step = 100

print("qdq:")
# %timeit qdq(input, scales, zero_points, axis, quant_min, quant_max, dtype)
for _ in range(warmup_step):
    qdq(input, scales, zero_points, quant_min, quant_max, dtype)
qdq_start_time = time.time()
for _ in range(test_step):
    qdq(input, scales, zero_points, quant_min, quant_max, dtype)
qdq_end_time = time.time()


print("fake quant:")
# %timeit fq(input, scales, zero_points, axis, quant_min, quant_max, dtype)
for _ in range(warmup_step):
    fq(input, scales, zero_points, quant_min, quant_max, dtype)
fq_start_time = time.time()
for _ in range(test_step):
    fq(input, scales, zero_points, quant_min, quant_max, dtype)
fq_end_time = time.time()


print("compiled qdq:")
compiled_qdq = torch.compile(qdq)
# warmup
for _ in range(warmup_step):
    compiled_qdq(input, scales, zero_points, quant_min, quant_max, dtype)
# %timeit compiled_qdq(input, scales, zero_points, axis, quant_min, quant_max, dtype)
compiled_qdq_start_time = time.time()
for _ in range(test_step):
    compiled_qdq(input, scales, zero_points, quant_min, quant_max, dtype)
compiled_qdq_end_time = time.time()

print("dqd time is: {}".format(qdq_end_time-qdq_start_time), flush=True)
print("fq time is: {}".format(fq_end_time-fq_start_time), flush=True)
print("compiled qdqd time is: {}".format(compiled_qdq_end_time-compiled_qdq_start_time), flush=True)

import torch
import torch.ao.quantization.fx._decomposed
from torch.ao.quantization.fx._decomposed import quantize_per_tensor, dequantize_per_tensor

def fn(
    x,
    dtype,
):
    # x = torch.ops.quantized_decomposed.quantize_per_tensor(
    #     x, scale, zero_point, quant_min, quant_max, dtype
    # )

    return x.to(dtype)

if __name__ == "__main__":
    dtype = torch.float8_e4m3fn
    # dtype = torch.uint8

    x = torch.randn(34)
    scale = 0.25
    zero_point = 0
    if dtype == torch.float8_e4m3fn:
        quant_min = int(torch.finfo(dtype).min)
        quant_max = int(torch.finfo(dtype).max)
    else:
        quant_min = 0
        quant_max = 255

    x_fp8 = torch.ops.quantized_decomposed.quantize_per_tensor(
        x, scale, zero_point, quant_min, quant_max, dtype
    )
    x_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(
        x_fp8,
        scale,
        zero_point,
        quant_min,
        quant_max,
        dtype,
        out_dtype=torch.float32,
    )

    x_fp32 = torch.tensor([0.5*(2 ** -6), 0.5*(2 ** -9), 3.1, 452.0, 489.0, 0.0], dtype=torch.float32)
    x_fp32 = x_fp32.repeat(5)

    x_fp8 = torch.tensor([0.127*(2 ** -7), 0.5*(2 ** -9), 3.1, 452.0, 489.0, 0.0], dtype=torch.float8_e4m3fn)
    x_fp8 = x_fp8.repeat(6)

    # x_fp8 = torch.tensor([0.5*(2 ** -6)], dtype=torch.float8_e4m3fn)

    test_to = False

    if test_to:
        ref_res = fn(x_fp32, torch.float8_e4m3fn)
        res = torch.compile(fn)(x_fp32, torch.float8_e4m3fn)
    else:
        ref_res = fn(x_fp8, torch.float32)
        res = torch.compile(fn)(x_fp8, torch.float32)
    print("ref_res: ", ref_res, flush=True)
    print("res: ", res, flush=True)
    torch.testing.assert_allclose(res.to(torch.float32), ref_res.to(torch.float32), rtol=1e-5, atol=1e-5)

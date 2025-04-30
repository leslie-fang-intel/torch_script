import torch
import torch.ao.quantization.fx._decomposed
from torch.ao.quantization.fx._decomposed import quantize_per_tensor, dequantize_per_tensor

def fn(
    x,
    scale,
    zero_point,
    quant_min,
    quant_max,
    dtype,
):
    # if use_dequant:
    #     x = torch.ops.quantized_decomposed.dequantize_per_tensor(
    #         x,
    #         scale,
    #         zero_point,
    #         quant_min,
    #         quant_max,
    #         dtype,
    #         out_dtype=dequant_out_dtype,
    #     )
    # x = torch.relu(x)
    # if use_quant:
    x = torch.ops.quantized_decomposed.quantize_per_tensor(
        x, scale, zero_point, quant_min, quant_max, dtype
    )
    return x

def fn2(
    x,
    scale,
    zero_point,
    quant_min,
    quant_max,
    dtype,
):
    x = torch.ops.quantized_decomposed.dequantize_per_tensor(
        x,
        scale,
        zero_point,
        quant_min,
        quant_max,
        dtype,
        out_dtype=torch.float32,
    )
    return x

if __name__ == "__main__":
    dtype = torch.float8_e4m3fn
    # dtype = torch.uint8

    x = torch.randn(256)
    scale = 0.25
    zero_point = 0
    if dtype == torch.float8_e4m3fn:
        quant_min = int(torch.finfo(dtype).min)
        quant_max = int(torch.finfo(dtype).max)
    else:
        quant_min = 0
        quant_max = 255
    ref_res = fn(x, scale, zero_point, quant_min, quant_max, dtype)
    res = torch.compile(fn)(x, scale, zero_point, quant_min, quant_max, dtype)
    print("ref_res: ", ref_res, flush=True)
    print("res: ", res, flush=True)
    torch.testing.assert_allclose(res.to(torch.float32), ref_res.to(torch.float32), rtol=1e-5, atol=1e-5)
    ref_res = fn2(ref_res, scale, zero_point, quant_min, quant_max, dtype)
    res = torch.compile(fn2)(res, scale, zero_point, quant_min, quant_max, dtype)
    print("x: ", x, flush=True)
    print("ref_res: ", ref_res, flush=True)
    print("res: ", res, flush=True)
    torch.testing.assert_allclose(res.to(torch.float32), ref_res.to(torch.float32), rtol=1e-5, atol=1e-5)

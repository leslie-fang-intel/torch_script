import torch
import torch.ao.quantization.fx._decomposed
from torch.ao.quantization.fx._decomposed import quantize_per_tensor, dequantize_per_tensor

def test_abs(fp32_x, fp32_y, quantized_x, quantized_y):
    ref_res = torch.abs(fp32_x).to(torch.float8_e4m3fn)
    res = torch.abs(quantized_x)
    print("ref_res: ", ref_res, flush=True)
    print("res: ", res, flush=True)
    torch.testing.assert_allclose(res.to(torch.float32), ref_res.to(torch.float32), rtol=1e-5, atol=1e-5)

def test_mul(fp32_x, fp32_y, quantized_x, quantized_y):
    ref_res = torch.mul(fp32_x, fp32_y).to(torch.float8_e4m3fn)
    print("start the fp8 run", flush=True)
    res = torch.mul(quantized_x, quantized_y)
    print("ref_res: ", ref_res, flush=True)
    print("res: ", res, flush=True)
    torch.testing.assert_allclose(res.to(torch.float32), ref_res.to(torch.float32), rtol=1e-5, atol=1e-5)

def test_eq(fp32_x, fp32_y, quantized_x, quantized_y):
    ref_res = torch.eq(fp32_x, fp32_y).to(torch.float8_e4m3fn)
    print("start the fp8 run", flush=True)
    res = torch.eq(quantized_x, quantized_y).to(torch.float8_e4m3fn)
    print("ref_res: ", ref_res, flush=True)
    print("res: ", res, flush=True)
    torch.testing.assert_allclose(res.to(torch.float32), ref_res.to(torch.float32), rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    dtype = torch.float8_e4m3fn
    x = torch.randn(512)
    y = torch.randn(512)

    # print("x is: {}".format(x), flush=True)
    # print("y is: {}".format(y), flush=True)

    scale = 0.25
    zero_point = 0
    quant_min = int(torch.finfo(dtype).min)
    quant_max = int(torch.finfo(dtype).max)

    # Test quantization
    quantized_x = torch.ops.quantized_decomposed.quantize_per_tensor(
        x, scale, zero_point, quant_min, quant_max, dtype
    )
    quantized_y = torch.ops.quantized_decomposed.quantize_per_tensor(
        y, scale, zero_point, quant_min, quant_max, dtype
    )

    print("quantized_x is: {}".format(quantized_x), flush=True)
    print("quantized_y is: {}".format(quantized_y), flush=True)

    fp32_x = quantized_x.to(torch.float32)
    fp32_y = quantized_y.to(torch.float32)

    test_abs(fp32_x, fp32_y, quantized_x, quantized_y)
    test_mul(fp32_x, fp32_y, quantized_x, quantized_y)
    # test_eq(fp32_x, fp32_y, quantized_x, quantized_y)

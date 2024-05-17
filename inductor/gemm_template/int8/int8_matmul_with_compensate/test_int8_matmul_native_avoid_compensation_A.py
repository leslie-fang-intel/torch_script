import torch
import random
import numpy as np

random.seed(2023)
torch.manual_seed(2023)
np.random.seed(2023)

def test_int8_single_matmul():
    K = 4
    a = torch.randint(1, 10, (3, K)).to(torch.float32)
    b = torch.randint(1, 10, (K, 3)).to(torch.float32)

    # Prepare inputs to do fp32/int8 calculation
    a_scale = 0.02
    a_zp = 10
    b_scale = 0.04
    b_zp = 5
    fake_a = torch.quantize_per_tensor(a, a_scale, a_zp, torch.quint8).dequantize()
    fake_b = torch.quantize_per_tensor(b, b_scale, b_zp, torch.quint8).dequantize()

    # Start the FP32 calculation
    ref_c = torch.matmul(fake_a, fake_b)

    # Start the int8 calculation
    int_a = torch.clamp(torch.round(fake_a / a_scale) + a_zp, 0, 255).to(torch.uint8)
    int_b = torch.clamp(torch.round(fake_b / b_scale) + b_zp, 0, 255).to(torch.uint8)

    # 注意： 这里可能有溢出！！！
    int_b = int_b - b_zp

    # Int8 matmul to get int32 result
    int_c = torch.matmul(int_a.to(torch.int32), int_b.to(torch.int32)).to(torch.float32)

    # Do the composentation
    scaled_c = a_scale * b_scale * int_c
    B_composentation = a_scale * b_scale * a_zp * torch.sum(int_b, dim=0, keepdim=True)
    
    # A_composentation = a_scale * b_scale * b_zp * torch.sum(int_a, dim=1, keepdim=True)
    # other_composentation = a_scale * a_zp * b_scale * b_zp * K
    # c = scaled_c - B_composentation - A_composentation + other_composentation
    c = scaled_c - B_composentation

    print("ref result is: {}".format(ref_c), flush=True)
    print("result is: {}".format(c), flush=True)
    print(torch.allclose(ref_c, c, atol=1e-3, rtol=1e-3), flush=True)

if __name__ == "__main__":
    test_int8_single_matmul()

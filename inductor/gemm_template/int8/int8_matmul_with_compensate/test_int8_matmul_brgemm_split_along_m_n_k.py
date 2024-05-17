import torch
import random
import numpy as np

random.seed(2023)
torch.manual_seed(2023)
np.random.seed(2023)


def test_int8_brgemm_matmul_split_along_k():
    M = 128
    K = 512
    N = 256
    a = torch.randint(1, 10, (M, K)).to(torch.float32)
    b = torch.randint(1, 10, (K, N)).to(torch.float32)

    # Prepare inputs to do fp32/int8 calculation
    a_scale = 0.02
    a_zp = 10
    b_scale = 0.04
    b_zp = 5
    fake_a = torch.quantize_per_tensor(a, a_scale, a_zp, torch.quint8).dequantize()
    fake_b = torch.quantize_per_tensor(b, b_scale, b_zp, torch.quint8).dequantize()

    # Start the FP32 calculation
    ref_c = torch.matmul(fake_a, fake_b)

    m_block_size = 16
    m_num_block = int((M - 1) // m_block_size + 1)
    k_block_size = 32
    k_num_block = int((K - 1) // k_block_size + 1) 

    int_a = torch.clamp(torch.round(fake_a / a_scale) + a_zp, 0, 255).to(torch.uint8)
    int_b = torch.clamp(torch.round(fake_b / b_scale) + b_zp, 0, 255).to(torch.uint8)

    B_composentation = a_scale * b_scale * a_zp * torch.sum(int_b, dim=0, keepdim=True)
    A_composentation = a_scale * b_scale * b_zp * torch.sum(int_a, dim=1, keepdim=True)
    other_composentation = a_scale * a_zp * b_scale * b_zp * K

    int_c = torch.zeros(M, N)
    c = torch.zeros(M, N)

    # Parallel along this dimension
    for m_block_idx in range(m_num_block):
        # Inside single thread
        int_c_block = int_c[m_block_idx * m_block_size:(m_block_idx + 1) * m_block_size, :]
        for k_block_idx in range(k_num_block):
            k_start = k_block_idx * k_block_size
            k_end = min(K, (k_block_idx + 1) * k_block_size)

            int_a_block = int_a[m_block_idx * m_block_size:(m_block_idx + 1) * m_block_size, k_start:k_end]
            int_b_block = int_b[k_start:k_end, :]

            # Int8 matmul to get int32 result
            int_c_block += torch.matmul(int_a_block.to(torch.int32), int_b_block.to(torch.int32))

            if k_block_idx == (k_num_block - 1):
                # Do the composentation for the last brgemm
                int_c_block = int_c_block.to(torch.float32)
                scaled_c = a_scale * b_scale * int_c_block
                c[m_block_idx * m_block_size:(m_block_idx + 1) * m_block_size, :] = (
                    scaled_c
                    - B_composentation
                    - A_composentation[m_block_idx * m_block_size:(m_block_idx + 1) * m_block_size, :]
                    + other_composentation
                )

    print("ref result is: {}".format(ref_c), flush=True)
    print("result is: {}".format(c), flush=True)
    print(torch.allclose(ref_c, c, atol=1e-3, rtol=1e-3), flush=True)

if __name__ == "__main__":
    test_int8_brgemm_matmul_split_along_k()
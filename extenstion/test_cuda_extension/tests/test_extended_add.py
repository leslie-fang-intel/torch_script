import torch
import torch_cuda_extension
import numpy as np
import random

local_seed = 2025

torch.manual_seed(local_seed) # Set PyTorch seed
np.random.seed(seed=local_seed) # Set Numpy seed
random.seed(local_seed) # Set the Python seed

def test_extended_add():
    shape = (512, 1024)
    a = torch.randn(*shape).to("cuda")
    b = torch.randn(*shape).to("cuda")
    ref_res = a + b
    res = torch.ops.torch_cuda_extension.extended_add(a, b)
    print("torch.allclose(res, ref_res) is: {}".format(torch.allclose(res, ref_res)), flush=True)
    assert torch.allclose(res, ref_res), "accuracy failed to check"

def test_extended_gemm():
    shape = (64, 64)
    epilogue = "none"
    transpose_B = False
    a = torch.randn(*shape).to("cuda")
    b = torch.randn(*shape).to("cuda")
    ref_res = torch.mm(a, b)
    res = torch.ops.torch_cuda_extension.extended_gemm(a, b, epilogue, transpose_B, api_level=0)
    # print("ref_res is: {}".format(ref_res), flush=True)
    # print("res is: {}".format(res), flush=True)
    accuracy_check = torch.allclose(res, ref_res, atol=1e-2, rtol=1e-2)
    print("torch.allclose(res, ref_res) is: {}".format(accuracy_check), flush=True)
    assert accuracy_check, "accuracy failed to check"

def test_extended_gemm_collective():
    # for epilogue in ["none", "relu"]:
    for epilogue in ["none",]:
        shape = (64, 64)
        a = torch.randn(*shape).to("cuda").to(torch.float16)
        b = torch.randn(*shape).to("cuda").to(torch.float16)
        ref_res = torch.mm(a, b)
        if epilogue == "relu":
            ref_res = torch.nn.functional.relu(ref_res)
        # Transpose B to column major
        transpose_B = True
        b = b.t().contiguous()
        res = torch.ops.torch_cuda_extension.extended_gemm(a, b, epilogue, transpose_B, api_level=1)
        accuracy_check = torch.allclose(res, ref_res, atol=1e-2, rtol=1e-1)
        print("torch.allclose(res, ref_res) is: {}".format(accuracy_check), flush=True)
        # torch.testing.assert_allclose(res, ref_res, atol=1e-2, rtol=1e-1)
        assert accuracy_check, "accuracy failed to check"
    print("---- Done test_extended_gemm_cute ----", flush=True)

def test_extended_gemm_cute():
    for epilogue in ["none", "relu"]:
        shape = (64, 64)
        a = torch.randn(*shape).to("cuda").to(torch.float16)
        b = torch.randn(*shape).to("cuda").to(torch.float16)
        ref_res = torch.mm(a, b)
        if epilogue == "relu":
            ref_res = torch.nn.functional.relu(ref_res)
        # Transpose B to column major
        transpose_B = True
        b = b.t().contiguous()
        res = torch.ops.torch_cuda_extension.extended_gemm(a, b, epilogue, transpose_B, api_level=2)
        accuracy_check = torch.allclose(res, ref_res, atol=1e-2, rtol=1e-1)
        print("torch.allclose(res, ref_res) is: {}".format(accuracy_check), flush=True)
        # torch.testing.assert_allclose(res, ref_res, atol=1e-2, rtol=1e-1)
        assert accuracy_check, "accuracy failed to check"
    print("---- Done test_extended_gemm_cute ----", flush=True)

def test_extended_gemm_cute_float():
    for epilogue in ["none", "relu"]:
        shape = (64, 64)
        a = torch.randn(*shape).to("cuda").to(torch.float16)
        b = torch.randn(*shape).to("cuda").to(torch.float16)
        ref_res = torch.mm(a, b).to(torch.float32)
        if epilogue == "relu":
            ref_res = torch.nn.functional.relu(ref_res)
        # Transpose B to column major
        transpose_B = True
        b = b.t().contiguous()
        res = torch.ops.torch_cuda_extension.extended_gemm(a, b, epilogue, transpose_B, torch.float32, api_level=2)
        accuracy_check = torch.allclose(res, ref_res, atol=1e-2, rtol=1e-1)
        print("torch.allclose(res, ref_res) is: {}".format(accuracy_check), flush=True)
        # torch.testing.assert_allclose(res, ref_res, atol=1e-2, rtol=1e-1)
        assert accuracy_check, "accuracy failed to check"
    print("---- Done test_extended_gemm_float ----", flush=True)

def test_extended_attention():
    dtype = torch.float16
    size = [32, 8, 128, 64]    
    query = torch.rand(*size, dtype=dtype, device="cuda")
    key = torch.rand(*size, dtype=dtype, device="cuda")
    value = torch.rand(*size, dtype=dtype, device="cuda")
    with torch.backends.cuda.sdp_kernel(enable_math=False), torch.no_grad():
        ref_res = torch.nn.functional.scaled_dot_product_attention(query, key, value)
        res = torch.ops.torch_cuda_extension.extended_attention(
            query.to(device="cpu"),
            key.to(device="cpu"),
            value.to(device="cpu"),
            api_level=1,
        )
        accuracy_check = torch.allclose(res.to(device="cpu"), ref_res.to(device="cpu"), atol=1e-2, rtol=1e-1)
        # print(res.to(device="cpu"), flush=True)
        # print(ref_res.to(device="cpu"), flush=True)
        # torch.testing.assert_allclose(res.to(device="cpu"), ref_res.to(device="cpu"), atol=1e-2, rtol=1e-1)
        print("torch.allclose(res, ref_res) is: {}".format(accuracy_check), flush=True)
        assert accuracy_check, "accuracy failed to check"
    print("---- Done test_extended_attention ----", flush=True)

if __name__ == "__main__":
    # TODO<leslie> support pytest
    test_extended_add()

    # Highest level API
    test_extended_gemm()

    # Test Collective API
    # test_extended_gemm_collective()

    # Test Cute API
    test_extended_gemm_cute()
    test_extended_gemm_cute_float()

    # Test Attention
    test_extended_attention()

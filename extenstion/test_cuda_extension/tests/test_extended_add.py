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
    res = torch.ops.torch_cuda_extension.extended_gemm(a, b, epilogue, transpose_B)
    # print("ref_res is: {}".format(ref_res), flush=True)
    # print("res is: {}".format(res), flush=True)
    accuracy_check = torch.allclose(res, ref_res, atol=1e-2, rtol=1e-2)
    print("torch.allclose(res, ref_res) is: {}".format(accuracy_check), flush=True)
    assert accuracy_check, "accuracy failed to check"

def test_extended_gemm_v2():
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
        res = torch.ops.torch_cuda_extension.extended_gemm(a, b, epilogue, transpose_B)
        accuracy_check = torch.allclose(res, ref_res, atol=1e-2, rtol=1e-1)
        print("torch.allclose(res, ref_res) is: {}".format(accuracy_check), flush=True)
        torch.testing.assert_allclose(res, ref_res, atol=1e-2, rtol=1e-1)
        assert accuracy_check, "accuracy failed to check"
    print("---- Done test_extended_gemm_v2 ----", flush=True)

if __name__ == "__main__":
    # TODO<leslie> support pytest
    test_extended_add()
    test_extended_gemm()
    test_extended_gemm_v2()

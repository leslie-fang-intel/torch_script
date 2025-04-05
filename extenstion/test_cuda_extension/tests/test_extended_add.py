import torch
import torch_cuda_extension

def test_extended_add():
    shape = (512, 1024)
    a = torch.randn(*shape).to("cuda")
    b = torch.randn(*shape).to("cuda")
    ref_res = a + b
    res = torch.ops.torch_cuda_extension.extended_add(a, b)
    print("torch.allclose(res, ref_res) is: {}".format(torch.allclose(res, ref_res)), flush=True)
    assert torch.allclose(res, ref_res), "accuracy failed to check"

def test_extended_gemm():
    shape = (1024, 1024)
    a = torch.randn(*shape).to("cuda")
    b = torch.randn(*shape).to("cuda")
    ref_res = torch.mm(a, b)
    res = torch.ops.torch_cuda_extension.extended_gemm(a, b)
    accuracy_check = torch.allclose(res, ref_res, atol=1e-3, rtol=1e-3)
    print("torch.allclose(res, ref_res) is: {}".format(accuracy_check), flush=True)
    assert accuracy_check, "accuracy failed to check"

if __name__ == "__main__":
    # TODO<leslie> support pytest
    test_extended_add()
    test_extended_gemm()

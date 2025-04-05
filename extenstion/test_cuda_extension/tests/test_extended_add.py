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

if __name__ == "__main__":
    # TODO<leslie> support pytest
    test_extended_add()

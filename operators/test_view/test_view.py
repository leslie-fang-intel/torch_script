import torch
import torch._dynamo.config

def test_without_permute():
    input = torch.randn((2, 2, 64, 16))

    input_reshape = input.reshape(4, 64, 16)

    # Can't view to (4, 64, 16)
    # since input.size(0) and input.size(1) is not contiguous
    input_view = input.view(4, 64, 16)

def test_with_permute():
    input = torch.randn((2, 64, 2, 16))
    input = input.permute(0, 2, 1, 3)

    # In reshape, have to do clone
    input_reshape = input.reshape(4, 64, 16)

    # Failed here: Can't view to (4, 64, 16)
    # since input.size(0) and input.size(1) is not contiguous
    input_view = input.view(4, 64, 16)


if __name__ == "__main__":
    # test_without_permute()
    test_with_permute()





import torch

lib = torch.library.Library("torchll", "FRAGMENT")
lib.define("toy_test_ll(Tensor a) -> Tensor")
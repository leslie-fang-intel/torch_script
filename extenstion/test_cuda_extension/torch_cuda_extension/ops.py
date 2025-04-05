import torch

lib = torch.library.Library("torch_cuda_extension", "FRAGMENT")
lib.define("extended_add(Tensor a, Tensor b) -> Tensor")  # implement by cuda
lib.define("extended_gemm(Tensor a, Tensor b) -> Tensor")  # implement by cutlass

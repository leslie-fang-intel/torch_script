import torch
import torchao
from torchao.dtypes.nf4tensor import (
    linear_nf4,
    NF4Tensor,
    to_nf4,
    _INNER_TENSOR_NAMES_FOR_SHARDING,
)

def test():
    a = torch.randn(32, 32, dtype=torch.float32, device='cpu')
    a_nf4 = torchao.dtypes.to_nf4(a, 16, 2)

    a_dq = a_nf4.get_original_weight()

    print(" a is :{}".format(a), flush=True)
    print(" a_dq is :{}".format(a_dq), flush=True)

    print("--- finish convert to nf4")

if __name__ == "__main__":
    test()


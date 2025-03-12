import torch
import torch._inductor.config as config
from torch._dynamo.utils import counters

config.freezing = True
config.max_autotune = True
config.max_autotune_gemm_backends = "CPP,"

config.cpp_wrapper = True

M=16
N=52
K=52


class Model(torch.nn.Module):
    def __init__(self,):
        super().__init__()
    def forward(self, x, x2):
        tmp = torch.matmul(x, x2)
        return torch.matmul(x, tmp)


if __name__ == "__main__":
    m = Model().eval()
    input = torch.randn(2, N, N)
    input2 = torch.randn(2, N, K)
    m(input, input2)
    with torch.autocast(device_type="cpu"), torch.no_grad():
        cm = torch.compile(m)
        cm(input, input2)
        print(counters["inductor"]["select_algorithm_autotune"], flush=True)

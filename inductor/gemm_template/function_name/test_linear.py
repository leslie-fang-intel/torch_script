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
        self.linear = torch.nn.Linear(K, N, True)
        self.linear2 = torch.nn.Linear(K, N, True)
        self.relu = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()

    def forward(self, x, x2):
        tmp = self.linear(x)
        tmp = self.relu(tmp)
        return tmp
        tmp = self.linear2(tmp)
        tmp = self.relu2(tmp)
        return tmp
        # return tmp + x2
        return self.relu2(tmp)


if __name__ == "__main__":
    m = Model().eval()
    print("m.linear.weight: ", m.linear.weight, flush=True)
    input = torch.randn(2, M//2, K)
    input2 = torch.randn(M, N)
    with torch.autocast(device_type="cpu"), torch.no_grad():
        cm = torch.compile(m)
        cm(input, input2)
        print(counters["inductor"]["select_algorithm_autotune"], flush=True)

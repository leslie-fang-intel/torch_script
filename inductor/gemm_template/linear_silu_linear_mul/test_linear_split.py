import torch
import torch._inductor.config as config


import random
import numpy as np

random.seed(2023)
torch.manual_seed(2023)
np.random.seed(2023)


config.freezing = True
config.max_autotune = True
# config.max_autotune_gemm_backends = "CPP,ATEN"
config.max_autotune_gemm_backends = "CPP,"
config.aggressive_fusion = True

# config.epilogue_fusion = False

# M=128
# N=1024
# K=4096

M=16
N=32
K=64


class Linear_Gate_Up(torch.nn.Module):
    def __init__(self, in_feature, out_feature, bias_gate=False, bias_up=False):
        super(Linear_Gate_Up, self).__init__()
        self.gate_proj = torch.nn.Linear(in_feature, out_feature, bias=bias_gate)
        self.out_feature = out_feature

    def forward(self, x):
        tmp = self.gate_proj(x)
        tmp1 = torch.nn.functional.silu(tmp[:, :self.out_feature//2])
        # tmp1 = torch.nn.functional.silu(tmp)
        return tmp1
        tmp2 = tmp1 * tmp[:, self.out_feature//2:]
        return tmp2


if __name__ == "__main__":
    m = Linear_Gate_Up(K, N).eval()
    input = torch.randn(M, K)
    with torch.autocast(device_type="cpu", enabled=False), torch.no_grad():
        ref_res = m(input)

        cm = torch.compile(m)
        res = cm(input)

        # print("ref_res: ", ref_res, flush=True)
        # print("res: ", res, flush=True)

        print("correctness: ", torch.allclose(ref_res, res, atol=1e-3, rtol=1e-3), flush=True)

import torch
import torch._inductor.config as config

config.freezing = True
config.max_autotune = True
config.max_autotune_gemm_backends = "CPP,ATEN"

M=128
N=4096
K=4096


class Linear_Gate_Up(torch.nn.Module):
    def __init__(self, in_feature, out_feature, bias_gate, bias_up):
        super(Linear_Gate_Up, self).__init__()
        self.gate_proj = torch.nn.Linear(in_feature, out_feature, bias=bias_gate)
        self.up_proj = torch.nn.Linear(in_feature, out_feature, bias=bias_up)

    def forward(self, x):
        return torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)


if __name__ == "__main__":
    m = Linear_Gate_Up().eval()
    print("m.linear.weight: ", m.linear.weight, flush=True)
    input = torch.randn(M, K)
    input2 = torch.randn(M, N)
    with torch.autocast(device_type="cpu"), torch.no_grad():
        cm = torch.compile(m)
        cm(input, input2)

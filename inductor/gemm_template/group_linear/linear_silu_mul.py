import torch
from torch._inductor import config

batch_size = 4
in_features = 512
out_features = 1024
dtype = torch.bfloat16
bias = False

config.freezing = True
config.max_autotune = True
config.max_autotune_gemm_backends = "CPP"
config.cpp.enable_linear_silu_linear_mul = True

class M(torch.nn.Module):
    def __init__(self, bias):
        super().__init__()
        # self.linear = torch.nn.Linear(in_features, out_features, bias)
        self.gate_proj = torch.nn.Linear(in_features, out_features, bias=bias)
        self.up_proj = torch.nn.Linear(in_features, out_features, bias=bias)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # return self.linear(x)
        return torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        # return self.relu(res)

if __name__ == "__main__":
    with torch.no_grad():
        input = torch.randn(batch_size, in_features, dtype=dtype)
        m = M(bias=bias).to(dtype=dtype).eval()
        ref_res = m(input)
        cm = torch.compile(m)
        act_res = cm(input)
        torch.testing.assert_close(ref_res, act_res, rtol=3e-2, atol=3e-2)
        print("--- finished -----", flush=True)

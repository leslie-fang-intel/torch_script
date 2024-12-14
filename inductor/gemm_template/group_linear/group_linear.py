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
        self.linear0 = torch.nn.Linear(in_features, out_features, bias=bias)
        self.linear1 = torch.nn.Linear(in_features, out_features, bias=bias)
        # self.linear2 = torch.nn.Linear(out_features, out_features, bias=bias)
        self.relu0 = torch.nn.ReLU()
        self.silu0 = torch.nn.SiLU()

    def forward(self, x, x2):
        # return self.linear0(x), self.linear1(x)
        # return self.relu0(self.linear0(x)), self.linear1(x)
        # return self.linear0(x), self.relu0(self.linear1(x))
        # return self.relu0(self.linear0(x)), self.relu0(self.linear1(x))
        # return self.silu0(self.linear0(x)), self.relu0(self.linear1(x))

        tmp1 = self.silu0(self.linear0(x))
        tmp2 = self.linear1(x)
        return tmp1 * tmp2, None

if __name__ == "__main__":
    with torch.no_grad():
        input = torch.randn(batch_size, in_features, dtype=dtype)
        input2 = torch.randn(batch_size, in_features, dtype=dtype)
        m = M(bias=bias).to(dtype=dtype).eval()
        ref_res, ref_res1 = m(input, input2)
        cm = torch.compile(m)
        act_res, act_res1 = cm(input, input2)
        torch.testing.assert_close(ref_res, act_res, rtol=3e-2, atol=3e-2)
        print("---- pass gemm 1 check ----", flush=True)
        torch.testing.assert_close(ref_res1, act_res1, rtol=3e-2, atol=3e-2)
        print("---- pass gemm 2 check ----", flush=True)
        print("---- finish ----", flush=True)

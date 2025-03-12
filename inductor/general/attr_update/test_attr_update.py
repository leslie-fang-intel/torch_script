import torch
from torch._inductor import config
import copy

batch_size = 4
in_features = 512
out_features = 1024
dtype = torch.bfloat16
bias = True

config.freezing = True
# config.max_autotune = True
# config.max_autotune_gemm_backends = "CPP"

# config.cpp.enable_grouped_gemm_template = True
# config.cpp_wrapper = True
# config.allow_buffer_reuse = False

# config.cpp.cpp_gemm_transverse_strategy = "VERTICAL, HORIZONTAL"

class M(torch.nn.Module):
    def __init__(self, bias):
        super().__init__()
        self.linear0 = torch.nn.Linear(in_features, out_features, bias=False)
        self.linear1 = torch.nn.Linear(in_features, out_features, bias=False)
        self.linear2 = torch.nn.Linear(in_features, out_features, bias=bias)
        
        self.linear3 = torch.nn.Linear(out_features, out_features, bias=bias)
        self.linear4 = torch.nn.Linear(out_features, out_features, bias=bias)
        # self.linear2 = torch.nn.Linear(out_features, out_features, bias=bias)
        self.relu0 = torch.nn.ReLU()
        self.relu1 = torch.nn.ReLU()
        self.silu0 = torch.nn.SiLU()
        self.scale = torch.tensor(0.01, dtype=dtype)

    def forward(self, x):
        # Case 5:
        # tmp1 = self.silu0(self.linear0(x))
        # tmp2 = self.linear1(x)
        # res = tmp1 * tmp2 * self.scale
        res = self.linear0(x) * self.scale
        res2 = torch.sum(res)
        # self.scale = self.scale * res2
        
        # it will not be replaced with frozen param since its a mutation
        # https://github.com/pytorch/pytorch/blob/c4523999a15db6092921003e90dd0ca5db68677c/torch/_inductor/freezing.py#L39-L58
        # verify with PyTorch: 90448f0128d8090a07325be24bd513c4d14bed6d
        self.scale *= res2
        return res, res2

if __name__ == "__main__":
    with torch.no_grad():
        input = torch.randn(batch_size, in_features, dtype=dtype)

        m = M(bias=bias).to(dtype=dtype).eval()
        eager_m = copy.deepcopy(m)
        cm = torch.compile(m)

        ref_res = eager_m(input)
        act_res = cm(input)
        print(torch.allclose(ref_res[0], act_res[0]), torch.allclose(ref_res[1], act_res[1]), flush=True)
        print(torch.allclose(eager_m.scale, cm.scale), flush=True)
        print("eager_m.scale {} cm.scale {}".format(eager_m.scale, cm.scale), flush=True)


        ref_res = eager_m(input)
        act_res = cm(input)
        print(torch.allclose(ref_res[0], act_res[0]), torch.allclose(ref_res[1], act_res[1]), flush=True)
        print(torch.allclose(eager_m.scale, cm.scale), flush=True)
        print("eager_m.scale {} cm.scale {}".format(eager_m.scale, cm.scale), flush=True)

        # print("0 cm.scale", cm.scale, flush=True)
        # act_res, res2 = cm(input)
        # print("1 cm.scale", cm.scale, flush=True)
        # print("1 res2", res2, flush=True)
        # act_res, res2 = cm(input)
        # print("2 cm.scale", cm.scale, flush=True)
        # print("2 res2", res2, flush=True)
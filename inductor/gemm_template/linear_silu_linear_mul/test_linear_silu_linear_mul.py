#  numactl -C 96-127 -m 3 python test_linear_silu_linear_mul.py 2>&1 | tee test.log

import torch
import torch._inductor.config as config
import time

shapes = [
    # BS: 1, input: 1024, output: 128 First token
    (4 * 1024, 11008, 4096),
    # BS: 1, input: 1024, output: 128 Next token
    (4, 11008, 4096),
    # (4, 4096, 4096)
    # (4, 32000, 4096)

    # BS: 4, input: 1024, output: 128 First token
    (4 * 4 * 1024, 11008, 4096),
    # BS: 4, input: 1024, output: 128 Next token
    (4 * 4, 11008, 4096),

    # BS: 1, input: 2016, output: 32 First token
    (4 * 2016, 11008, 4096),
    # BS: 4, input: 2016, output: 32 First token
    (4 * 4 * 2016, 11008, 4096),
]


shapes = [
    (4, 11008, 4096),
]


x2 = torch.randn(4 * 4 * 2016, 11008).to(torch.bfloat16)

class Linear_Gate_Up(torch.nn.Module):
    def __init__(self, in_feature, out_feature, bias_gate=False, bias_up=False):
        super(Linear_Gate_Up, self).__init__()
        self.gate_proj = torch.nn.Linear(in_feature, out_feature, bias=bias_gate)
        self.up_proj = torch.nn.Linear(in_feature, out_feature, bias=bias_up)

    def forward(self, x):
        # return torch.nn.functional.silu(self.gate_proj(x))
        # return self.up_proj(x) * x2
        return torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)


config.freezing = True
config.max_autotune = True
config.max_autotune_gemm_backends = "CPP,ATEN"
torch._inductor.config.enable_linear_silu_linear_mul = False
test_ipex = True

if __name__ == "__main__":

    for shape in shapes:
        M, N, K = shape
        dtype = torch.bfloat16

        m = Linear_Gate_Up(K, N).eval()
        input = torch.randn(M, K).to(dtype)

        if not test_ipex:
            with torch.autocast(device_type="cpu"), torch.no_grad():
                ref_res = m(input)
                cm = torch.compile(m)
                res = cm(input)

                torch.testing.assert_close(ref_res, res, atol=3e-2, rtol=3e-2)
                
                warm_up_step = 20
                measurement_step = 100
                total_time = 0.0
                for _ in range(warm_up_step):
                    _ = cm(input)
                for i in range(measurement_step):
                    start_time = time.time()
                    _ = cm(input)
                    total_time += (time.time() - start_time)
                print("Torch M: {}; N: {}; K: {}; avg time is: {} ms".format(M, N, K, (total_time * 1000/measurement_step)), flush=True)
        else:
            import intel_extension_for_pytorch as ipex
            from intel_extension_for_pytorch.cpu._auto_kernel_selection import (
                _enable_tpp,
                _disable_tpp,
            )
            with torch.no_grad():
                _enable_tpp()
                model = ipex.optimize(m, dtype=dtype)
                out = torch.ops.torch_ipex.tpp_fused_gate_up_proj(
                    input,
                    model.gate_proj.weight,
                    model.gate_proj.bias,
                    model.up_proj.weight,
                    model.up_proj.bias,
                )

                # exit(-1)

                # out_linear_silu = torch.ops.torch_ipex.tpp_linear_silu(
                #     input, model.gate_proj.weight, model.gate_proj.bias
                # )

                warm_up_step = 20
                measurement_step = 100
                total_time = 0.0
                for _ in range(warm_up_step):
                    _ = torch.ops.torch_ipex.tpp_fused_gate_up_proj(
                    input,
                    model.gate_proj.weight,
                    model.gate_proj.bias,
                    model.up_proj.weight,
                    model.up_proj.bias,
                )
                for i in range(measurement_step):
                    start_time = time.time()
                    _ = torch.ops.torch_ipex.tpp_fused_gate_up_proj(
                        input,
                        model.gate_proj.weight,
                        model.gate_proj.bias,
                        model.up_proj.weight,
                        model.up_proj.bias,
                    )
                    total_time += (time.time() - start_time)
                print("IPEX M: {}; N: {}; K: {}; avg time is: {} ms".format(M, N, K, (total_time * 1000/measurement_step)), flush=True)

                _disable_tpp()



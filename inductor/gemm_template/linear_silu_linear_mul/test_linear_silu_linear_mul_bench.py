# rm -rf /tmp/torchinductor_leslie/* && rm -rf torch_compile_debug/* && clear && numactl -C 96-127 -m 3 python test_linear_silu_linear_mul_bench.py 2>&1 | tee test.log

import torch
import torch._inductor.config as config
import time

shapes = [
    # BS: 1, input: 1024, output: 128 First token
    (4 * 1024, 11008, 4096),
    # BS: 1, input: 1024, output: 128 Next token
    (4, 11008, 4096),
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
    # BS: 4, input: 2016, output: 32 First token
    (4 * 4 * 2016, 11008, 4096),
]

class Linear_Gate_Up(torch.nn.Module):
    def __init__(self, in_feature, out_feature, bias_gate=False, bias_up=False):
        super(Linear_Gate_Up, self).__init__()
        self.gate_proj = torch.nn.Linear(in_feature, out_feature, bias=bias_gate)
        self.up_proj = torch.nn.Linear(in_feature, out_feature, bias=bias_up)

    def forward(self, x):
        return torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)

config.freezing = True
config.max_autotune = True
config.max_autotune_gemm_backends = "CPP,ATEN"
test_ipex = True

warm_up_step = 5
measurement_step = 10
count = 50
verbose = True


if __name__ == "__main__":

    for shape in shapes:

        M, N, K = shape
        dtype = torch.bfloat16

        mods = []

        def run_eager(x):
            y = []
            for i, mod in enumerate(mods):
                y.append(mod(x[i]))
            return y

        def run_compiled(x):
            return run_eager(x)

        def benchmark(fn, x, iters):
            import gc
            gc.disable()
            durations = []
            for _ in range(iters):
                t0 = time.perf_counter()
                fn(*x)
                duration = (time.perf_counter() - t0) / count
                if verbose:
                    print(f'duration: {duration * 1000:.4f} ms', flush=True)
                durations.append(duration)
            gc.enable()
            return durations

        inputs = []
        if not test_ipex:
            for _ in range(count):
                inputs.append(torch.randn(M, K).to(dtype))
            for _ in range(count):
                mods.append(Linear_Gate_Up(K, N).eval().to(dtype=dtype).eval())

            with torch.no_grad():
                print("---- start the warm up run ----", flush=True)
                benchmark(torch.compile(run_compiled), inputs, warm_up_step)
                print("---- start the formal up run ----", flush=True)
                benchmark(torch.compile(run_compiled), inputs, measurement_step)
        else:
            exit(-1)
            # import intel_extension_for_pytorch as ipex
            # from intel_extension_for_pytorch.cpu._auto_kernel_selection import (
            #     _enable_tpp,
            #     _disable_tpp,
            # )
            # with torch.no_grad():
            #     _enable_tpp()

            #     for _ in range(count):
            #         mods.append(ipex.optimize(Linear_Gate_Up(K, N).eval(), dtype=dtype))
            #     for i in range(count):
            #         inputs.append(torch.randn(M, K).to(dtype))
            #         #     (
            #         #         torch.randn(M, K).to(dtype),
            #         #         mods[i].gate_proj.weight,
            #         #         mods[i].gate_proj.bias,
            #         #         mods[i].up_proj.weight,
            #         #         mods[i].up_proj.bias,
            #         #     )
            #         # )

            #     with torch.no_grad():
            #         print("---- start the ipex warm up run ----", flush=True)
            #         benchmark(torch.ops.torch_ipex.tpp_fused_gate_up_proj, inputs, warm_up_step)
            #         print("---- start the ipex formal up run ----", flush=True)
            #         benchmark(torch.ops.torch_ipex.tpp_fused_gate_up_proj, inputs, measurement_step)

            #     _disable_tpp()



import torch
import time

if __name__ == "__main__":
    group_norm = torch.nn.GroupNorm(32, 960)
    input = torch.randn(2, 960, 96, 96).to(memory_format=torch.channels_last).to(torch.bfloat16)


    warmup_steps = 100

    steps = 1000

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        ref_res = group_norm(input)

        for _ in range(warmup_steps):
            group_norm(input)

        ref_start = time.time()
        for _ in range(steps):
            group_norm(input)
        ref_end = time.time()
    print("ref time is: {}".format(ref_end - ref_start), flush=True)


    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        c_group_norm = torch.compile(group_norm)
        inductor_res = c_group_norm(input)

        for _ in range(warmup_steps):
            c_group_norm(input)

        inductor_start = time.time()
        for _ in range(steps):
            c_group_norm(input)
        inductor_end = time.time()
    print("inductor time is: {}".format(inductor_end - inductor_start), flush=True)
    print(torch.allclose(ref_res, inductor_res, atol=0.01, rtol=0.01), flush=True)




import torch
import time


class M(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 128, 1)
        self.conv3 = torch.nn.Conv2d(64, 128, 1)
        self.group_norm = torch.nn.GroupNorm(32, 128)
        self.conv2 = torch.nn.Conv2d(128, 128, 1)

    def forward(self, input):
        conv = self.conv(input)
        conv3 = torch.relu(self.conv3(input))
        attn_weights = self.group_norm(conv)
        attn_weights = attn_weights + conv3
        attn_weights = torch.relu(attn_weights)
        return self.conv2(attn_weights)

if __name__ == "__main__":
    with torch.no_grad():
        group_norm = M().eval()
        # input = torch.randn(64, 64, 4, 4).to(memory_format=torch.channels_last).to(torch.bfloat16)
        input = torch.randn(64, 64, 4, 4)

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


        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            c_group_norm = torch.compile(group_norm)
            inductor_res = c_group_norm(input)

            for _ in range(warmup_steps):
                c_group_norm(input)

            inductor_start = time.time()
            for _ in range(steps):
                c_group_norm(input)
            inductor_end = time.time()
        print("ref time is: {}".format(ref_end - ref_start), flush=True)
        print("inductor time is: {}".format(inductor_end - inductor_start), flush=True)
        print(torch.allclose(ref_res, inductor_res, atol=0.01, rtol=0.01), flush=True)




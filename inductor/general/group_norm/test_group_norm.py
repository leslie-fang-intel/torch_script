import torch
import torch.nn as nn

class GN(nn.Module):
    def __init__(self, num_groups, num_channels):
        super(GN, self).__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels)

    def forward(self, x):
        return self.gn(x)

input = torch.randn(2, 960, 96, 96).to(memory_format=torch.channels_last)
m = GN(32, 960).eval()
ref_res = m(input)
compiled_m = torch.compile(m)

with torch.no_grad():
    for _ in range(3):
        res = compiled_m(input)

print(torch.allclose(ref_res, res, atol=1e-1, rtol=1e-2), flush=True)

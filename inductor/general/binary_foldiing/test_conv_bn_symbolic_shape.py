import torch

import torch
import torch._inductor.config as config
import torch.nn as nn

config.freezing = True

def test_binary_folding():
    class Conv_Bn_Relu(nn.Module):
        def __init__(self):
            super(Conv_Bn_Relu, self).__init__()
            self.conv = nn.Conv2d(128, 128, 3, bias=False)
            self.bn = nn.BatchNorm2d(128)
            self.act = nn.Hardswish(inplace=True)
    
        def forward(self, x1):
            return self.act(self.bn(self.conv(x1)))

    mod = Conv_Bn_Relu().eval()

    v = torch.randn((128, 128, 8, 8), dtype=torch.float32, requires_grad=False).add(1) # Failed to run
    # v = torch.randn((3, 128, 8, 8), dtype=torch.float32, requires_grad=False).add(1) # Passed

    with torch.no_grad(), torch.autocast(device_type="cpu"):
        torch._dynamo.mark_dynamic(v, 0)
        cmod = torch.compile(mod, dynamic=True)
        ref_res = cmod(v)
        print("---- finished ----", flush=True)

if __name__ == "__main__":
    test_binary_folding()

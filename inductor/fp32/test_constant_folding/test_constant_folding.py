import torch
from torch._inductor import codecache, config, metrics

torch._dynamo.config.verbose = True
torch._inductor.config.trace.enabled = True
torch._inductor.config.trace.debug_log = True
torch._inductor.config.debug = True

class ConvBN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=False, dtype=torch.float, **kwargs)
        self.bn = torch.nn.BatchNorm2d(out_channels, eps=0.001, dtype=torch.float)

    def forward(self, x):
        return self.bn(self.conv(x))

def test():
    @torch.compile()
    def foo(mod, x):
        return mod(x)
    with config.patch({"optimize_for_inference": True}), torch.no_grad():
    #with config.patch({"optimize_for_inference": False}), torch.no_grad():
        mod = ConvBN(3, 9, kernel_size=3, stride=2).cpu().eval()
        x = torch.rand(1, 3, 9, 9).cpu()
        ref_y = mod(x)
        
        ##y = foo(mod, x)

        compiler_mod = torch.compile(mod)

        y = compiler_mod(x)

        print("ref_y is: {}".format(ref_y), flush=True)
        print("y is: {}".format(y), flush=True)


if __name__ == "__main__":
    test()

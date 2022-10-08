from typing import List
import torch
import torchdynamo
import intel_extension_for_pytorch as ipex

class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv = torch.nn.Conv2d(64, 128, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    def forward(self, x):
        x1 = self.conv(x)
        y = torch.flatten(x1, start_dim=1)
        return y

model = SimpleNet()
model = model.to(memory_format=torch.channels_last)
model.eval()
x = torch.rand(64, 64, 3, 3).contiguous(memory_format=torch.channels_last)

with torch.no_grad():
    model.eval()
    x_trace = torch.rand(64, 64, 3, 3)
    model = ipex.optimize(model, dtype=torch.bfloat16, inplace=True, sample_input=x_trace)
    x_trace = x_trace.contiguous(memory_format=torch.channels_last)
    x_trace = x_trace.to(torch.bfloat16)
    with torch.cpu.amp.autocast():
        model = torch.jit.trace(model, x_trace).eval()
    model = torch.jit.freeze(model)
    # Warm up
    print("---------start warm up-----------------")
    for i in range(3):
        model(x)
    print("---------finish warm up-----------------")

x = torch.rand(64, 64, 3, 3).contiguous(memory_format=torch.channels_last).to(torch.bfloat16)

with torch.no_grad():
    for _ in range(2):
        # toy_example(torch.randn(10), torch.randn(10))
        print("------------one step---------------------")
        model(x)
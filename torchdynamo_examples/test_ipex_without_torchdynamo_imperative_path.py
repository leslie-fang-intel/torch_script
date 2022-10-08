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

with torch.no_grad():
    # model.eval()
    model = ipex.optimize(model, dtype=torch.bfloat16, inplace=True)

x = torch.rand(64, 64, 3, 3).contiguous(memory_format=torch.channels_last).to(torch.bfloat16)

with torch.no_grad():
    for _ in range(2):
        # toy_example(torch.randn(10), torch.randn(10))
        print("------------one step---------------------")
        model(x)
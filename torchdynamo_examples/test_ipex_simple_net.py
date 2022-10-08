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

class SimpleNet_if_condition(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv = torch.nn.Conv2d(64, 128, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    def forward(self, x, a):
        x1 = self.conv(x)
        if a.sum() < 0:
            return x
        return x1

model = SimpleNet()
model = model.to(memory_format=torch.channels_last)
model.eval()

def ipex_compiler(model: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print(example_inputs.__len__())
    print(example_inputs[0].size())
    print("-------------ipex_compiler() called with FX graph---------------")
    with torch.no_grad():
        model = model.to(memory_format=torch.channels_last)
        model.eval()
        model = ipex.optimize(model, dtype=torch.bfloat16, inplace=True)

        # print(model)
        # print("-------------------tete-----------")
        # x_bf16 = torch.rand(64, 64, 3, 3).contiguous(memory_format=torch.channels_last).to(torch.bfloat16)
        # model(x_bf16)
        # print("-------------------finish tete-----------")
        return model  # return a python callable

x = torch.rand(64, 64, 3, 3).contiguous(memory_format=torch.channels_last)

with torchdynamo.optimize(ipex_compiler), torch.no_grad(), torch.cpu.amp.autocast():
    print("------------inside torch dynamo optimize---------------------")
    model(x)

with torchdynamo.run(), torch.no_grad(), torch.cpu.amp.autocast():
    model(x)

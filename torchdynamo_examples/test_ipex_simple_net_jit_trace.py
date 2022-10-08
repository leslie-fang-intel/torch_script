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

def ipex_compiler(model: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print(example_inputs.__len__())
    print(example_inputs[0].size())
    print("-------------ipex_compiler() called with FX graph---------------")
    with torch.no_grad():
        model = model.to(memory_format=torch.channels_last)
        model.eval()
        model = ipex.optimize(model, dtype=torch.bfloat16, inplace=True, sample_input=example_inputs)

        with torch.cpu.amp.autocast():
            model = torch.jit.trace(model, example_inputs).eval()
        model = torch.jit.freeze(model)
        # Warm up
        print("---------start warm up-----------------")
        for i in range(3):
            model(*example_inputs)
        return model  # return a python callable

x = torch.rand(64, 64, 3, 3).contiguous(memory_format=torch.channels_last)

with torchdynamo.optimize(ipex_compiler), torch.no_grad(), torch.cpu.amp.autocast():
    print("------------inside torch dynamo optimize---------------------")
    model(x)

with torchdynamo.run(), torch.no_grad(), torch.cpu.amp.autocast():
    print("------------inside torch dynamo run---------------------")
    model(x)

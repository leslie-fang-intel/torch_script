from typing import List
import torch
import torchdynamo
import intel_extension_for_pytorch as ipex

# torchdynamo.config.debug = True
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
        super(SimpleNet_if_condition, self).__init__()
        self.conv = torch.nn.Conv2d(64, 64, (3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(64, 128, (3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv3 = torch.nn.Conv2d(64, 256, (3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x, a):
        x1 = self.relu(self.bn(self.conv(x)))

        if a[0] < 0:
            x2 = self.relu(self.conv2(x1))
            return torch.flatten(x2, start_dim=1)
        return self.relu(self.conv3(x1))

model = SimpleNet_if_condition()
model = model.to(memory_format=torch.channels_last)
model.eval()

def ipex_compiler(model: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    # print(example_inputs.__len__())
    # print(example_inputs[0].size())
    print("-------------ipex_compiler() called with FX graph---------------")
    model.graph.print_tabular()
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
        print(model.graph_for(*example_inputs))
        return model  # return a python callable

x = torch.rand(64, 64, 7, 7).contiguous(memory_format=torch.channels_last)
a = torch.randn(10)
with torchdynamo.optimize(ipex_compiler), torch.no_grad(), torch.cpu.amp.autocast():
    print("------------inside torch dynamo optimize---------------------")
    for i in range(10):
        a = torch.randn(10)
        print(a[0])
        res = model(x, a)
        print("res.size() is: {0}".format(res.size()))

# # with torchdynamo.optimize(ipex_compiler), torch.no_grad(), torch.cpu.amp.autocast():
# #     ref_result = model(x, a)

# with torchdynamo.run(), torch.no_grad(), torch.cpu.amp.autocast():
#     for _ in range(100):
#         print("------------inside torch dynamo run---------------------")
#         a = torch.randn(10)
#         res = model(x, a)
#         print(res.size())

#     # print("------------inside torch dynamo run---------------------")
#     # model(x, a)

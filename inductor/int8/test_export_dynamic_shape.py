import torch
import torch.nn as nn
import torch._dynamo as torchdynamo
import copy

class Mod(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1024, 1000)

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.linear(x)

if __name__ == "__main__":
    x = torch.randn(16, 512, 2)
    example_inputs = (x,)
    model = Mod().eval()
    ref_result = model(*example_inputs)

    model, guards = torchdynamo.export(
        model,
        *copy.deepcopy(example_inputs),
        aten_graph=True,
        tracing_mode="real",
    )

    print("guards is: {}".format(guards), flush=True)
    print("model after export is: {}".format(model), flush=True)

    model(torch.randn(2, 512, 2))

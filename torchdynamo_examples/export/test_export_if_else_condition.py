import torch
import torch._dynamo as torchdynamo
import copy

class Mod(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        if x.size(0) > 10:
            return self.relu(x + x)
        else:
            return self.relu(x)

if __name__ == "__main__":
    example_inputs = (torch.randn(14, 3, 224, 224),)
    m = Mod().eval()
    m(*example_inputs)
    # program capture
    m, guards = torchdynamo.export(
        m,
        *copy.deepcopy(example_inputs),
        aten_graph=True,
        tracing_mode="symbolic",
    )
    print(guards)
    print(m)
import torch

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x):
        #return self.linear(x + self.param).clamp(min=0.0, max=1.0)
        x2 = x + x
        # Split along dim0
        x3 = torch.split(x2, torch.div(x2.size(dim=0), 2, rounding_mode="floor"))
        linear1 = self.linear(x3[0])
        linear2 = self.linear(x3[1])
        return torch.cat((linear1, linear2))

module = MyModule()

from torch.fx import symbolic_trace
# Symbolic tracing frontend - captures the semantics of the module
original_graph_module : torch.fx.GraphModule = symbolic_trace(module)
print(original_graph_module.graph)
print(original_graph_module.code)
original_graph_module.graph.print_tabular()
print(original_graph_module.graph._len)

input_tensor = torch.rand(3, 4)
print(original_graph_module(input_tensor))


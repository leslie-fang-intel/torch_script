import torch
_quantized_add_pattern_example_inputs = (
    torch.randn(1, 1, 3, 3),  # input1
    torch.randn(1, 1, 1, 1),  # weight1
    torch.randn(1),           # bias1
    torch.randn(1, 1, 3, 3),  # input2
    torch.randn(1, 1, 1, 1),  # weight2
    torch.randn(1),           # bias2
)

def conv_add_pattern(input1, weight1, bias1, input2, weight2, bias2):
    conv1 = torch.nn.functional.conv2d(input1, weight1, bias1)
    conv2 = torch.nn.functional.conv2d(input2, weight2, bias2)
    output = conv1 + conv2
    return output, {"input1": input1, "weight1": weight1, "bias1": bias1, "input2": input2, "weight2": weight2, "bias2": bias2, "output": output}

from torch.fx.passes.utils.matcher_with_name_node_map_utils import SubgraphMatcherWithNameNodeMap
from torch._export import capture_pre_autograd_graph

conv_add_pattern_gm = capture_pre_autograd_graph(conv_add_pattern, _quantized_add_pattern_example_inputs)

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3)
        self.conv2 = torch.nn.Conv2d(3, 3, 3)        
    
    def forward(self, x, y):
        x = self.conv1(x)
        y = self.conv2(y)
        return x + y

example_inputs = (torch.randn(1, 3, 3, 3), torch.randn(1, 3, 3, 3))
m = capture_pre_autograd_graph(M().eval(), example_inputs)

matcher = SubgraphMatcherWithNameNodeMap(conv_add_pattern_gm)
matches = matcher.match(m.graph)
print(matches)
print("__len__(matches) is: {}".format(len(matches)), flush=True)

import torch
_quantized_conv_relu_pattern_example_inputs = (
    torch.randn(1, 1, 3, 3),  # input1
    torch.randn(1, 1, 1, 1),  # weight1
    torch.randn(1),           # bias1
)

def conv_relu_pattern(input1, weight1, bias1):
    conv1 = torch.nn.functional.conv2d(input1, weight1, bias1)
    output = torch.nn.functional.relu(conv1)
    return output, {"input1": input1, "weight1": weight1, "bias1": bias1, "output": output}

from torch.fx.passes.utils.matcher_with_name_node_map_utils import SubgraphMatcherWithNameNodeMap
from torch._export import capture_pre_autograd_graph


def test1():
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 3, 3)   
            self.relu = torch.nn.ReLU()  
        
        def forward(self, x):
            x = self.conv1(x)
            return self.relu(x)

    conv_add_pattern_gm = capture_pre_autograd_graph(conv_relu_pattern, _quantized_conv_relu_pattern_example_inputs)

    example_inputs = (torch.randn(1, 3, 3, 3),)
    m = capture_pre_autograd_graph(M().eval(), example_inputs)

    matcher = SubgraphMatcherWithNameNodeMap(conv_add_pattern_gm)
    matches = matcher.match(m.graph)
    print(matches)
    print("__len__(matches) is: {}".format(len(matches)), flush=True)

def test2():
    class M2(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 3, 3, 1, 1)   
            self.relu = torch.nn.ReLU()  
        
        def forward(self, x):
            x = self.conv1(x)
            return self.relu(x)

    conv_add_pattern_gm = capture_pre_autograd_graph(conv_relu_pattern, _quantized_conv_relu_pattern_example_inputs)

    example_inputs = (torch.randn(1, 3, 3, 3),)
    m2 = capture_pre_autograd_graph(M2().eval(), example_inputs)

    matcher = SubgraphMatcherWithNameNodeMap(conv_add_pattern_gm)
    matches = matcher.match(m2.graph)
    print(matches)
    print("__len__(matches) is: {}".format(len(matches)), flush=True)

if __name__ == "__main__":
    test1()
    test2()
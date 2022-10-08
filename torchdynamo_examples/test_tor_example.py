from typing import List
import torch
import torchdynamo
import intel_extension_for_pytorch as ipex

def toy_example(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b
    # x = a / (torch.abs(a) + 1)
    # a =a * 2
    # return a + b

def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print(example_inputs.__len__())
    print(example_inputs[0])
    print(example_inputs[1])
    print("my_compiler() called with FX graph:")
    # import pdb
    # pdb.set_trace()
    print(gm)
    result = gm(example_inputs[0], example_inputs[1])
    gm.graph.print_tabular()
    return gm.forward  # return a python callable

with torchdynamo.optimize(my_compiler):
    for _ in range(3):
        toy_example(torch.randn(10), torch.randn(10))
    # toy_example(torch.randn(10), torch.randn(10))


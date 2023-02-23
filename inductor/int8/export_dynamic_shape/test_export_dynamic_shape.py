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

def fail_case(model, example_inputs):
    ref_result = model(*example_inputs)

    model, guards = torchdynamo.export(
        model,
        *copy.deepcopy(example_inputs),
        aten_graph=True,
        tracing_mode="real",
        #tracing_mode="symbolic",
    )

    print("guards is: {}".format(guards), flush=True)
    print("model after export is: {}".format(model), flush=True)

    res = model(*example_inputs)
    print(type(list(guards)[0]))

    res_changed_bs = model(torch.randn(2, 512, 2))   

def solution1(model, example_inputs):
    # Solution1: When use tracing_mode="symbolic", model support dynamic shapes
    print("---- start solution1 ----", flush=True)
    ref_result = model(*example_inputs)

    model, guards = torchdynamo.export(
        model,
        *copy.deepcopy(example_inputs),
        aten_graph=True,
        tracing_mode="symbolic",
    )

    print("guards is: {}".format(guards), flush=True)
    print("model after export is: {}".format(model), flush=True)

    res = model(*example_inputs)
    print(type(list(guards)[0]))

    res_changed_bs = model(torch.randn(2, 512, 2)) 

def solution2(model, example_inputs):
    # Solution2: manually check the guard
    print("---- start solution2 ----", flush=True)
    ref_result = model(*example_inputs)

    optimized_model, guards = torchdynamo.export(
        model,
        *copy.deepcopy(example_inputs),
        aten_graph=True,
        tracing_mode="real",
    )

    print("model after export is: {}".format(optimized_model), flush=True)

    res = optimized_model(*example_inputs)
    print("guards is: {}".format(guards), flush=True)
    # We get guards as: https://gist.github.com/leslie-fang-intel/b42c8f080a14e212d6c6978d389c6942
    # * guards 是个列表，每一个元素是个 Guard(pytorch/torch/_guards.py::Guard)的对象
    # * Guard对象有很多字段
    #   * source字段对应了一个GuardSource对象，比如例子中的第一个guard的GuardSource就是GuardSource.SHAPE_ENV
    #   * creat_fn是一个GuardBuilder(pytorch/torch/_dynamo/guards.py::GuardBuilder)对象
    #   * guard_types
    #   * code_list


    # 当 tracing_mode="symbolic"，才会有GuardSource.SHAPE_ENV的guard，这个guard的code_list下面有各种需要保证的形状信息
    # 当 tracing_mode="real"，默认假设只能支持 trace时候的输入形状

    input2 = torch.randn(2, 512, 2)
    if input2.size() == example_inputs[0].size():
        res_changed_bs = optimized_model(input2) 
    else:
        res_changed_bs = model(input2)
    print("---- finish test ----")

if __name__ == "__main__":
    x = torch.randn(16, 512, 2)
    example_inputs = (x,)
    model = Mod().eval()
    fail_case(model, example_inputs)

    # Solution1 is to use tracing_mode="symbolic"
    solution1(model, example_inputs)

    # Solution2 is manually checking the guard
    solution2(model, example_inputs)

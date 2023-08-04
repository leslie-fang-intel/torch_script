import torch
import torch._dynamo as torchdynamo
import copy
from torch._inductor.compile_fx import compile_fx

torch._dynamo.config.verbose = True
torch._inductor.config.trace.enabled = True
torch._inductor.config.debug = True
torch._inductor.config.trace.debug_log = True

torch._inductor.config.freezing = True

class M(torch.nn.Module):
    def __init__(self,):
        super().__init__()
    def forward(self, x):
        return torch.round(x)

def test_pytorch_inductor_round():
    example_inputs = (torch.ones((17)) * 4.4, )
    m = M().eval()
    export_model, guards = torchdynamo.export(
        m,
        *copy.deepcopy(example_inputs),
        aten_graph=True
    )
    compiled = compile_fx(export_model, example_inputs)
    res = compiled(*example_inputs)
    print("input is: {}; round res is: {}".format(example_inputs[0], res), flush=True)
    example_inputs = (torch.ones((17)) * 4.6, )
    res = compiled(*example_inputs)
    print("input is: {}; round res is: {}".format(example_inputs[0], res), flush=True)    
    example_inputs = (torch.ones((17)) * 4.5, )
    res = compiled(*example_inputs)
    print("input is: {}; round res is: {}".format(example_inputs[0], res), flush=True)   
    example_inputs = (torch.ones((17)) * 5.5, )
    res = compiled(*example_inputs)
    print("input is: {}; round res is: {}".format(example_inputs[0], res), flush=True)   

class M2(torch.nn.Module):
    def __init__(self,):
        super().__init__()
    def forward(self, x):
        return x.to(torch.uint8)

def test_pytorch_inductor_to_uint8():
    example_inputs = (torch.ones((17)) * 4.4, )
    m = M2().eval()
    export_model, guards = torchdynamo.export(
        m,
        *copy.deepcopy(example_inputs),
        aten_graph=True
    )
    compiled = compile_fx(export_model, example_inputs)
    res = compiled(*example_inputs)
    print("input is: {}; to_int8 res is: {}".format(example_inputs[0], res), flush=True)
    example_inputs = (torch.ones((17)) * 4.6, )
    print(example_inputs[0].dtype)
    res = compiled(*example_inputs)
    print("input is: {}; to_int8 res is: {}".format(example_inputs[0], res), flush=True)    
    example_inputs = (torch.ones((17)) * 4.5, )
    res = compiled(*example_inputs)
    print("input is: {}; to_int8 res is: {}".format(example_inputs[0], res), flush=True)   
    example_inputs = (torch.ones((17)) * 5.5, )
    res = compiled(*example_inputs)
    print("input is: {}; to_int8 res is: {}".format(example_inputs[0], res), flush=True)  

if __name__ == "__main__":
    # test_pytorch_inductor_round()
    test_pytorch_inductor_to_uint8()


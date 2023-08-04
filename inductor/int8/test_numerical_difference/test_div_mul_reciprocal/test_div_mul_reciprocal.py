import torch
import torch._dynamo as torchdynamo
import copy
from torch._inductor.compile_fx import compile_fx

class DivModule(torch.nn.Module):
    def __init__(self,):
        super().__init__()
    def forward(self, x, scale):
        return x/scale
        #return torch.round(x / scale)

def test_div():
    scale = 0.03
    # example_inputs = (torch.ones((17)) * 0.4950000047683716, scale)
    example_inputs = (torch.ones((17)) * 0.4950000047683716, scale)
    m = DivModule().eval()
    export_model, guards = torchdynamo.export(
        m,
        *copy.deepcopy(example_inputs),
        aten_graph=True
    )
    eager_res = export_model(*example_inputs)
    compiled = compile_fx(export_model, example_inputs)
    inductor_res = compiled(*example_inputs)
    # print("div input[0] is: {:.10f}".format(example_inputs[0][0]), flush=True)
    # print("div inductor_res[0] is: {:.10f}".format(inductor_res[0]), flush=True)
    # print("div eager_res[0] is: {:.10f}".format(eager_res[0]), flush=True)

class MulReciprocalModule(torch.nn.Module):
    def __init__(self,):
        super().__init__()
    def forward(self, x, scale):
        inv_scale = 1.0 / scale
        return x * inv_scale
        #return torch.round(x * inv_scale)

def test_mul_reciprocal():
    scale = 0.03
    # example_inputs = (torch.ones((17)) * 0.4950000047683716, scale)
    example_inputs = (torch.ones((17)) * 0.4950000047683716, scale)
    m = MulReciprocalModule().eval()
    export_model, guards = torchdynamo.export(
        m,
        *copy.deepcopy(example_inputs),
        aten_graph=True
    )
    eager_res = export_model(*example_inputs)
    compiled = compile_fx(export_model, example_inputs)
    inductor_res = compiled(*example_inputs)
    # print("mul_reciprocal input[0] is: {:.10f}".format(example_inputs[0][0]), flush=True)
    # print("mul_reciprocal inductor_res[0] is: {:.10f}".format(inductor_res[0]), flush=True)
    # print("mul_reciprocal eager_res[0] is: {:.10f}".format(eager_res[0]), flush=True)

if __name__ == "__main__":
    test_div()
    test_mul_reciprocal()
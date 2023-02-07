import torch
import logging
import torch._dynamo as torchdynamo
import torch._inductor
from torch import _dynamo, _inductor
from torch._inductor import config

torch._dynamo.config.log_level = logging.DEBUG
torch._dynamo.config.verbose = True
torch._inductor.config.trace.enabled = True
torch._inductor.config.debug = True

torch.manual_seed(0)

p1 = torch.rand(13, 7, 3)
p0 = torch.rand(1, 1)

print(p1)
print(p0)

print(p1[0, 0])
# tensor([0.4963, 0.7682, 0.0885])

def fn(x):
    o = torch.where(x, p1, p0)
    return o

x = torch.tensor([[True]])
print(fn(x)[0, 0]) # the same as print(p1[0, 0])
# tensor([0.4963, 0.7682, 0.0885])

compiled = torch.compile(fn)
print(compiled(x)[0, 0]) # WRONG! different from print(p1[0, 0])!
# tensor([0.6080, 0.6080, 0.6080])
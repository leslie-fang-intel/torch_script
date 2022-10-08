from typing import List
import torch
import torchdynamo
import intel_extension_for_pytorch as ipex
import torchvision.models as models

model = models.__dict__["resnet50"](pretrained=True)
model = model.to(memory_format=torch.channels_last)
model.eval()

def ipex_compiler(model: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("-------------ipex_compiler() called with FX graph---------------")
    with torch.no_grad():
        model = model.to(memory_format=torch.channels_last)
        model.eval()
        # model = ipex.optimize(model, dtype=torch.bfloat16, inplace=True, sample_input=torch.rand(1, 3, 224, 224))
        # without sample input, optimize guess a wrong format
        model = ipex.optimize(model, dtype=torch.bfloat16, inplace=True)
        return model  # return a python callable

x = torch.rand(1, 3, 224, 224).contiguous(memory_format=torch.channels_last)

with torchdynamo.optimize(ipex_compiler), torch.no_grad(), torch.cpu.amp.autocast():
    print("------------inside torch dynamo optimize---------------------")
    model(x)

with torchdynamo.run(), torch.no_grad(), torch.cpu.amp.autocast():
    print("------------inside torch dynamo run---------------------")
    model(x)

import torch
import torch.fx.experimental.optimization as optimization
import intel_extension_for_pytorch as ipex
import torchvision.models as models

model = models.__dict__["resnet50"](pretrained=True)
model = model.to(memory_format=torch.channels_last).eval()
x = torch.randn(1, 3, 224, 224).contiguous(memory_format=torch.channels_last)
model = ipex.optimize(model, dtype=torch.float32, inplace=True)
traced_model = torch.jit.trace(model, x).eval()
traced_model = torch.jit.freeze(traced_model)

# Warm up
for i in range(3):
    traced_model(x)

import pdb
pdb.set_trace()
traced_model(x)

torch.jit.save(traced_model, 'scriptmodule.pt')
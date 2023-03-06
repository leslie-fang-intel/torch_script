import torch
import torch._dynamo as torchdynamo
import torchvision.models as models
import copy
from torch._inductor.compile_fx import compile_fx

import logging
torch._dynamo.config.log_level = logging.DEBUG
torch._dynamo.config.verbose = True
torch._inductor.config.trace.enabled = True
torch._inductor.config.debug = True

def test_fp32():
    model = models.__dict__["resnet50"](pretrained=True).eval()
    #tracing_mode = "real"
    tracing_mode = "symbolic"
    example_inputs = (torch.randn(116, 3, 224, 224).to(memory_format=torch.channels_last),)
    m, guards = torchdynamo.export(
        model,
        *copy.deepcopy(example_inputs),
        aten_graph=True,
        tracing_mode=tracing_mode,
    )
    with torch.no_grad():
        base_res = m(*example_inputs)
        base_res = m(*example_inputs)
        print("base_res.size() is: {}".format(base_res.size()), flush=True)
        input2 = torch.randn(200, 3, 224, 224).contiguous(memory_format=torch.channels_last)
        base_res2 = m(input2)
        print("base_res2.size() is: {}".format(base_res2.size()), flush=True)
        input3 = torch.randn(2, 3, 224, 224).contiguous(memory_format=torch.channels_last)
        base_res3 = m(input3)
        print("base_res3.size() is: {}".format(base_res3.size()), flush=True)


        run = compile_fx(m, example_inputs)

        inductor_res = run(*example_inputs)
        inductor_res = run(*example_inputs)
        print("inductor_res.size() is: {}".format(inductor_res.size()), flush=True)
        input2 = torch.randn(200, 3, 224, 224).contiguous(memory_format=torch.channels_last)
        inductor_res2 = run(input2)
        print("inductor_res2.size() is: {}".format(inductor_res2.size()), flush=True)
        input3 = torch.randn(2, 3, 224, 224).contiguous(memory_format=torch.channels_last)
        inductor_res3 = run(input3)
        print("inductor_res3.size() is: {}".format(inductor_res3.size()), flush=True)


if __name__ == "__main__":
    test_fp32()

# Execution Command:
# TORCHINDUCTOR_FREEZING=1 python inductor_quant_int8_mixed_bf16.py
import torch
import torchvision.models as models
import torch._dynamo as torchdynamo
import copy
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import random
import numpy as np
from torch._export import capture_pre_autograd_graph, dynamic_dim

random.seed(2023)
torch.manual_seed(2023)
np.random.seed(2023)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def run_model(model_name):

    torch._dynamo.config.verbose = True
    torch._inductor.config.trace.enabled = True
    torch._inductor.config.trace.debug_log = True
    torch._inductor.config.debug = True
    torch._inductor.config.freezing = True

    class Mod(torch.nn.Module):
        def __init__(self, inplace_add=False, inplace_relu=False) -> None:
            super().__init__()
            bias = True
            # bias = False
            
            self.conv = torch.nn.Conv2d(
                # in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1
                in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=bias
            )
            self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=bias)
            self.conv3 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=bias)
            self.relu = torch.nn.ReLU(inplace=inplace_relu)
            self.relu2 = torch.nn.ReLU(inplace=inplace_relu)

        def forward(self, x):
            # if not self.inplace_add:
            tmp = self.conv(x)
            # tmp = self.relu(tmp)
            # tmp = self.relu2(tmp)
            return self.conv2(tmp)

    model = Mod().eval()

    traced_bs = 50

    x = torch.randn(traced_bs, 3, 224, 224).contiguous(memory_format=torch.channels_last)
    # res = model(x)
    # print(res, flush=True)

    # exit(-1)
    example_inputs = (x,)
    with torch.no_grad():
        exported_model = capture_pre_autograd_graph(
            model,
            example_inputs
        )

        # Create X86InductorQuantizer
        quantizer = X86InductorQuantizer()
        quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())
        # PT2E Quantization flow
        prepared_model = prepare_pt2e(exported_model, quantizer)

        prepared_model(x)

        converted_model = convert_pt2e(prepared_model)
        torch.ao.quantization.move_exported_model_to_eval(converted_model)
        # print("converted_model is: {}".format(converted_model), flush=True)
        # Lower into Inductor

        enable_int8_mixed_bf16 = True
        # enable_int8_mixed_bf16 = False

        ref_res = converted_model(x)

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=enable_int8_mixed_bf16):
            optimized_model = torch.compile(converted_model)

            for step in range(3):
                print("start step: {}".format(step), flush=True)
                res = optimized_model(x)
        
        if enable_int8_mixed_bf16:
            ref_res = ref_res.to(torch.bfloat16)
        # print("ref_res is: {}".format(ref_res), flush=True)
        # print("res is: {}".format(res), flush=True)
        print(torch.allclose(ref_res, res, atol=1e-2, rtol=1e-2), flush=True)

    print("Finish int8 test of model: {}".format(model_name), flush=True)

if __name__ == "__main__":
    model_list=["alexnet","shufflenet_v2_x1_0","mobilenet_v3_large","vgg16","densenet121","mnasnet1_0","squeezenet1_1","mobilenet_v2","resnet50","resnet152","resnet18","resnext50_32x4d"]
    model_list = ["resnet50"]

    for model in model_list:
        run_model(model)


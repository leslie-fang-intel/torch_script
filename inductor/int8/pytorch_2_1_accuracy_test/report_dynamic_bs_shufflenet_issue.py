# Execution Command:
# TORCHINDUCTOR_FREEZING=1 python inductor_quant_acc.py
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
    print("start int8 test of model: {}".format(model_name), flush=True)
    valdir = "/home/dlboostbkc/dataset/Pytorch/val/"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    traced_bs = 49
    val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=traced_bs, shuffle=False,
    num_workers=4, pin_memory=True)
    cal_loader = copy.deepcopy(val_loader)
    model = models.__dict__[model_name](pretrained=True).eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    quant_top1 = AverageMeter('Acc@1', ':6.2f')
    quant_top5 = AverageMeter('Acc@5', ':6.2f')
    x = torch.randn(traced_bs, 3, 224, 224).contiguous(memory_format=torch.channels_last)
    example_inputs = (x,)
    with torch.no_grad():
        exported_model = capture_pre_autograd_graph(
            model,
            example_inputs
        )

        print("exported_model is: {}".format(exported_model), flush=True)

        x1 = torch.randn(traced_bs, 3, 224, 224).contiguous(memory_format=torch.channels_last)
        print("---start x1 run -----", flush=True)
        exported_model(x1)

        x2 = torch.randn(32, 3, 224, 224).contiguous(memory_format=torch.channels_last)
        print("---start x2 run -----", flush=True)
        exported_model(x2)

        return

if __name__ == "__main__":
    model_list = ["shufflenet_v2_x1_0"]
    for model in model_list:
        run_model(model)


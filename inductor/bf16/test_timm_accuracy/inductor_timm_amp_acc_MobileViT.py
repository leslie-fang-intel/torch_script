# Execution Command:
# TORCHINDUCTOR_FREEZING=1 python inductor_timm_amp_acc.py
import torch
import torchvision.models as models
import torch._dynamo as torchdynamo
import copy
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import random
import numpy as np
import timm

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

def run_model_fp32(model_name):
    torch._dynamo.config.verbose = True
    torch._inductor.config.trace.enabled = True
    torch._inductor.config.trace.debug_log = True
    torch._inductor.config.debug = True
    # torch._inductor.config.freezing = True

    print("start FP32 test of model: {}".format(model_name), flush=True)
    model = timm.create_model(model_name, pretrained=True).eval()
    valdir = "/home/dlboostbkc/dataset/Pytorch/val/"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
    #                                  std=[0.5, 0.5, 0.5])

    val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=50, shuffle=False,
    num_workers=0, pin_memory=True)

    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])

    # datasets.ImageFolder(valdir, transforms.Compose([
    #     transforms.Resize(288),
    #     transforms.CenterCrop(256),
    #     transforms.ToTensor(),
    #     # normalize,
    # ])),
    # batch_size=50, shuffle=False,
    # num_workers=4, pin_memory=True)



    # cal_loader = copy.deepcopy(val_loader)
    # model = models.__dict__[model_name](pretrained=True).eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    quant_top1 = AverageMeter('Acc@1', ':6.2f')
    quant_top5 = AverageMeter('Acc@5', ':6.2f')
    # x = torch.randn(50, 3, 224, 224).contiguous(memory_format=torch.channels_last)
    x = torch.randn(50, 3, 224, 224)
    example_inputs = (x,)
    with torch.no_grad():
        # Lower into Inductor
        # optimized_model = torch.compile(model)
        optimized_model = model
        # Benchmark
        for i, (images, target) in enumerate(val_loader):

            # mean = torch.mean(images)
            # std_dev = torch.std(images)
            # images = (images - mean) / std_dev

            quant_output = optimized_model(images)
            quant_acc1, quant_acc5 = accuracy(quant_output, target, topk=(1, 5))
            quant_top1.update(quant_acc1[0], images.size(0))
            quant_top5.update(quant_acc5[0], images.size(0))
            if i % 9 == 0:
                print('step: {}, * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                    .format(i, top1=quant_top1, top5=quant_top5))
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=quant_top1, top5=quant_top5))
        print("Finish fp32 test of model: {}".format(model_name), flush=True)

def run_model_amp(model_name):
    print("start amp test of model: {}".format(model_name), flush=True)
    model = timm.create_model(model_name, pretrained=True).eval()

    valdir = "/home/dlboostbkc/dataset/Pytorch/val/"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=50, shuffle=False,
    num_workers=4, pin_memory=True)
    cal_loader = copy.deepcopy(val_loader)
    # model = models.__dict__[model_name](pretrained=True).eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    quant_top1 = AverageMeter('Acc@1', ':6.2f')
    quant_top5 = AverageMeter('Acc@5', ':6.2f')
    x = torch.randn(50, 3, 224, 224).contiguous(memory_format=torch.channels_last)
    example_inputs = (x,)
    with torch.no_grad(), torch.cpu.amp.autocast():
        # Lower into Inductor
        optimized_model = torch.compile(model)
    # Benchmark
    for i, (images, target) in enumerate(val_loader):
        with torch.no_grad(), torch.cpu.amp.autocast():
            quant_output = optimized_model(images)
        quant_acc1, quant_acc5 = accuracy(quant_output, target, topk=(1, 5))
        quant_top1.update(quant_acc1[0], images.size(0))
        quant_top5.update(quant_acc5[0], images.size(0))

        if i % 9 == 0:
            print('step: {}, * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                .format(i, top1=quant_top1, top5=quant_top5))
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            .format(top1=quant_top1, top5=quant_top5))
    print("Finish amp test of model: {}".format(model_name), flush=True)

if __name__ == "__main__":    
    import os
    os.system("rm -rf /home/lesliefang/pytorch_1_7_1/torch_inductor/torch_script/inductor/bf16/test_timm_accuracy/torch_compile_debug/*")
    os.system("rm -rf /tmp/torchinductor_root/*")
    # model_name_list = ["gluon_inception_v3", "lcnet_050", "mobilevit_s"]
    model_name_list = ["gluon_inception_v3", "lcnet_050"]
    # According https://github.com/huggingface/pytorch-image-models/tree/main, mobilevit_s has no pretrained weight
    # model_name_list = ["mobilevit_s",]
    for model_name in model_name_list:
        run_model_fp32(model_name)
        run_model_amp(model_name)

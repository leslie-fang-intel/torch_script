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
from torch._inductor import config

random.seed(2023)
torch.manual_seed(2023)
np.random.seed(2023)

torch._dynamo.config.verbose = True
torch._inductor.config.trace.enabled = True
torch._inductor.config.trace.debug_log = True
torch._inductor.config.debug = True
# torch._inductor.config.freezing = True

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

    print("start FP32 test of model: {}".format(model_name), flush=True)
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
    quant_top1 = AverageMeter('Acc@1', ':6.2f')
    quant_top5 = AverageMeter('Acc@5', ':6.2f')
    # x = torch.randn(50, 3, 224, 224).contiguous(memory_format=torch.channels_last)
    x = torch.randn(50, 3, 224, 224)
    example_inputs = (x,)
    with torch.no_grad():
        # Lower into Inductor
        # optimized_model = torch.compile(model)

        # FP32 CONV_BN FOLDING
        
        # import torch.fx.experimental.optimization as optimization
        # model = optimization.fuse(model, inplace=True)
        # import torch.fx as fx
        # model = fx.symbolic_trace(model)
        # from torch._inductor.fx_passes.pre_grad import fuse_conv_bn
        # model = fuse_conv_bn(model)

        # print(model)

        # exit(-1)

        optimized_model = torch.compile(model)
        # optimized_model = model

        # Benchmark
        for i, (images, target) in enumerate(val_loader):
            quant_output = optimized_model(images)
            quant_acc1, quant_acc5 = accuracy(quant_output, target, topk=(1, 5))
            quant_top1.update(quant_acc1[0], images.size(0))
            quant_top5.update(quant_acc5[0], images.size(0))
            # if i % 9 == 0:
            print('step: {}, * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                .format(i, top1=quant_top1, top5=quant_top5), flush=True)
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=quant_top1, top5=quant_top5), flush=True)
        print("Finish fp32 test of model: {}".format(model_name), flush=True)

def run_model_amp(model_name):
    print("start amp test of model: {}".format(model_name), flush=True)
    model = timm.create_model(model_name, pretrained=True).eval()

    # x = torch.randn(50, 3, 224, 224)
    # torch.onnx.export(model, x, '/home/lesliefang/pytorch_1_7_1/torch_inductor/torch_script/inductor/bf16/test_timm_accuracy/lcnet_050.onnx')

    # exit(-1)

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
    quant_top1 = AverageMeter('Acc@1', ':6.2f')
    quant_top5 = AverageMeter('Acc@5', ':6.2f')
    
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
                .format(i, top1=quant_top1, top5=quant_top5), flush=True)
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            .format(top1=quant_top1, top5=quant_top5), flush=True)
    print("Finish amp test of model: {}".format(model_name), flush=True)

def run_ipex_amp_jit(model_name):
    import intel_extension_for_pytorch as ipex
    print("start amp test of model: {}".format(model_name), flush=True)
    model = timm.create_model(model_name, pretrained=True).eval()

    # x = torch.randn(50, 3, 224, 224)
    # torch.onnx.export(model, x, '/home/lesliefang/pytorch_1_7_1/torch_inductor/torch_script/inductor/bf16/test_timm_accuracy/lcnet_050.onnx')

    # exit(-1)

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
    quant_top1 = AverageMeter('Acc@1', ':6.2f')
    quant_top5 = AverageMeter('Acc@5', ':6.2f')
    # with torch.no_grad(), torch.cpu.amp.autocast():
    #     # Lower into Inductor
    #     optimized_model = torch.compile(model)

    x = torch.randn(50, 3, 224, 224).to(memory_format=torch.channels_last)

    model = ipex.optimize(model, dtype=torch.bfloat16, inplace=True)

    with torch.cpu.amp.autocast(), torch.no_grad():
        optimized_model = torch.jit.trace(model, x).eval()

    optimized_model = torch.jit.freeze(optimized_model)
    with torch.no_grad():
        optimized_model(x)
        optimized_model(x)

    # Benchmark
    for i, (images, target) in enumerate(val_loader):
        with torch.no_grad():
            quant_output = optimized_model(images)
        quant_acc1, quant_acc5 = accuracy(quant_output, target, topk=(1, 5))
        quant_top1.update(quant_acc1[0], images.size(0))
        quant_top5.update(quant_acc5[0], images.size(0))

        if i % 9 == 0:
            print('step: {}, * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                .format(i, top1=quant_top1, top5=quant_top5), flush=True)
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            .format(top1=quant_top1, top5=quant_top5), flush=True)
    print("Finish amp test of model: {}".format(model_name), flush=True)


def run_inductor_ipex_amp_jit(model_name):
    import intel_extension_for_pytorch as ipex
    print("start amp test of model: {}".format(model_name), flush=True)
    model = timm.create_model(model_name, pretrained=True).eval()

    inductor_model = timm.create_model(model_name, pretrained=True).eval()

    # print("inductor_model is: {}".format(inductor_model), flush=True)
    # exit(-1)
    # x = torch.randn(50, 3, 224, 224)
    # torch.onnx.export(model, x, '/home/lesliefang/pytorch_1_7_1/torch_inductor/torch_script/inductor/bf16/test_timm_accuracy/lcnet_050.onnx')

    # print(inductor_model.conv_stem)
    # exit(-1)

    valdir = "/home/dlboostbkc/dataset/Pytorch/val/"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    batch_size = 1

    val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=batch_size, shuffle=False,
    num_workers=4, pin_memory=True)
    quant_top1 = AverageMeter('Acc@1', ':6.2f')
    quant_top5 = AverageMeter('Acc@5', ':6.2f')

    quant_top1_inductor = AverageMeter('Acc@1', ':6.2f')
    quant_top5_inductor = AverageMeter('Acc@5', ':6.2f')
    # with torch.no_grad(), torch.cpu.amp.autocast():
    #     # Lower into Inductor
    #     optimized_model = torch.compile(model)

    x = torch.randn(batch_size, 3, 224, 224).to(memory_format=torch.channels_last)

    model = ipex.optimize(model, dtype=torch.bfloat16, inplace=True)
    # print("model is: {}".format(model), flush=True)

    # print(model.bn1.weight.dtype, flush=True)

    with torch.cpu.amp.autocast(), torch.no_grad():
        optimized_model = torch.jit.trace(model, x).eval()

    optimized_model = torch.jit.freeze(optimized_model)
    with torch.no_grad():
        optimized_model(x)
        optimized_model(x)
        print(optimized_model.graph_for(x), flush=True)
    
    # exit(-1)

    with config.patch({"cpp.simdlen": 1}):
        import torch.fx.experimental.optimization as optimization
        
        # inductor_model = optimization.fuse(inductor_model, inplace=True)
        # print("inductor_model is: {}".format(inductor_model), flush=True)
        
        with torch.no_grad(), torch.cpu.amp.autocast():
            # Lower into Inductor
            # optimized_inductor_model = torch.compile(inductor_model, backend="aot_eager_decomp_partition")
            optimized_inductor_model = torch.compile(inductor_model)
            optimized_inductor_model(x)
            optimized_inductor_model(x)
        # Benchmark
        for i, (images, target) in enumerate(val_loader):
            print("---- start step: {} ----".format(i), flush=True)
            # if i == 31:
            #     torch.save(images, '/home/lesliefang/pytorch_1_7_1/torch_inductor/torch_script/inductor/bf16/test_timm_accuracy/images.pt')
            #     torch.save(target, '/home/lesliefang/pytorch_1_7_1/torch_inductor/torch_script/inductor/bf16/test_timm_accuracy/target.pt')

            images = torch.load('/home/lesliefang/pytorch_1_7_1/torch_inductor/torch_script/inductor/bf16/test_timm_accuracy/images.pt')
            target = torch.load('/home/lesliefang/pytorch_1_7_1/torch_inductor/torch_script/inductor/bf16/test_timm_accuracy/target.pt')

            images = images.to(memory_format=torch.channels_last)
            with torch.no_grad():
                quant_output = optimized_model(images)
            quant_acc1, quant_acc5 = accuracy(quant_output, target, topk=(1, 5))
            quant_top1.update(quant_acc1[0], images.size(0))
            quant_top5.update(quant_acc5[0], images.size(0))

            # if i % 9 == 0:
            #     print('step: {}, * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            #         .format(i, top1=quant_top1, top5=quant_top5), flush=True)
            print('step: {}, * Acc@1 {top1:.3f} Acc@5 {top5:.3f}'
                .format(i, top1=quant_acc1[0], top5=quant_acc5[0]), flush=True)

            with torch.no_grad(), torch.cpu.amp.autocast():
                quant_output_inductor = optimized_inductor_model(images)
            
            # print("quant_output_inductor.size() is: {}".format(quant_output_inductor.size()), flush=True)

            quant_acc1_inductor, quant_acc5_inductor = accuracy(quant_output_inductor, target, topk=(1, 5))
            quant_top1_inductor.update(quant_acc1_inductor[0], images.size(0))
            quant_top5_inductor.update(quant_acc5_inductor[0], images.size(0))

            # if i % 9 == 0:
            #     print('Inductor step: {}, * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            #         .format(i, top1=quant_top1_inductor, top5=quant_top5_inductor), flush=True)
            print('Inductor step: {}, * Acc@1 {top1:.3f} Acc@5 {top5:.3f}'
                .format(i, top1=quant_acc1_inductor[0], top5=quant_acc5_inductor[0]), flush=True)

            print(torch.allclose(quant_output, quant_output_inductor, rtol=0.5, atol=0.5), flush=True)

            # print("quant_output is: {}".format(quant_output), flush=True)
            # print("quant_output_inductor is: {}".format(quant_output_inductor), flush=True)

            exit(-1)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                .format(top1=quant_top1, top5=quant_top5), flush=True)
        print("Finish amp test of model: {}".format(model_name), flush=True)


if __name__ == "__main__":    
    # import os
    # os.system("rm -rf /home/lesliefang/pytorch_1_7_1/torch_inductor/torch_script/inductor/bf16/test_timm_accuracy/torch_compile_debug/*")
    # os.system("rm -rf /tmp/torchinductor_root/*")
    
    # model_name_list = ["gluon_inception_v3", "lcnet_050", "mobilevit_s"]
    # model_name_list = ["gluon_inception_v3", "lcnet_050"]
    model_name_list = ["lcnet_050"]
    for model_name in model_name_list:
        # run_model_fp32(model_name)
        run_model_amp(model_name)
        # run_ipex_amp_jit(model_name)
        # run_inductor_ipex_amp_jit(model_name)



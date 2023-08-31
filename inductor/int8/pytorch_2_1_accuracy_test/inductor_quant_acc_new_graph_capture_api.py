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

    torch._dynamo.config.verbose = True
    torch._inductor.config.trace.enabled = True
    torch._inductor.config.trace.debug_log = True
    torch._inductor.config.debug = True
    # torch._inductor.config.freezing = True

    print("start int8 test of model: {}".format(model_name), flush=True)
    valdir = "/home/dlboostbkc/dataset/Pytorch/val/"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    traced_bs = 50
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
        # Generate the FX Module
        # export_with_dynamic_shape = False
        # exported_model = capture_pre_autograd_graph(
        #     model,
        #     example_inputs,
        #     constraints=[dynamic_dim(example_inputs[0], 0)]
        #     if export_with_dynamic_shape
        #     else [],
        # )

        exported_model = capture_pre_autograd_graph(
            model,
            example_inputs
        )

        # Create X86InductorQuantizer
        quantizer = X86InductorQuantizer()
        quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())
        # PT2E Quantization flow
        prepared_model = prepare_pt2e(exported_model, quantizer)
        # print("prepared_model is: {}".format(prepared_model), flush=True)
        # from torch.fx.passes.graph_drawer import FxGraphDrawer
        # g = FxGraphDrawer(prepared_model, "shuffnetv2")
        # g.get_dot_graph().write_svg("//home/lesliefang/pytorch_1_7_1/inductor_quant/torch_script/inductor/int8/pytorch_2_1_accuracy_test/new_frontend_shuffnetv2_prepare.svg")
        # Calibration
        for i, (images, _) in enumerate(cal_loader):
            prepared_model(images)
            if i==10: break
        converted_model = convert_pt2e(prepared_model).eval()
        # print("converted_model is: {}".format(converted_model), flush=True)
        # Lower into Inductor
        optimized_model = torch.compile(converted_model)

        # Benchmark
        for i, (images, target) in enumerate(val_loader):
            #output = model(images)
            #acc1, acc5 = accuracy(output, target, topk=(1, 5))
            #top1.update(acc1[0], images.size(0))
            #top5.update(acc5[0], images.size(0))
            quant_output = optimized_model(images)
            quant_acc1, quant_acc5 = accuracy(quant_output, target, topk=(1, 5))
            quant_top1.update(quant_acc1[0], images.size(0))
            quant_top5.update(quant_acc5[0], images.size(0))

            if i % 9 == 0:
                print('step: {}, * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                    .format(i, top1=quant_top1, top5=quant_top5), flush=True)    

        print(model_name + " int8: ")
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=quant_top1, top5=quant_top5))
    print("Finish int8 test of model: {}".format(model_name), flush=True)

def run_model_fp32(model_name):
    print("start fp32 test of model: {}".format(model_name), flush=True)
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
    model = models.__dict__[model_name](pretrained=True).eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    quant_top1 = AverageMeter('Acc@1', ':6.2f')
    quant_top5 = AverageMeter('Acc@5', ':6.2f')
    x = torch.randn(50, 3, 224, 224).contiguous(memory_format=torch.channels_last)
    example_inputs = (x,)
    with torch.no_grad():
        # Lower into Inductor
        optimized_model = torch.compile(model)
        # Benchmark
        for i, (images, target) in enumerate(val_loader):
            #output = model(images)
            #acc1, acc5 = accuracy(output, target, topk=(1, 5))
            #top1.update(acc1[0], images.size(0))
            #top5.update(acc5[0], images.size(0))
            quant_output = optimized_model(images)
            quant_acc1, quant_acc5 = accuracy(quant_output, target, topk=(1, 5))
            quant_top1.update(quant_acc1[0], images.size(0))
            quant_top5.update(quant_acc5[0], images.size(0))

            # if i % 9 == 0:
            #     print('step: {}, * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            #         .format(i, top1=quant_top1, top5=quant_top5))
        print(model_name + " fp32: ")
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=quant_top1, top5=quant_top5))
    print("Finish fp32 test of model: {}".format(model_name), flush=True)

def run_model_fx_x86_int8(model_name):
    print("start fx x86 int8 test of model: {}".format(model_name), flush=True)
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
    model = models.__dict__[model_name](pretrained=True).eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    quant_top1 = AverageMeter('Acc@1', ':6.2f')
    quant_top5 = AverageMeter('Acc@5', ':6.2f')
    x = torch.randn(50, 3, 224, 224).contiguous(memory_format=torch.channels_last)
    example_inputs = (x,)
    with torch.no_grad():
        # Generate the FX Module
        from torch.ao.quantization import get_default_qconfig_mapping
        from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
        torch.backends.quantized.engine = 'x86'
        qconfig_mapping = get_default_qconfig_mapping('x86')
        # torch.backends.quantized.engine = 'onednn'
        # qconfig_mapping = get_default_qconfig_mapping('onednn')
        prepared_model = prepare_fx(model, qconfig_mapping, x)

        # Calibration
        for i, (images, _) in enumerate(cal_loader):
            prepared_model(images)
            if i==10: break
        converted_model = convert_fx(prepared_model).eval()
        # Jit Trace
        with torch.no_grad():
            optimized_model = torch.jit.trace(converted_model, x).eval()
            optimized_model = torch.jit.freeze(optimized_model)
            for _ in range(3):
                optimized_model(x)
            print("---- print jit model -----", flush=True)
            print(optimized_model.graph_for(x), flush=True)

        # Benchmark
        for i, (images, target) in enumerate(val_loader):
            #output = model(images)
            #acc1, acc5 = accuracy(output, target, topk=(1, 5))
            #top1.update(acc1[0], images.size(0))
            #top5.update(acc5[0], images.size(0))
            quant_output = optimized_model(images)
            quant_acc1, quant_acc5 = accuracy(quant_output, target, topk=(1, 5))
            quant_top1.update(quant_acc1[0], images.size(0))
            quant_top5.update(quant_acc5[0], images.size(0))

            # if i % 9 == 0:
            #     print('step: {}, * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            #         .format(i, top1=quant_top1, top5=quant_top5))                
        print(model_name + " int8: ")
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=quant_top1, top5=quant_top5))
    print("Finish fx x86 int8 test of model: {}".format(model_name), flush=True)

def run_model_ipex_int8(model_name):
    print("start ipex int8 test of model: {}".format(model_name), flush=True)
    import intel_extension_for_pytorch as ipex
    from torch.ao.quantization import HistogramObserver, MinMaxObserver, PerChannelMinMaxObserver, QConfig
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
    model = models.__dict__[model_name](pretrained=True).eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    quant_top1 = AverageMeter('Acc@1', ':6.2f')
    quant_top5 = AverageMeter('Acc@5', ':6.2f')
    x = torch.randn(50, 3, 224, 224).contiguous(memory_format=torch.channels_last)
    example_inputs = (x,)
    with torch.no_grad():
        # Generate the FX Module
        # qconfig = QConfig(
        #         activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8),
        #         weight= PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))
        qconfig = QConfig(
                activation=HistogramObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.quint8),
                weight= PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))
        prepared_model = ipex.quantization.prepare(model, qconfig, x, inplace=True)

        # Calibration
        for i, (images, _) in enumerate(cal_loader):
            prepared_model(images.contiguous(memory_format=torch.channels_last))
            if i==10: break
        converted_model = ipex.quantization.convert(prepared_model).eval()
        # Jit Trace
        with torch.no_grad():
            optimized_model = torch.jit.trace(converted_model, x).eval()
            optimized_model = torch.jit.freeze(optimized_model)
            for _ in range(3):
                optimized_model(x)

        # Benchmark
        for i, (images, target) in enumerate(val_loader):
            #output = model(images)
            #acc1, acc5 = accuracy(output, target, topk=(1, 5))
            #top1.update(acc1[0], images.size(0))
            #top5.update(acc5[0], images.size(0))
            quant_output = optimized_model(images.contiguous(memory_format=torch.channels_last))
            quant_acc1, quant_acc5 = accuracy(quant_output, target, topk=(1, 5))
            quant_top1.update(quant_acc1[0], images.size(0))
            quant_top5.update(quant_acc5[0], images.size(0))

            # if i % 9 == 0:
            #     print('step: {}, * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            #         .format(i, top1=quant_top1, top5=quant_top5), flush=True)                
        print(model_name + " int8: ")
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=quant_top1, top5=quant_top5))
    print("Finish ipex int8 test of model: {}".format(model_name), flush=True)

if __name__ == "__main__":
    # model_list=["alexnet","shufflenet_v2_x1_0","mobilenet_v3_large","vgg16","densenet121","mnasnet1_0","squeezenet1_1","mobilenet_v2","resnet50","resnet152","resnet18","resnext50_32x4d"]
    # model_list = ["resnet50","squeezenet1_1","mobilenet_v2","mobilenet_v3_large"]
    # model_list = ["densenet121","mnasnet1_0","squeezenet1_1","mobilenet_v2","resnet50","resnet152","resnet18","resnext50_32x4d"]
    
    # model_list = ["shufflenet_v2_x1_0",]
    model_list = ["resnet50"]
    
    import os
    os.system("rm -rf /home/lesliefang/pytorch_1_7_1/inductor_quant/torch_script/inductor/int8/pytorch_2_1_accuracy_test/torch_compile_debug/*")
    os.system("rm -rf /tmp/torchinductor_root/*")
    for model in model_list:
        run_model(model)
        # run_model_fx_x86_int8(model)
        # run_model_fp32(model)
        # run_model_ipex_int8(model)

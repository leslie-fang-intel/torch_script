# Execution Command: TORCHINDUCTOR_FREEZING=1 python x86inductorquantizer_qat_acc_all_models.py
import torch
import torchvision.models as models
import copy
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_qat_pt2e, prepare_pt2e
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import random
import numpy as np
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer

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

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))

    print("new lr is: {}".format(lr), flush=True)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def run_fine_tune_qat_model(model_name):
    print("start int8 pt2e QAT test of model: {}".format(model_name), flush=True)
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

    # traindir = "/home/dlboostbkc/dataset/Pytorch/train/"
    # training_traced_bs = traced_bs
    # train_loader = torch.utils.data.DataLoader(
    # datasets.ImageFolder(traindir, transforms.Compose([
    #     transforms.RandomResizedCrop(224),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     normalize,
    # ])),
    # batch_size=training_traced_bs,
    # shuffle=True,
    # num_workers=4,
    # pin_memory=True,
    # sampler=None)

    cal_loader = copy.deepcopy(val_loader)

    # model = models.__dict__[model_name](pretrained=True).train()

    pretrained=True

    model = models.__dict__[model_name](pretrained=pretrained)

    model = model.train()

    # print("model is: {}".format(model), flush=True)

    quant_qat_top1 = AverageMeter('Acc@1', ':6.2f')
    quant_qat_top5 = AverageMeter('Acc@5', ':6.2f')
    quant_top1 = AverageMeter('Acc@1', ':6.2f')
    quant_top5 = AverageMeter('Acc@5', ':6.2f')
    
    for i, (images, _) in enumerate(cal_loader):
        print("start to capture the graph", flush=True)
        images = images

        exported_model = capture_pre_autograd_graph(
            model,
            (images,),
        )
        break

    # Option 1: X86InductorQuantizer
    quantizer = X86InductorQuantizer()
    quantizer.set_global(xiq.get_default_x86_inductor_quantization_config(is_qat=True))
    # PT2E Quantization flow
    print("---- start prepare_qat_pt2e ----", flush=True)
    prepared_model = prepare_qat_pt2e(exported_model, quantizer)

    torch.ao.quantization.move_exported_model_to_eval(prepared_model)


    # ## Option 2: XNNPACKQuantizer
    # print("---- use XNNPACKQuantizer ----", flush=True)
    # quantizer = XNNPACKQuantizer()
    # quantizer.set_global(
    #     get_symmetric_quantization_config(
    #         is_per_channel=True, is_qat=True
    #     )
    # )
    # prepared_model = prepare_qat_pt2e(exported_model, quantizer)
 
    # if model_name == "mobilenet_v3_large":
    #     lr = 1e-4
    # else:
    #     lr = 0.1 if not pretrained else 0.0001
    total_epoch = 100 if not pretrained else 1 
    # momentum = 0.9
    # weight_decay = 1e-4
    # optimizer = torch.optim.SGD(prepared_model.parameters(), lr,
    #                             momentum=momentum,
    #                             weight_decay=weight_decay)
    # optimizer.zero_grad()
    # criterion = torch.nn.CrossEntropyLoss()

    # QAT
    if model_name == "mobilenet_v3_large":
        qat_training_step = 10
    else:
        qat_training_step = 1
    for epoch in range(total_epoch):
        print("start epoch: {}".format(epoch), flush=True)
        
        # adjust_learning_rate(optimizer, epoch, lr)
    
        for i, (images, target) in enumerate(cal_loader):
            # print(" start QAT Calibration step: {}".format(i), flush=True)
            # images = images
            # target = target
            output = prepared_model(images)
            # loss = criterion(output, target)
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
    
            # quant_qat_acc1, quant_qat_acc5 = accuracy(output, target, topk=(1, 5))
            # quant_qat_top1.update(quant_qat_acc1[0], images.size(0))
            # quant_qat_top5.update(quant_qat_acc5[0], images.size(0))
            # # if i % 9 == 0:
            # print('realtime step: {}, * Acc@1 {top1.val:.3f} Acc@5 {top5.val:.3f}'
            #     .format(i, top1=quant_qat_top1, top5=quant_qat_top5), flush=True)       
            # print('avg step: {}, * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            #     .format(i, top1=quant_qat_top1, top5=quant_qat_top5), flush=True)  
            
            if i == qat_training_step:
                break

    with torch.no_grad():
        # print("---- start convert_pt2e ----", flush=True)
        converted_model = convert_pt2e(prepared_model)
        torch.ao.quantization.move_exported_model_to_eval(converted_model)
        
        print("converted_model is: {}".format(converted_model), flush=True)
        
        # optimized_model = torch.compile(model)

        # model.eval()
        # torch.ao.quantization.move_exported_model_to_eval(prepared_model)
        optimized_model = converted_model

        # Benchmark
        for i, (images, target) in enumerate(val_loader):
            images = images.to(memory_format=torch.channels_last)

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
    print("Finish int8 pt2e QAT test of model: {}".format(model_name), flush=True)

def run_fine_tune_ptq_model(model_name):

    print("start int8 PTQ test of model: {}".format(model_name), flush=True)
    valdir = "/home/dlboostbkc/dataset/Pytorch/val/"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    traced_bs = 50
    # traced_bs = 1


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
    quant_top1 = AverageMeter('Acc@1', ':6.2f')
    quant_top5 = AverageMeter('Acc@5', ':6.2f')
    x = torch.randn(traced_bs, 3, 224, 224).contiguous(memory_format=torch.channels_last)
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
        # print("prepared_model is: {}".format(prepared_model), flush=True)
        # from torch.fx.passes.graph_drawer import FxGraphDrawer
        # g = FxGraphDrawer(prepared_model, "shuffnetv2")
        # g.get_dot_graph().write_svg("//home/lesliefang/pytorch_1_7_1/inductor_quant/torch_script/inductor/int8/pytorch_2_1_accuracy_test/new_frontend_shuffnetv2_prepare.svg")
        # Calibration
        for i, (images, _) in enumerate(cal_loader):
            prepared_model(images)
            if i==10: break
        converted_model = convert_pt2e(prepared_model)
        torch.ao.quantization.move_exported_model_to_eval(converted_model)
        print("converted_model is: {}".format(converted_model), flush=True)
        # Lower into Inductor

        optimized_model = converted_model
        # optimized_model = torch.compile(converted_model)

        # Benchmark
        for i, (images, target) in enumerate(val_loader):
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
    print("Finish int8 PTQ test of model: {}".format(model_name), flush=True)

if __name__ == "__main__":
    model_list=["alexnet",
                "shufflenet_v2_x1_0",
                "mobilenet_v3_large",
                "vgg16",
                "densenet121",
                "mnasnet1_0",
                "squeezenet1_1",
                "mobilenet_v2",
                "resnet50",
                "resnet152",
                "resnet18",
                "resnext50_32x4d"
    ]

    model_list=["mobilenet_v3_large",
    ]
    
    for model in model_list:
        run_fine_tune_qat_model(model)
        # run_fine_tune_ptq_model(model)


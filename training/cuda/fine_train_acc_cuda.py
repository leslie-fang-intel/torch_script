# Execution Command: python fine_train_acc_cuda.py
# Val accuracy after fine training:  * Acc@1 77.048 Acc@5 93.400 
import torch
import torchvision.models as models
import copy
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_qat_pt2e
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import random
import numpy as np
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)

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

def run_model_training(model_name):
    print("start training of model: {}".format(model_name), flush=True)
    valdir = "/home/t/leslie/imagenet/val/"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    traced_bs = 128
    val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=traced_bs, shuffle=False,
    num_workers=4, pin_memory=True)

    traindir = "/home/t/leslie/imagenet/train/"
    train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(traindir, transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=traced_bs,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    sampler=None)

    pretrained=True

    model = models.__dict__[model_name](pretrained=pretrained)
    model = model.cuda().train()

    quant_qat_top1 = AverageMeter('Acc@1', ':6.2f')
    quant_qat_top5 = AverageMeter('Acc@5', ':6.2f')
    quant_top1 = AverageMeter('Acc@1', ':6.2f')
    quant_top5 = AverageMeter('Acc@5', ':6.2f')

    lr = 0.1 if not pretrained else 0.0001
    momentum = 0.9
    weight_decay = 1e-4
    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)
    optimizer.zero_grad()
    criterion = torch.nn.CrossEntropyLoss().cuda()

    train_epoch = 100 if not pretrained else 1

    # Training
    for epoch in range(train_epoch):
        print("start epoch: {}".format(epoch), flush=True)
        
        adjust_learning_rate(optimizer, epoch, lr)

        for i, (images, target) in enumerate(train_loader):
            # print(" start QAT Calibration step: {}".format(i), flush=True)
            images = images.cuda()
            target = target.cuda()
            output = model(images)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            quant_qat_acc1, quant_qat_acc5 = accuracy(output, target, topk=(1, 5))
            quant_qat_top1.update(quant_qat_acc1[0], images.size(0))
            quant_qat_top5.update(quant_qat_acc5[0], images.size(0))
            # if i % 9 == 0:
            print('realtime step: {}, * Acc@1 {top1.val:.3f} Acc@5 {top5.val:.3f}'
                .format(i, top1=quant_qat_top1, top5=quant_qat_top5), flush=True)       
            print('avg step: {}, * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                .format(i, top1=quant_qat_top1, top5=quant_qat_top5), flush=True)  
        # if i==999: break

    print("Finish the Fine Training", flush=True)
    test_val_accuracy= True

    if test_val_accuracy:
        with torch.no_grad():
            optimized_model = model.eval()
            # Benchmark
            for i, (images, target) in enumerate(val_loader):
                images = images.to(memory_format=torch.channels_last).cuda()

                quant_output = optimized_model(images)

                quant_acc1, quant_acc5 = accuracy(quant_output, target.cuda(), topk=(1, 5))
                quant_top1.update(quant_acc1[0], images.size(0))
                quant_top5.update(quant_acc5[0], images.size(0))

                if i % 9 == 0:
                    print('step: {}, * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                        .format(i, top1=quant_top1, top5=quant_top5), flush=True)    

            print(model_name + " int8: ")
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                .format(top1=quant_top1, top5=quant_top5))
    print("Finish int8 pt2e QAT test of model: {}".format(model_name), flush=True)

if __name__ == "__main__":
    model_list = ["resnet50"]
    
    for model in model_list:
        run_model_training(model)


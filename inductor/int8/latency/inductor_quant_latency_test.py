# Execution Command:
# KMP_BLOCKTIME=1 KMP_AFFINITY="granularity=fine,compact,1,0" OMP_NUM_TRHEADS=4 TORCHINDUCTOR_FREEZING=1 numactl -C 0-27 -m 0 python inductor_quant_latency_test.py
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
import time
import threading

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

    # torch._dynamo.config.verbose = True
    # torch._inductor.config.trace.enabled = True
    # torch._inductor.config.trace.debug_log = True
    # torch._inductor.config.debug = True
    # torch._inductor.config.freezing = True

    print("start int8 test of model: {}".format(model_name), flush=True)
    traced_bs = 1
    model = models.__dict__[model_name](pretrained=True).eval()
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
        prepared_model(x)
        converted_model = convert_pt2e(prepared_model)
        torch.ao.quantization.move_exported_model_to_eval(converted_model)
        # Lower into Inductor
        optimized_model = torch.compile(converted_model)

        for _ in range(3):
            # Warm up run
            optimized_model(x)

        def run_weights_sharing_model(m, tid):
            # print("start thread {} for latency testing".format(tid), flush=True)
            steps = 300
            start_time = time.time()
            num_images = 0
            time_consume = 0
            timeBuff = []
            warmup_iterations = 50
            x = torch.randn(traced_bs, 3, 224, 224).contiguous(memory_format=torch.channels_last)
            with torch.no_grad():
                while num_images < steps:
                    print("start thread: {0} step: {1}".format(tid, num_images), flush=True)
                    start_time = time.time()
                    m(x)
                    end_time = time.time()
                    if num_images > warmup_iterations:
                        time_consume += end_time - start_time
                        timeBuff.append(end_time - start_time)
                    num_images += 1
                fps = (steps - warmup_iterations) / time_consume
                avg_time = time_consume * 1000 / (steps - warmup_iterations)
                timeBuff = np.asarray(timeBuff)
                p99 = np.percentile(timeBuff, 99)
                print('P99 Latency {:.2f} ms'.format(p99*1000))
                print('Instance num: %d Avg Time/Iteration: %f msec Throughput: %f fps' %(tid, avg_time, fps))            
        
        number_instance = 7
        threads = []
        for i in range(1, number_instance+1):
            thread = threading.Thread(target=run_weights_sharing_model, args=(optimized_model, i))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
    print("Finish int8 test of model: {}".format(model_name), flush=True)

if __name__ == "__main__":
    model_list=["alexnet","shufflenet_v2_x1_0","mobilenet_v3_large","vgg16","densenet121","mnasnet1_0","squeezenet1_1","mobilenet_v2","resnet50","resnet152","resnet18","resnext50_32x4d"]
    model_list = ["resnet50"]
    

    for model in model_list:
        run_model(model)


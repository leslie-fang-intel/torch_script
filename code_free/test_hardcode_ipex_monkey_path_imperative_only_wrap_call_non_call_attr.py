import torch
import intel_extension_for_pytorch as ipex
from typing import Callable
import functools

CodeFreeAutocastEnabled = False
## Option1: Wrap nn.Module __call__ function
def set_optimized_attr(model):
    model.optimized = True
    for child_name, child in model.named_children():
        set_optimized_attr(child)

# _orig_module_call: Callable = torch.nn.Module.__call__
_orig_module_init: Callable = torch.nn.Module.__init__

def wrap_function(cls, attr_name):
    _orig_module_function: Callable = getattr(cls, attr_name)
    @functools.wraps(_orig_module_function)
    def optimized_wrap_attr_function(mod, *args, **kwargs):
        def optimized_attr_function(mod, *args, **kwargs):
            if not hasattr(mod, "optimized"):
                set_optimized_attr(mod)
                dataType = torch.bfloat16 if (CodeFreeAutocastEnabled == True) else torch.float32
                optimized_m = ipex.optimize(mod.eval(), dtype=dataType).eval()

                set_optimized_attr(optimized_m)
                def optimized_m_forward(*args, **kwargs):
                    with torch.cpu.amp.autocast(enabled=CodeFreeAutocastEnabled), torch.no_grad():
                        return getattr(optimized_m, attr_name)(*args, **kwargs)
                if attr_name == "__call__":
                    setattr(mod, "forward", optimized_m_forward)
                else:
                    setattr(mod, attr_name, optimized_m_forward)
            return _orig_module_function(mod, *args, **kwargs)
        return optimized_attr_function(mod, *args, **kwargs)
    setattr(cls, attr_name, optimized_wrap_attr_function)

wrap_function(torch.nn.Module, "__call__")

@functools.wraps(_orig_module_init)
def module_init_wrapper(mod, *args, **kwargs):
    def init(mod, *args, **kwargs):
        if hasattr(mod.__class__, "forward") and getattr(mod.__class__, "forward") is not torch.nn.Module.forward:
            wrap_function(mod.__class__, "forward")
        if hasattr(mod.__class__, "inference"):
            wrap_function(mod.__class__, "inference")
        return _orig_module_init(mod, *args, **kwargs)
    return init(mod, *args, **kwargs)

setattr(torch.nn.Module, "__init__", module_init_wrapper)

class ConvBatchNorm(torch.nn.Module):
    name_t = "ttt"
    def __init__(self,):
        super(ConvBatchNorm, self).__init__()
        self.input1 = torch.randn(1, 3, 224, 224)
        self.conv = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.bn = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x):
        # y = self.bn(self.conv(x))
        # if torch.all(x.bool()):
        #     return y + 1.0
        return self.bn(self.conv(x))
        #return self.conv(x)

    # def inference(self, x):
    #     #import pdb;pdb.set_trace()
    #     return self.bn(self.conv(x))

import torch
import torch.fx.experimental.optimization as optimization
import torchvision.models as models
import time
import argparse

"""
export LD_PRELOAD="/pytorch/leslie/jemalloc/lib/libjemalloc.so":$LD_PRELOAD
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export LD_PRELOAD=$LD_PRELOAD:/pytorch/leslie/anaconda3/pkgs/intel-openmp-2021.4.0-h06a4308_3561/lib/libiomp5.so
KMP_BLOCKTIME=1 KMP_AFFINITY=granularity=fine,compact,1,0 OMP_NUM_THREADS=56 numactl -C 0-55 -m 0 python baseline.py --datatype int8
"""
parser = argparse.ArgumentParser(description='AI everywhere experiments')
parser.add_argument('--datatype', default='int8', help='path to dataset')

if __name__ == "__main__":
    args = parser.parse_args()

    if args.datatype == "bf16":
        CodeFreeAutocastEnabled = True

    iteration = 100
    batch_size = 56

    print("datatype is:{}".format(args.datatype))
    print("batch_size is:{}".format(batch_size))

    #model = models.__dict__["resnet50"](pretrained=True)
    model = ConvBatchNorm()

    if args.datatype == "fp32" or args.datatype == "bf16":
        model = model.to(memory_format=torch.channels_last).eval()

    x = torch.randn(batch_size, 3, 224, 224).contiguous(memory_format=torch.channels_last)

    if args.datatype == "fp32":
        # print(isinstance(model, torch.nn.Module))
        # print(type(model) is torch.nn.Module)
        # model = ipex.optimize(model, dtype=torch.float32, inplace=False)
        # print(isinstance(model, torch.nn.Module))
        # print(type(model) is torch.nn.Module)
        # print(model.__class__)
        # print(isinstance(model, (torch.nn.Module, torch.fx.GraphModule)))
        # print(isinstance(model, torch.fx.GraphModule))
        # exit(-1)

        # model = ipex.optimize(model, dtype=torch.float32)

        # traced_model = torch.jit.trace(model, x).eval()
        # #traced_model = torch.jit.freeze(traced_model)
        # print(type(traced_model))
        # exit(-1)
        # traced_model = model
        pass
    elif args.datatype == "bf16":
        # model = ipex.optimize(model, dtype=torch.bfloat16, inplace=True)
        # x = x.to(torch.bfloat16)
        # with torch.cpu.amp.autocast(), torch.no_grad():
        #     traced_model = torch.jit.trace(model, x).eval()
        # traced_model = torch.jit.freeze(traced_model)
        # traced_model = model
        pass
    else:
        print("unsupported data type", flush=True)
        exit(-1)

    with torch.no_grad():
        # warm up
        for i in range(3):
            print("warm step{}".format(i))
            # print("----id(model):{}".format(id(model)))
            # print("----id(model.forward):{}".format(id(model.forward)))
            model(x)
        print("finish warm up step")

        # print(getattr(model, "__call__"))
        # import pdb;pdb.set_trace()
        # model(x)
        # # getattr(model, "__call__")(x)
        # exit(-1)
        start = time.time()
        for i in range(iteration):
            # print("--i is:{}".format(i))
            if i == 29:
                with torch.profiler.profile(on_trace_ready=torch.profiler.tensorboard_trace_handler("./profile_log")) as prof:
                    model(x)
                print(prof.key_averages().table(sort_by="self_cpu_time_total"))
            else:
                model(x)
            # model.inference(x)
        end = time.time()
        print("time for one iteration is:{} ms".format((end-start)/iteration*1000))

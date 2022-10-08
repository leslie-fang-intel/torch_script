import torch
import intel_extension_for_pytorch as ipex
from typing import Callable
import functools
import inspect

# _orig_module_function: Callable = getattr(cls, attr_name)
# print("--------------{}".format(hasattr(torch.nn.Module, "__new__")))
# print(getattr(torch.nn.Module, "__new__"))
# print(getattr(object, "__new__"))
# exit(-1)

origin_new = getattr(torch.nn.Module, "__new__")

def wrap_new(cls, *args, **kwargs):
    print("----inside the new __new__----")
    obj = object.__new__(cls)
    print("obj.__class__ is:{}".format(obj.__class__), flush=True)
    isOutSideModel = True
    for frameinfo in inspect.stack():
        print(frameinfo.function)
        if frameinfo.function == "__init__":
            isOutSideModel = False

    insideIPEXOptimize = False
    for frameinfo in inspect.stack():
        print(frameinfo.function)
        if frameinfo.function == "optimize":
            insideIPEXOptimize = True

    if isOutSideModel and not insideIPEXOptimize:
        origin_class_init = getattr(obj.__class__, "__init__")
        def new_init_class(mod, *args, **kwargs):
            print("----inside the new new_init_class----")
            origin_class_init(mod, *args, **kwargs)
            mod = mod.eval()
            # **TODO** Here has problem: mod can't be replaced here
            mod = ipex.optimize(mod, dtype=torch.float32).eval()
            # Possible solution1: https://stackoverflow.com/questions/7940470/is-it-possible-to-overwrite-self-to-point-to-another-object-inside-self-method
            # optimized_self = ipex.optimize(self.eval(), dtype=torch.float32).eval()
            # self.__class__ = optimized_self.__class__
            # self.__dict__ = optimized_self.__dict__
            # It works, but model after ipex.optimize can't to channel_last: model = ipex.optimize(model.eval()).to(memory_format=torch.channels_last).eval()
            # Possible solution2: substitue the call, forward method of mod here
            print(hash(mod))
            print("----finish the init of outside module----", flush=True)
        setattr(obj.__class__, "__init__", new_init_class)
    return obj

setattr(torch.nn.Module, "__new__", wrap_new)

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

    model = models.__dict__["resnet50"](pretrained=True)
    if args.datatype == "fp32" or args.datatype == "bf16":
        model = model.to(memory_format=torch.channels_last).eval()

    x = torch.randn(batch_size, 3, 224, 224).contiguous(memory_format=torch.channels_last)

    if args.datatype == "fp32":
        # model = ipex.optimize(model, dtype=torch.float32)
        pass
    elif args.datatype == "bf16":
        # model = ipex.optimize(model, dtype=torch.bfloat16)
        pass
    else:
        print("unsupported data type", flush=True)
        exit(-1)

    with torch.no_grad():
        # warm up
        for i in range(3):
            # print("----warm up step: {}".format(i))
            model(x)

        print("hash(model) is:{}".format(hash(model)))
        start = time.time()
        for i in range(iteration):
            # print("----step: {}".format(i))
            if i == 29:
                with torch.profiler.profile(on_trace_ready=torch.profiler.tensorboard_trace_handler("./profile_log")) as prof:
                    model(x)
                print(prof.key_averages().table(sort_by="self_cpu_time_total"))
            else:
                model(x)
            # model(x)
        end = time.time()
        print("time for one iteration is:{} ms".format((end-start)/iteration*1000))

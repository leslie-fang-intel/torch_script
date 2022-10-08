import torch
import intel_extension_for_pytorch as ipex

## Option1: Substitute torch.nn.Module
# class myModule(torch.nn.Module):
#     def __init__(self) -> None:
#         print("----------inside myModule __init__-------", flush=True)
#         super().__init__()
#     def __call__(self, *input, **kwargs):
#         print("----------inside myModule __call__-------", flush=True)
#         super()._call_impl(*input, **kwargs)

# originModule = torch.nn.Module
# torch.nn.Module = myModule

# originModule = torch.nn.modules.module.Module
# torch.nn.modules.module.Module = myModule
# exit(-1)

## Option2: Global hook in forward
def set_optimized_attr(model):
    model.optimized = True
    for child_name, child in model.named_children():
        set_optimized_attr(child)

def forward_pre_hook(m, input):
    if not hasattr(m, "optimized"):
        # print("--------inside the prehook----------")

        # Step1:
        # Symbolic trace inside ipex.optimize will invoke forward method of submodules
        # Add optimized attr to all the submodule to avoid recursion invoking of ipex.optimize in submodules
        set_optimized_attr(m)

        # Step2:
        # Invoke optimize to generate new optimized module
        OptimizedLevel = "O1"
        ConvBNFolding = True
        WeightPrepack = True
        AutoKernelSelection = True
        dataType = torch.bfloat16 if (ipex.CodeFreeAutocastEnabled == True) else torch.float32
        optimized_m = ipex.optimize(m.eval(), dtype=dataType, level=OptimizedLevel, conv_bn_folding=ConvBNFolding, weights_prepack=WeightPrepack, auto_kernel_selection=AutoKernelSelection, inplace=True).eval()
        # m = ipex.optimize(m.eval(), dtype=torch.float32, inplace=True).eval()

        # Set optimized attr for the new optimized module
        optimized_m.optimized = True

        if ConvBNFolding:
            # Step3:
            # Substitue the forward method of m with forward method of new optimized module
            def optimized_m_forward(*args, **kwargs):
                return optimized_m(*args, **kwargs)
            m.forward = optimized_m_forward

            # Step4:
            # To support Jit trace:
            # Modules that are called during a trace must be registered as submodules of the thing being traced.
            # Register the submodules of new optimized module into m
            for child_name, child in optimized_m.named_children():
                m.add_module("optimized_"+child_name, child)

        # For Autocast Imperative Path: will Enable Autocast Globally
        if ipex.CodeFreeAutocastEnabled and ipex.CodeFreeImperativePath:
            torch.set_autocast_cpu_enabled(True)
            ipex._C.set_autocast_dtype(torch.bfloat16)
            torch.autocast_increment_nesting()
            torch.set_autocast_cache_enabled(True)

    return input

handle = torch.nn.modules.module.register_module_forward_pre_hook(forward_pre_hook)
ipex.CodeFreeEnabled = True
ipex.set_global_forward_prehook_handle(handle)

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
parser.add_argument("--jit", action='store_true', default=False,
                    help="Enable jit path")

if __name__ == "__main__":
    args = parser.parse_args()

    if args.datatype == "bf16":
        ipex.CodeFreeAutocastEnabled = True
    
    if args.jit:
        ipex.CodeFreeJITPath = True
    else:
        ipex.CodeFreeImperativePath = True

    iteration = 100
    batch_size = 56

    print("datatype is:{}".format(args.datatype))
    print("batch_size is:{}".format(batch_size))

    model = models.__dict__["resnet50"](pretrained=True)
    if args.datatype == "fp32" or args.datatype == "bf16":
        model = model.to(memory_format=torch.channels_last).eval()

    x = torch.randn(batch_size, 3, 224, 224).contiguous(memory_format=torch.channels_last)

    if args.datatype == "fp32":
        #model = ipex.optimize(model, dtype=torch.float32, inplace=True)
        #model = ipex.optimize(model, dtype=torch.float32, auto_kernel_selection=True, inplace=True)
        #traced_model = torch.jit.trace(model, x).eval()
        #traced_model = torch.jit.freeze(traced_model)
        if args.jit:
            traced_model = torch.jit.trace(model, x).eval()
            traced_model = torch.jit.freeze(traced_model)
        else:
            traced_model = model
    elif args.datatype == "bf16":
        #model = ipex.optimize(model, dtype=torch.bfloat16, inplace=True)
        # x = x.to(torch.bfloat16)
        # with torch.cpu.amp.autocast(), torch.no_grad():
        #     traced_model = torch.jit.trace(model, x).eval()
        # traced_model = torch.jit.freeze(traced_model)
        if args.jit:
            traced_model = torch.jit.trace(model, x).eval()
            traced_model = torch.jit.freeze(traced_model)
        else:
            traced_model = model
    else:
        print("unsupported data type", flush=True)
        exit(-1)

    with torch.no_grad():
        # warm up
        for i in range(3):
            traced_model(x)
        if args.jit:
            print(traced_model.graph_for(x))

        start = time.time()
        for i in range(iteration):
            if i == 29:
                with torch.profiler.profile(on_trace_ready=torch.profiler.tensorboard_trace_handler("./fp32_log")) as prof:
                    traced_model(x)
                print(prof.key_averages().table(sort_by="self_cpu_time_total"))
            else:
                traced_model(x)
        end = time.time()
        print("time for one iteration is:{} ms".format((end-start)/iteration*1000))

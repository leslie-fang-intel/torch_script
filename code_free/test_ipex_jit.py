import torch
import torch.fx.experimental.optimization as optimization
import intel_extension_for_pytorch as ipex
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

    iteration = 100
    batch_size = 56

    print("datatype is:{}".format(args.datatype))
    print("batch_size is:{}".format(batch_size))

    model = models.__dict__["resnet50"](pretrained=True)
    if args.datatype == "fp32" or args.datatype == "bf16":
        model = model.to(memory_format=torch.channels_last).eval()

    x = torch.randn(batch_size, 3, 224, 224).contiguous(memory_format=torch.channels_last)
    if args.datatype == "fp32":
        model = ipex.optimize(model, dtype=torch.float32, auto_kernel_selection=True, inplace=True)
        traced_model = torch.jit.trace(model, x).eval()
        traced_model = torch.jit.freeze(traced_model)
    elif args.datatype == "bf16":
        model = ipex.optimize(model, dtype=torch.bfloat16, auto_kernel_selection=True, inplace=True)
        x = x.to(torch.bfloat16)
        with torch.cpu.amp.autocast(), torch.no_grad():
            traced_model = torch.jit.trace(model, x).eval()
        traced_model = torch.jit.freeze(traced_model)
    else:
        print("unsupported data type", flush=True)
        exit(-1)

    with torch.no_grad():
        # warm up
        for i in range(3):
            traced_model(x)
        print(traced_model.graph_for(x))

        start = time.time()
        for i in range(iteration):
            traced_model(x)
        end = time.time()
        print("time for one iteration is:{} ms".format((end-start)/iteration*1000))
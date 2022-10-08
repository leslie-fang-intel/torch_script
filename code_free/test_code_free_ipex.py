import torch
import torch.fx.experimental.optimization as optimization
import torchvision.models as models
import time
import argparse
import sys

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

    model = models.__dict__["resnet50"](pretrained=True)
    model = model.to(memory_format=torch.channels_last).eval()

    x = torch.randn(batch_size, 3, 224, 224).contiguous(memory_format=torch.channels_last)

    with torch.no_grad():
        # warm up
        for i in range(3):
            model(x)

        start = time.time()
        for i in range(iteration):
            if i == 29:
                with torch.profiler.profile(on_trace_ready=torch.profiler.tensorboard_trace_handler("./fp32_log")) as prof:
                    model(x)
                print(prof.key_averages().table(sort_by="self_cpu_time_total"))
            else:
                model(x)
        end = time.time()
        print("time for one iteration is:{} ms".format((end-start)/iteration*1000))
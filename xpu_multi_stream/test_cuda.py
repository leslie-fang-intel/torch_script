import torch
import intel_extension_for_pytorch as ipex
import torchvision.models as models
import copy
import time

USE_CUDA = True

CUDA_ONLY = False
CPU_ONLY = False
USE_XPU = True

CUDA_TASK = True

def measurement_performance():
    bs_cpu = 14
    bs_gpu = 60
    global_bs = bs_cpu + bs_gpu
    x = torch.randn(global_bs, 3, 224, 224).contiguous(memory_format=torch.channels_last)
    model = models.__dict__["resnet50"]().to(memory_format=torch.channels_last).eval()

    x_cpu = x[0:bs_cpu]
    # **TODO** copy.deepcopy may fail for some models
    model_cpu = copy.deepcopy(model)

    if USE_CUDA:
        x_gpu = x[bs_cpu:global_bs].to(device="cuda")
        model_gpu = model.to(device="cuda")
        if CUDA_TASK:
            cpu_pool = ipex.cpu.runtime.CPUPool(core_ids=[0,])
            #cpu_pool = ipex.cpu.runtime.CPUPool(node_id=0)
            model_gpu = ipex.cpu.runtime.Task(model_gpu, cpu_pool)

    warm_up_iterations = 20
    for i in range(warm_up_iterations):
        if CUDA_ONLY or USE_XPU:
            res_gpu = model_gpu(x_gpu)
        if CPU_ONLY or USE_XPU:
            res_cpu = model_cpu(x_cpu)
        if CUDA_TASK:
            res_gpu = res_gpu.get()

    iterations = 100
    start = time.time()
    for i in range(iterations):
        if CUDA_ONLY or USE_XPU:
            res_gpu = model_gpu(x_gpu)
        if CPU_ONLY or USE_XPU:
            res_cpu = model_cpu(x_cpu)
        if CUDA_TASK:
            res_gpu = res_gpu.get()
    time_cost = (time.time() - start)

    print("time_cost: {}".format(time_cost))
    if CUDA_ONLY:
        throughtput = bs_gpu * iterations / time_cost
    elif CPU_ONLY:
        throughtput = bs_cpu * iterations / time_cost
    else:
        throughtput = (bs_cpu + bs_gpu) * iterations / time_cost
    print("throughtput: {}".format(throughtput))
    return

def test_cuda_task():
    x = torch.randn(1000, 3, 224, 224).contiguous(memory_format=torch.channels_last)
    model = models.__dict__["resnet50"]().to(memory_format=torch.channels_last).eval()
    x_gpu = x[14:28].to(device="cuda")
    model_gpu = model.to(device="cuda")
    # cpu_pool = ipex.cpu.runtime.CPUPool(node_id=0)
    cpu_pool = ipex.cpu.runtime.CPUPool(core_ids=[0,])
    task = ipex.cpu.runtime.Task(model_gpu, cpu_pool)
    y_runtime_future = task(x_gpu)
    y_runtime = y_runtime_future.get()
    print(y_runtime)

    y_res = model_gpu(x_gpu)
    print(torch.allclose(y_res, y_runtime))

if __name__ == "__main__":
    measurement_performance()
    #test_cuda_task()


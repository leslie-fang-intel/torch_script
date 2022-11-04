import torch
import intel_extension_for_pytorch as ipex
import torchvision.models as models
import copy
import time
import opentuner
from opentuner import ConfigurationManipulator
from opentuner import IntegerParameter, EnumParameter
from opentuner import MeasurementInterface
from opentuner import Result
import argparse
from opentuner.api import TuningRunManager
from opentuner.measurement.interface import DefaultMeasurementInterface

USE_CUDA = True

CUDA_ONLY = True
CPU_ONLY = False
USE_XPU = False

CUDA_TASK = False


model_gpu = None
model_cpu = None

best_throughtput = 0.0
best_cpu_bs = None
best_gpu_bs = None

def single_measurement(x_gpu, x_cpu, bs_gpu, bs_cpu):
    global model_gpu
    global model_cpu
    with torch.no_grad():
        warm_up_iterations = 20
        for i in range(warm_up_iterations):
            if CUDA_ONLY or USE_XPU:
                res_gpu = model_gpu(x_gpu)
            if CPU_ONLY or USE_XPU:
                res_cpu = model_cpu(x_cpu)
            if CUDA_TASK:
                res_gpu = res_gpu.get()

        iterations = 50
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
        return throughtput

def measurement_performance(cfg):
    global model_gpu
    global model_cpu

    bs_cpu = int(cfg['bs_cpu'])
    bs_gpu = int(cfg['bs_gpu'])
    global_bs = bs_cpu + bs_gpu
    x = torch.randn(global_bs, 3, 224, 224).contiguous(memory_format=torch.channels_last)

    print("bs_cpu:{0}, bs_gpu:{1}".format(bs_cpu, bs_gpu))
    x_cpu = x[0:bs_cpu]
    print(x_cpu.size())

    model_gpu = None
    x_gpu = None
    if USE_CUDA:
        x_gpu = x[bs_cpu:global_bs].to(device="cuda")
        print(x_gpu.size())
    throught = single_measurement(x_gpu, x_cpu, bs_gpu, bs_cpu)

    global best_throughtput
    global best_cpu_bs
    global best_gpu_bs
    if throught > best_throughtput:
        best_cpu_bs = bs_cpu
        best_gpu_bs = bs_gpu
        best_throughtput = throught
        print("Update best result, best_gpu_bs:{0}, best_cpu_bs:{1}, best_throughtput:{2}".format(best_gpu_bs, best_cpu_bs, best_throughtput))
    return 1.0 / throught


if __name__ == "__main__":
    model = models.__dict__["resnet50"]().to(memory_format=torch.channels_last).eval()
    # **TODO** copy.deepcopy may fail for some models
    model_cpu = copy.deepcopy(model)
    if USE_CUDA:
        model_gpu = model.to(device="cuda")
        if CUDA_TASK:
            cpu_pool = ipex.cpu.runtime.CPUPool(core_ids=[0,])
            #cpu_pool = ipex.cpu.runtime.CPUPool(node_id=0)
            model_gpu = ipex.cpu.runtime.Task(model_gpu, cpu_pool)

    parser = argparse.ArgumentParser(parents=opentuner.argparsers())
    args = parser.parse_args()
    manipulator = ConfigurationManipulator()
    manipulator.add_parameter(
        IntegerParameter('bs_cpu', 1, 1000)) # batchSize = streamNumber * batchSize_uniform
    manipulator.add_parameter(
        IntegerParameter('bs_gpu', 1, 1000))
    interface = DefaultMeasurementInterface(args=args,
                                            manipulator=manipulator,
                                            project_name='CUDA',
                                            program_name='rn50_fp32',
                                            program_version='0.1')
    api = TuningRunManager(interface, args)
    for x in range(1000):
        desired_result = api.get_next_desired_result()
        if desired_result is None:
            # The search space for this example is very small, so sometimes
            # the techniques have trouble finding a config that hasn't already
            # been tested.  Change this to a continue to make it try again.
            break
        cfg = desired_result.configuration.data
        result = Result(time=measurement_performance(cfg))
        api.report_result(desired_result, result)


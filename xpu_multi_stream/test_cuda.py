import torch
import torchvision.models as models
import copy
import time
import torch.fx.experimental.optimization as optimization
import os

CUDA_ONLY = False
CPU_ONLY = False
USE_XPU = True

ASYNC_TASK = True
USE_JIT = True

if ASYNC_TASK or USE_JIT:
    import intel_extension_for_pytorch as ipex

PROFILE = False

warm_up_iterations = 20
iterations = 50
bs_cpu = 126
bs_gpu = 1536

def measurement_xpu_performance():
    global bs_cpu
    global bs_gpu
    global_bs = bs_cpu + bs_gpu
    x = torch.randn(global_bs, 3, 224, 224).contiguous(memory_format=torch.channels_last)
    model = models.__dict__["resnet50"]().to(memory_format=torch.channels_last).eval()

    x_cpu = x[0:bs_cpu]
    # **TODO** copy.deepcopy may fail for some models
    model_cpu = copy.deepcopy(model)

    numa_node0_cores = ipex.cpu.runtime.get_core_list_of_node_id(0)
    print("numa_node0_cores is: {}".format(numa_node0_cores))

    cuda_thread_coreid = 0
    numa_node0_cores.remove(cuda_thread_coreid)
    cpu_computation_cores = numa_node0_cores

    if USE_JIT and (CPU_ONLY or USE_XPU):
        if ASYNC_TASK:
            traced_cpu_pool = ipex.cpu.runtime.CPUPool([0])
            num_streams = cpu_computation_cores.__len__()
            with torch.no_grad(), ipex.cpu.runtime.pin(traced_cpu_pool):
                x_traced = x_cpu[0:(bs_cpu//num_streams)]
                with torch.no_grad():
                    model_cpu = torch.jit.trace(model_cpu, x_traced, check_trace=False).eval()
                model_cpu = torch.jit.freeze(model_cpu)
                for _ in range(3):
                    model_cpu(x_traced)
                print("Finish CPU Jit Warmup", flush=True)
            cpu_pool = ipex.cpu.runtime.CPUPool(core_ids=cpu_computation_cores)
            model_cpu = ipex.cpu.runtime.MultiStreamModule(model_cpu, num_streams=num_streams, cpu_pool=cpu_pool)
        else:
            with torch.no_grad():
                model_cpu = torch.jit.trace(model_cpu, x_cpu, check_trace=False).eval()
            model_cpu = torch.jit.freeze(model_cpu)
            with torch.no_grad():
                for _ in range(3):
                    model_cpu(x_cpu)
                print("Finish CPU Jit Warmup", flush=True)

    if CUDA_ONLY or USE_XPU:
        x_gpu = x[bs_cpu:global_bs].to(device="cuda")
        model_gpu = model.to(device="cuda")
        ipex._C.disable_jit_opt()
        if USE_JIT:
            with torch.no_grad():
                model_gpu = torch.jit.trace(model_gpu, x_gpu, check_trace=False).eval().to(device="cuda")
            model_gpu = torch.jit.freeze(model_gpu)
            with torch.no_grad():
                for _ in range(3):
                    model_gpu(x_gpu)
                print("Finish GPU Jit Warmup", flush=True)
        ipex._C.enable_jit_opt()
        if ASYNC_TASK:
            cpu_pool = ipex.cpu.runtime.CPUPool(core_ids=[cuda_thread_coreid,])
            #cpu_pool = ipex.cpu.runtime.CPUPool(node_id=0)
            model_gpu = ipex.cpu.runtime.Task(model_gpu, cpu_pool)
            #pass

    affinity_mask = set(cpu_computation_cores)
    os.sched_setaffinity(0, affinity_mask)
    main_thread_cpupool = ipex.cpu.runtime.CPUPool(cpu_computation_cores)
    with torch.no_grad(), ipex.cpu.runtime.pin(main_thread_cpupool):
        global warm_up_iterations
        for i in range(warm_up_iterations):
            if CUDA_ONLY or USE_XPU:
                res_gpu = model_gpu(x_gpu)
            if CPU_ONLY or USE_XPU:
                res_cpu = model_cpu(x_cpu)
            if (CUDA_ONLY or USE_XPU) and ASYNC_TASK:
                res_gpu = res_gpu.get()
                #pass
        print("Finish the warm up", flush=True)

        global iterations
        start = time.time()
        for i in range(iterations):
            if CUDA_ONLY or USE_XPU:
                if i == 20 and PROFILE:
                    with torch.profiler.profile(on_trace_ready=torch.profiler.tensorboard_trace_handler("./int8_log")) as prof:
                        res_gpu = model_gpu(x_gpu)
                    print(prof.key_averages().table(sort_by="self_cpu_time_total"))
                else:
                    res_gpu = model_gpu(x_gpu)
            if CPU_ONLY or USE_XPU:
                res_cpu = model_cpu(x_cpu)
            if (CUDA_ONLY or USE_XPU) and ASYNC_TASK:
                res_gpu = res_gpu.get()
                #pass
            res_gpu_cpu = res_gpu.to(device="cpu")
            res = torch.cat((res_cpu, res_gpu_cpu))
        time_cost = (time.time() - start)

        print("time_cost: {}".format(time_cost))
        print("Time per iteration: {}".format(time_cost/iterations))
        if CUDA_ONLY:
            throughtput = bs_gpu * iterations / time_cost
        elif CPU_ONLY:
            throughtput = bs_cpu * iterations / time_cost
        else:
            throughtput = (bs_cpu + bs_gpu) * iterations / time_cost
        print("throughtput: {}".format(throughtput))
        return

def measurement_performance_cpu(use_async_task=False, use_jit=False):
    global bs_cpu
    global_bs = bs_cpu
    x = torch.randn(global_bs, 3, 224, 224).contiguous(memory_format=torch.channels_last)
    model = models.__dict__["resnet50"]().to(memory_format=torch.channels_last).eval()

    x_cpu = x
    model_cpu = model

    numa_node0_cores = ipex.cpu.runtime.get_core_list_of_node_id(0)
    print("numa_node0_cores is: {}".format(numa_node0_cores))

    cpu_computation_cores = numa_node0_cores

    if use_jit:
        if ASYNC_TASK:
            traced_cpu_pool = ipex.cpu.runtime.CPUPool([0])
            num_streams=cpu_computation_cores.__len__()
            with torch.no_grad(), ipex.cpu.runtime.pin(traced_cpu_pool):
                traced_X = x_cpu[0:(bs_cpu//num_streams)]
                with torch.no_grad():
                    model_cpu = torch.jit.trace(model_cpu, traced_X, check_trace=False).eval()
                model_cpu = torch.jit.freeze(model_cpu)
                for _ in range(3):
                    model_cpu(traced_X)
                print("Finish Async CPU Jit Warmup", flush=True)
            cpu_pool = ipex.cpu.runtime.CPUPool(core_ids=cpu_computation_cores)
            model_cpu = ipex.cpu.runtime.MultiStreamModule(model_cpu, num_streams=num_streams, cpu_pool=cpu_pool)
        else:
            with torch.no_grad():
                model_cpu = torch.jit.trace(model_cpu, x_cpu, check_trace=False).eval()
            model_cpu = torch.jit.freeze(model_cpu)
            with torch.no_grad():
                for _ in range(3):
                    model_cpu(x_cpu)
                print("Finish CPU Jit Warmup", flush=True)

    affinity_mask = set(cpu_computation_cores)
    os.sched_setaffinity(0, affinity_mask)
    main_thread_cpupool = ipex.cpu.runtime.CPUPool(cpu_computation_cores)
    with torch.no_grad(), ipex.cpu.runtime.pin(main_thread_cpupool):
        global warm_up_iterations
        for i in range(warm_up_iterations):
            res_cpu = model_cpu(x_cpu)
        print("Finish the warm up", flush=True)

        global iterations
        start = time.time()
        for i in range(iterations):
            res_cpu = model_cpu(x_cpu)
        time_cost = (time.time() - start)

        print("time_cost: {}".format(time_cost))
        print("Time per iteration: {}".format(time_cost/iterations))
        throughtput = bs_cpu * iterations / time_cost
        print("throughtput: {}".format(throughtput))
        return


def measurement_performance_raw_cuda(use_async_task=False, use_jit=False):
    global bs_gpu
    x = torch.randn(bs_gpu, 3, 224, 224).contiguous(memory_format=torch.channels_last)
    model = models.__dict__["resnet50"]().to(memory_format=torch.channels_last).eval()

    if use_jit:
        ipex._C.disable_jit_opt()
        with torch.no_grad():
            model_gpu = torch.jit.trace(model, x, check_trace=False).eval().to(device="cuda")
        model_gpu = torch.jit.freeze(model_gpu)
        x_gpu = x.to(device="cuda")
        with torch.no_grad():
            for _ in range(3):
                res_gpu = model_gpu(x_gpu)
                torch.cuda.synchronize(device="cuda")
            print("Finish Jit Warmup")
        ipex._C.enable_jit_opt()
    else:
        model_gpu = model.to(device="cuda").eval()
        x_gpu = x.to(device="cuda")

    if use_async_task:
        cpu_pool = ipex.cpu.runtime.CPUPool(core_ids=[0,])
        model_gpu = ipex.cpu.runtime.Task(model_gpu, cpu_pool)

    with torch.no_grad():
        global warm_up_iterations
        for i in range(warm_up_iterations):
            res_gpu = model_gpu(x_gpu)
            if use_async_task:
                res_gpu.get()
            torch.cuda.synchronize(device="cuda")
        print("Finish the warm up")

        global iterations
        start = time.time()
        for i in range(iterations):
            if i == 20 and PROFILE:
                with torch.profiler.profile(on_trace_ready=torch.profiler.tensorboard_trace_handler("./int8_log")) as prof:
                    res_gpu = model_gpu(x_gpu)
                print(prof.key_averages().table(sort_by="self_cpu_time_total"))
            else:
                res_gpu = model_gpu(x_gpu)
            if use_async_task:
                res_gpu.get()
            torch.cuda.synchronize(device="cuda")
        time_cost = (time.time() - start)

        print("time_cost: {}".format(time_cost))
        print("Time per iteration: {}".format(time_cost/iterations))
        throughtput = bs_gpu * iterations / time_cost
        print("throughtput: {}".format(throughtput))
    return

def test_cuda_task_correctness():
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
    #measurement_xpu_performance()

    if CUDA_ONLY:
        print("CUDA_ONLY:{}".format(CUDA_ONLY))
        if ASYNC_TASK and USE_JIT:
            print("ASYNC_TASK:{0}, USE_JIT:{1}".format(ASYNC_TASK, USE_JIT))
            measurement_performance_raw_cuda(True, True)
        elif ASYNC_TASK:
            print("ASYNC_TASK:{0}, USE_JIT:{1}".format(ASYNC_TASK, USE_JIT))
            measurement_performance_raw_cuda(True, False)
        elif USE_JIT:
            print("ASYNC_TASK:{0}, USE_JIT:{1}".format(ASYNC_TASK, USE_JIT))
            measurement_performance_raw_cuda(False, True)
        else:
            print("ASYNC_TASK:{0}, USE_JIT:{1}".format(ASYNC_TASK, USE_JIT))
            measurement_performance_raw_cuda(False, False)
    elif CPU_ONLY:
        print("CPU_ONLY:{}".format(CPU_ONLY))
        if ASYNC_TASK and USE_JIT:
            print("ASYNC_TASK:{0}, USE_JIT:{1}".format(ASYNC_TASK, USE_JIT))
            measurement_performance_cpu(True, True)
        elif ASYNC_TASK:
            print("ASYNC_TASK:{0}, USE_JIT:{1}".format(ASYNC_TASK, USE_JIT))
            measurement_performance_cpu(True, False)
        elif USE_JIT:
            print("ASYNC_TASK:{0}, USE_JIT:{1}".format(ASYNC_TASK, USE_JIT))
            measurement_performance_cpu(False, True)
        else:
            print("ASYNC_TASK:{0}, USE_JIT:{1}".format(ASYNC_TASK, USE_JIT))
            measurement_performance_cpu(False, False)
    else:
        measurement_xpu_performance()

    # test_cuda_task_correctness()

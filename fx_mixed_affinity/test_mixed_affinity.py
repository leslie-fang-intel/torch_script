import torch
import torchvision.models as models
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.fx.xpu_auto_shard import TaggingTracer, init_node_annotator, mixed_affinity_search_pass, sharding_lowering_pass, heuristic_annotator
import copy
import time
import sys

use_bf16 = True
use_multi_stream = True
profile = False
if __name__ == "__main__":
    module = models.__dict__["resnet50"]().to(memory_format=torch.channels_last).eval()
    graph = TaggingTracer().trace(module)
    graph_module = torch.fx.GraphModule(module, graph).eval()
    graph_module = ipex.optimize(graph_module, dtype=torch.bfloat16 if use_bf16 else torch.float32, inplace=True)
    graph_module = init_node_annotator(graph_module)

    config = {
            "start_node_index" : 0,
            "find_end_node": True
        }

    search_times = 100
    best_cost = sys.maxsize
    while search_times > 0:
        config, status = heuristic_annotator(graph_module, config=copy.deepcopy(config))
        start_node_idx = config["cur_start_index"]
        end_node_idx = config["cur_end_index"]

        # **TODO** opentuner parameters
        loop_times = config["loop_times"]
        mini_bs = config["mini_bs"]

        ## HardCode best parameters
        # start_node_idx = 0
        # end_node_idx = 54
        # loop_times = 3
        # mini_bs = 3

        batch_size = 126
        x = torch.randn(batch_size, 3, 224, 224).contiguous(memory_format=torch.channels_last)
        if use_bf16:
            x = x.to(torch.bfloat16)

        if use_multi_stream:
            num_streams = 14
        print("search_times is: {}".format(search_times), flush=True)
        print("config is: {}".format(config), flush=True)
        graph_module_ = mixed_affinity_search_pass(graph_module, start_node_idx=start_node_idx, end_node_idx=end_node_idx, loop_times=loop_times, mini_bs=mini_bs)
        mixed_affinity_module = sharding_lowering_pass(graph_module_)

        with torch.no_grad():
            if use_multi_stream:
                core_list = ipex.cpu.runtime.get_core_list_of_node_id(0)
                core_number = core_list.__len__()
                traced_cpu_pool = ipex._C.CPUPool(core_list[0:(core_number//num_streams)])
                ipex._C.pin_cpu_cores(traced_cpu_pool)
            traced_x = x[0:(batch_size//num_streams)] if use_multi_stream else x
            with torch.cpu.amp.autocast(dtype=torch.bfloat16):
                traced_model = torch.jit.trace(mixed_affinity_module, traced_x).eval()
            traced_model = torch.jit.freeze(traced_model)
            # Running warm up steps
            for _ in range(3):
                traced_model(traced_x)
            test_model = traced_model

            if use_multi_stream:
                cpu_pool = ipex.cpu.runtime.CPUPool(node_id=0)
                multi_stream_model = ipex.cpu.runtime.MultiStreamModule(traced_model, num_streams=num_streams, cpu_pool=cpu_pool)
                test_model = multi_stream_model

            # Warm_up run
            for _ in range(50):
                result = test_model(x)
            iterations = 150
            # Formal Run
            elapse_cost = 0.0
            for step in range(iterations):
                start_time = time.time()
                if step == 99 and profile:
                    with torch.profiler.profile(on_trace_ready=torch.profiler.tensorboard_trace_handler("./int8_log")) as prof:
                        result = test_model(x)
                    print(prof.key_averages().table(sort_by="self_cpu_time_total"))
                else:
                    result = test_model(x)
                elapse_cost += (time.time() - start_time)
            print("elapse_cost is: {} second".format(elapse_cost), flush=True)
            print("Time per step is: {} second".format(elapse_cost / iterations), flush=True)
            current_throughput = (batch_size * iterations) / elapse_cost
            print("Throughput is: {} fps".format(current_throughput), flush=True)

            if elapse_cost < best_cost:
                best_cost = elapse_cost
                best_config = config
                best_throughput = current_throughput
                print("Update best config is: {0}. best_throughput: {1}".format(best_config, best_throughput), flush=True)
            #break
            search_times -= 1
    print("Finish best config is: {0}. best_throughput: {1}".format(best_config, best_throughput), flush=True)



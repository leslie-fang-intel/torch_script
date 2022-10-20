import torch
import torch.fx.experimental.optimization as optimization
import intel_extension_for_pytorch as ipex
import torchvision.models as models

from intel_extension_for_pytorch.fx.xpu_auto_shard import TaggingTracer, init_node_annotator, mixed_affinity_search_pass, sharding_lowering_pass, heuristic_annotator
import torchvision.models as models
import copy
import time
import sys

use_bf16 = True
use_multi_stream = True
if __name__ == "__main__":
    module = models.__dict__["resnet50"](pretrained=True).eval()
    graph = TaggingTracer().trace(module)
    graph_module = torch.fx.GraphModule(module, graph).eval()
    graph_module = ipex.optimize(graph_module, dtype=torch.bfloat16)

    graph_module = init_node_annotator(graph_module)

    config = {
            "start_node_index" : 0,
            "find_end_node": True
        }
    x = torch.randn(4, 3, 224, 224)
    if use_bf16:
        x = x.to(torch.bfloat16)

    search_times = 100
    best_cost = sys.maxsize
    while search_times > 0:
        config, status = heuristic_annotator(graph_module, config=copy.deepcopy(config))
        start_node_idx = config["start_node_index"]
        end_node_idx = config["end_node_index"]

        # **TODO** opentuner parameters
        loop_times = config["loop_times"]
        mini_bs = config["mini_bs"]
        if use_multi_stream:
            num_streams = 2
        print("search_times is: {}".format(search_times), flush=True)
        print("config is: {}".format(config), flush=True)
        graph_module_ = mixed_affinity_search_pass(graph_module, start_node_idx=start_node_idx, end_node_idx=end_node_idx, loop_times=2, mini_bs=None)
        mixed_affinity_module = sharding_lowering_pass(graph_module_)

        with torch.no_grad():
            if use_multi_stream:
                core_list = ipex.cpu.runtime.get_core_list_of_node_id(0)
                core_number = core_list.__len__()
                traced_cpu_pool = ipex._C.CPUPool(core_list[0:(core_number//num_streams)])
                ipex._C.pin_cpu_cores(traced_cpu_pool)
            with torch.cpu.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16):
                traced_model = torch.jit.trace(mixed_affinity_module, x)
            torch.jit.freeze(traced_model)
            for _ in range(3):
                traced_model(x)
            test_model = traced_model

            if use_multi_stream:
                cpu_pool = ipex.cpu.runtime.CPUPool(node_id=0)
                multi_stream_model = ipex.cpu.runtime.MultiStreamModule(traced_model, num_streams=num_streams, cpu_pool=cpu_pool)
                test_model = multi_stream_model

            # Warm_up run
            for _ in range(20):
                result = test_model(x)
            # Formal Run
            start_time = time.perf_counter()
            for _ in range(50):
                result = test_model(x)
            elapse_cost = time.perf_counter() - start_time
            print("elapse_cost is: {}".format(elapse_cost), flush=True)
            if elapse_cost < best_cost:
                best_cost = elapse_cost
                best_config = config
                print("Update best config is: ", best_config, flush=True)
        search_times -= 1
    print("Finish best config is: ", best_config, flush=True)



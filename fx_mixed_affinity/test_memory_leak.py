import torch
import torch.fx.experimental.optimization as optimization
import intel_extension_for_pytorch as ipex
import torchvision.models as models
from intel_extension_for_pytorch.fx.xpu_auto_shard import TaggingTracer, init_node_annotator, mixed_affinity_search_pass, sharding_lowering_pass, heuristic_annotator, clear_trunk_node_name_set
from intel_extension_for_pytorch.cpu.runtime.multi_stream import MultiStreamModuleHint, default_multi_stream_module_split_hint, default_multi_stream_module_concat_hint
import copy
import os
import time
import argparse

import opentuner
from opentuner.api import TuningRunManager
from opentuner.measurement.interface import DefaultMeasurementInterface
from opentuner import ConfigurationManipulator
from opentuner import IntegerParameter, EnumParameter
from opentuner import Result

import multiprocessing as mp

# def performance_measurement(graph_module, x):
#     with torch.no_grad():
#         original_traced_model = torch.jit.trace(graph_module, x).eval()
#         original_traced_model = torch.jit.freeze(original_traced_model)
#         for _ in range(3):
#             original_traced_model(x)
#         test_model = original_traced_model
#         # test_model = original_traced_model
#         stream_number = 28
#         cpu_pool = ipex.cpu.runtime.CPUPool(node_id=0)
#         multi_stream_model = ipex.cpu.runtime.MultiStreamModule(original_traced_model,
#                                                             num_streams=stream_number,
#                                                             cpu_pool=cpu_pool)
#         test_model = multi_stream_model

#         for _ in range(30):
#             test_model(x)

#         for _ in range(50):
#             test_model(x)
# def single_performance_measurement(test_model, batch_size, sample_input_bs_changed):
#     with torch.no_grad():
#         # Warm_up run
#         for _ in range(mixed_affinity_tuner_config.warmup_iterations):
#             test_model(*sample_input_bs_changed)
#         # Formal Run
#         elapse_cost = 0.0
#         for _ in range(mixed_affinity_tuner_config.benchmark_iterations):
#             start_time = time.time()
#             test_model(*sample_input_bs_changed)
#             elapse_cost += (time.time() - start_time)
#         current_throughput = (batch_size * mixed_affinity_tuner_config.benchmark_iterations) / elapse_cost
#         print("---- New opentuner cfg throughput is: {} fps".format(current_throughput), flush=True)
#         return 1.0 / current_throughput


def performance_measurement_v2(graph_module, x, config, q):
    if isinstance(x, torch.Tensor):
        sample_input = (x,)
    sample_input_bs = sample_input[0].size(0) # Will be used to measurement the baseline performance
    global_original_graph_module = graph_module
    use_bf16 = False

    def created_multi_stream_module(graph_module, start_node_idx, end_node_idx, stream_number, loop_times, mini_bs, use_bf16):
        batch_size = stream_number * loop_times * mini_bs
        # Here we assume the batchsize is the first dimention
        sample_input_bs_changed = []
        for x in sample_input:
            new_size = list(x.size())
            new_size[0] = batch_size
            sample_input_bs_changed.append(torch.rand(new_size, dtype=x.dtype, layout=x.layout, device=x.device, requires_grad=x.requires_grad))
        sample_input_bs_changed = tuple(sample_input_bs_changed)

        core_list = ipex.cpu.runtime.get_core_list_of_node_id(0)
        core_number = core_list.__len__()
        traced_cpu_pool = ipex.cpu.runtime.CPUPool(core_list[0:(core_number//stream_number)])
        with torch.no_grad(), ipex.cpu.runtime.pin(traced_cpu_pool):
            # To use the fast path for better performance
            # We ensure the same OMP_THREADS and BS during trace and formal running
            graph_module_ = mixed_affinity_search_pass(graph_module, start_node_idx=start_node_idx, end_node_idx=end_node_idx, loop_times=loop_times, mini_bs=mini_bs)
            mixed_affinity_module = sharding_lowering_pass(graph_module_)
            traced_x = []
            for x in sample_input_bs_changed:
                traced_x.append(x[0:(batch_size//stream_number)])
            traced_x = tuple(traced_x)
            with torch.cpu.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16):
                traced_model = torch.jit.trace(mixed_affinity_module, traced_x).eval()
            traced_model = torch.jit.freeze(traced_model)
            # Running warm up steps
            for _ in range(3):
                traced_model(*traced_x)
            cpu_pool = ipex.cpu.runtime.CPUPool(node_id=0)
            multi_stream_model = ipex.cpu.runtime.MultiStreamModule(traced_model,
                                                                    num_streams=stream_number,
                                                                    cpu_pool=cpu_pool)
            return multi_stream_model, sample_input_bs_changed

    def single_performance_measurement(test_model, batch_size, sample_input_bs_changed):
        with torch.no_grad():
            # Warm_up run
            for _ in range(20):
                test_model(*sample_input_bs_changed)
            # Formal Run
            elapse_cost = 0.0
            for _ in range(50):
                start_time = time.time()
                test_model(*sample_input_bs_changed)
                elapse_cost += (time.time() - start_time)
            current_throughput = (batch_size * 50) / elapse_cost
            print("---- New opentuner cfg throughput is: {} fps".format(current_throughput), flush=True)
            return 1.0 / current_throughput

    def performance_measurement(cfg):
        stream_number = int(cfg['stream_number'])
        loop_times = int(cfg['loop_times'])
        batchsize_uniform = int(cfg['batchsize_uniform'])
        start_node_idx = int(cfg["cur_start_index"])
        end_node_idx = int(cfg["cur_end_index"])

        batch_size = batchsize_uniform * loop_times * stream_number
        print("---- start new opentuner cfg performance measurement ----", flush=True)
        print("batchsize_uniform is:{}".format(batchsize_uniform), flush=True)
        print("loop_times is:{}".format(loop_times), flush=True)
        print("stream_number is:{}".format(stream_number), flush=True)
        print("batch_size is:{}".format(batch_size), flush=True)
        # import pdb;pdb.set_trace()
        test_model, sample_input_bs_changed = created_multi_stream_module(graph_module=global_original_graph_module,
                                                                        start_node_idx=start_node_idx,
                                                                        end_node_idx=end_node_idx,
                                                                        stream_number=stream_number,
                                                                        loop_times=loop_times,
                                                                        mini_bs=batchsize_uniform,
                                                                        use_bf16=use_bf16)
        print("finish create multi_stream_module")
        return single_performance_measurement(test_model, batch_size, sample_input_bs_changed)

    manipulator = ConfigurationManipulator()
    manipulator.add_parameter(
        EnumParameter('batchsize_uniform', [1, 2, 3]))
    manipulator.add_parameter(
        EnumParameter('loop_times', [1, 2, 3]))
    cores_per_socket = ipex.cpu.runtime.get_core_list_of_node_id(0).__len__()
    stream_number_candidate = []
    for i in range(1, cores_per_socket+1):
        if cores_per_socket%i == 0:
            stream_number_candidate.append(i)
    manipulator.add_parameter(
        EnumParameter('stream_number', stream_number_candidate))  # batchSize = loop_times * batchsize_uniform * stream_number
    manipulator.add_parameter(
        EnumParameter('cur_start_index', [config["cur_start_index"]]))
    manipulator.add_parameter(
        EnumParameter('cur_end_index', [config["cur_end_index"]]))
    parser = argparse.ArgumentParser(parents=opentuner.argparsers())
    args, unknown = parser.parse_known_args()
    interface = DefaultMeasurementInterface(args=args,
                                            manipulator=manipulator,
                                            project_name='mixed_affnity',
                                            program_name='mixed_affnity_tuner',
                                            program_version='0.1')
    api = TuningRunManager(interface, args)
    total_tuning_time = 0.0
    try:
        for _ in range(20):
            start_time = time.time()
            desired_result = api.get_next_desired_result()
            if desired_result is None:
                # The search space for this example is very small, so sometimes
                # the techniques have trouble finding a config that hasn't already
                # been tested.  Change this to a continue to make it try again.
                break
            cfg = desired_result.configuration.data
            result = Result(time=performance_measurement(cfg))
            api.report_result(desired_result, result)
            total_tuning_time += time.time() - start_time
            print("total_tuning_time is: {}".format(total_tuning_time))
        best_cfg = api.get_best_configuration()
        api.finish()
        test_bs = int(best_cfg['batchsize_uniform'])*int(best_cfg['loop_times'])*int(best_cfg['stream_number'])
        test_model, sample_input_bs_changed = created_multi_stream_module(graph_module=global_original_graph_module,
                                                                        start_node_idx=int(best_cfg["cur_start_index"]),
                                                                        end_node_idx=int(best_cfg["cur_end_index"]),
                                                                        stream_number=int(best_cfg['stream_number']),
                                                                        loop_times=int(best_cfg['loop_times']),
                                                                        mini_bs=int(best_cfg['batchsize_uniform']),
                                                                        use_bf16=use_bf16)
        throughput = 1.0 / single_performance_measurement(test_model, test_bs, sample_input_bs_changed)
        print("throughput is: {}".format(throughput))
        q.put(total_tuning_time)
        return total_tuning_time
    except:
        # For issue of opentuner: https://github.com/jansel/opentuner/issues/158
        # When the tuning step of opentuner_search_iterations is smaller than the warm up requirement of  search algorithm
        # No best_cfg is found and api.finish will fail
        exit(0)

def test_func(model, x):
    graph = TaggingTracer().trace(model)
    graph_module = torch.fx.GraphModule(model, graph).eval()
    graph_module = ipex.optimize(graph_module, dtype=torch.float32)
    graph_module = init_node_annotator(graph_module)
    global_original_graph_module = graph_module

    # test_func(graph_module, x)
    config = {
            "start_node_index" : 0,
            "find_end_node": True
        }
    total_tuning_time = 0.0
    search_time = 3000
    q = mp.Queue()
    while total_tuning_time < search_time:
        config, status = heuristic_annotator(global_original_graph_module, config=copy.deepcopy(config))
        if status is False:
            # If heuristic_annotator return False, we will not continue the below mixed affinity search
            print("heuristic_annotator status return False. Continue next heuristic_annotator.", flush=True)
            break
        p = mp.Process(target=performance_measurement_v2, args=(graph_module, x, config, q))
        p.start()
        running_time = q.get()
        p.join()
        print("running_time is: {}".format(running_time))
        total_tuning_time += running_time
        # running_time = performance_measurement_v2(graph_module, x, config)
        # total_tuning_time += running_time
        print("Final total_tuning_time is: {}".format(total_tuning_time))

if __name__ == "__main__":
    x = torch.randn(28, 3, 224, 224).contiguous(memory_format=torch.channels_last)
    model = models.__dict__["resnet50"]().to(memory_format=torch.channels_last).eval()
    test_func(model, x)




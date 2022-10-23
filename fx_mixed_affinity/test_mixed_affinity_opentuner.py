import torch
import torchvision.models as models
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.fx.xpu_auto_shard import TaggingTracer, init_node_annotator, mixed_affinity_search_pass, sharding_lowering_pass, heuristic_annotator
import copy
import time
import sys

import opentuner
from opentuner import ConfigurationManipulator
from opentuner import IntegerParameter, EnumParameter
from opentuner import MeasurementInterface
from opentuner import Result
import argparse
from opentuner.api import TuningRunManager
from opentuner.measurement.interface import DefaultMeasurementInterface

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

    parser = argparse.ArgumentParser(parents=opentuner.argparsers())
    args = parser.parse_args()

    search_times = 100
    best_cost = sys.maxsize
    best_throughput = 0.0
    while search_times > 0:
        config, status = heuristic_annotator(graph_module, config=copy.deepcopy(config))

        print("**** Start new heuristic_annotator: start_index:{0}, end_index:{1}".format(config["cur_start_index"], config["cur_end_index"]), flush=True)
        # # For HardCode test
        # config["cur_start_index"] = 0
        # config["cur_end_index"] = 54
        # config["loop_times"] = 3
        # config["mini_bs"] = 3

        manipulator = ConfigurationManipulator()
        # manipulator.add_parameter(
        #     IntegerParameter('batchsize_uniform', 1, 9))
        # manipulator.add_parameter(
        #     IntegerParameter('loop_times', 1, 9))
        manipulator.add_parameter(
            EnumParameter('batchsize_uniform', [1, 2, 3]))
        manipulator.add_parameter(
            EnumParameter('loop_times', [1, 2, 3]))
        manipulator.add_parameter(
            EnumParameter('stream_number', [1, 4, 14, 56]))  # batchSize = loop_times * batchsize_uniform * stream_number
        interface = DefaultMeasurementInterface(args=args,
                                                manipulator=manipulator,
                                                project_name='mixed_affnity',
                                                program_name='rn50_bf16',
                                                program_version='0.1')
        api = TuningRunManager(interface, args)

        def single_performance_measurement(graph_module, batch_size, num_streams, loop_times, batchsize_uniform, config, use_bf16, use_multi_stream):
            x = torch.randn(batch_size, 3, 224, 224).contiguous(memory_format=torch.channels_last)
            if use_bf16:
                x = x.to(torch.bfloat16)

            if use_multi_stream:
                num_streams = num_streams

            start_node_idx = config["cur_start_index"]
            end_node_idx = config["cur_end_index"]
            loop_times = loop_times
            mini_bs = batchsize_uniform

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

                warmup_iterations = 20
                iterations = 50
                # Warm_up run
                for _ in range(warmup_iterations):
                    result = test_model(x)
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
                # print("elapse_cost is: {} second".format(elapse_cost), flush=True)
                # print("Time per step is: {} second".format(elapse_cost / iterations), flush=True)
                current_throughput = (batch_size * iterations) / elapse_cost
                print("---- New opentuner cfg throughput is: {} fps".format(current_throughput), flush=True)
                return 1.0 / current_throughput

        def performance_measurement(cfg):
            global graph_module
            global config

            batchsize_uniform = int(cfg['batchsize_uniform'])
            loop_times = int(cfg['loop_times'])
            stream_number = int(cfg['stream_number'])

            batch_size = batchsize_uniform * loop_times * stream_number
            print("---- start new opentuner cfg performance measurement ----", flush=True)
            print("batchsize_uniform is:{}".format(batchsize_uniform), flush=True)
            print("loop_times is:{}".format(loop_times), flush=True)
            print("stream_number is:{}".format(stream_number), flush=True)
            print("batch_size is:{}".format(batch_size), flush=True)
            return single_performance_measurement(graph_module, batch_size, stream_number, loop_times, batchsize_uniform, config, True, True)

        for x in range(10):
            desired_result = api.get_next_desired_result()
            if desired_result is None:
                # The search space for this example is very small, so sometimes
                # the techniques have trouble finding a config that hasn't already
                # been tested.  Change this to a continue to make it try again.
                break
            cfg = desired_result.configuration.data
            result = Result(time=performance_measurement(cfg))
            api.report_result(desired_result, result)
        best_cfg = api.get_best_configuration()
        api.finish()
        test_bs = int(best_cfg['batchsize_uniform'])*int(best_cfg['loop_times'])*int(best_cfg['stream_number'])
        throughput = 1.0 / single_performance_measurement(graph_module, test_bs, int(best_cfg['stream_number']), int(best_cfg['loop_times']), int(best_cfg['batchsize_uniform']), config, True, True)
        
        print('best opentuner cfg found was, cur_start_index:{0}, cur_start_index:{1}, batchsize_uniform: {2}, loop_times:{3}, stream_number:{4}, throughput:{5}'.format(\
                                                                            config["cur_start_index"],\
                                                                            config["cur_end_index"],\
                                                                            best_cfg['batchsize_uniform'],\
                                                                            best_cfg['loop_times'],\
                                                                            best_cfg['stream_number'],\
                                                                            throughput))
        if throughput > best_throughput:
            best_throughput = throughput
            best_config = config
            best_config["loop_times"] = best_cfg['loop_times']
            best_config["mini_bs"] = best_cfg['batchsize_uniform']
            best_config["stream_number"] = best_cfg['stream_number']

        search_times -= 1
    print("Final best config is: {0}. best_throughput: {1}".format(best_config, best_throughput), flush=True)



import torch
import torchvision.models as models
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.fx.xpu_auto_shard import mixed_affinity_tuner
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
    x = torch.randn(1, 3, 224, 224).contiguous(memory_format=torch.channels_last).to(torch.bfloat16)
    best_performance_model = mixed_affinity_tuner(module, x, dtype=torch.bfloat16, search_time=600.0)
    
    stream_number = best_performance_model.stream_number
    loop_times = best_performance_model.loop_times
    mini_bs = best_performance_model.mini_bs
    batch_size = stream_number * loop_times * mini_bs

    input = torch.randn(batch_size, 3, 224, 224).contiguous(memory_format=torch.channels_last).to(torch.bfloat16)
    with torch.no_grad():
        warmup_iterations = 20
        iterations = 50
        # Warm_up run
        for _ in range(warmup_iterations):
            best_performance_model(input)
        # Formal Run
        elapse_cost = 0.0
        for _ in range(iterations):
            start_time = time.time()
            best_performance_model(input)
            elapse_cost += (time.time() - start_time)
        current_throughput = (batch_size * iterations) / elapse_cost
        print("---- New opentuner cfg throughput is: {} fps".format(current_throughput), flush=True)




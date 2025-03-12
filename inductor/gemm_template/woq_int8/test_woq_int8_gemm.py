import torch
import torch._inductor.config as config
from torchao.quantization import quant_api
from torchao.utils import unwrap_tensor_subclass
import copy
import time
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.quantization import (
    prepare,
    convert,
)
import random
import numpy as np
import csv

local_seed = 2024

torch.manual_seed(local_seed) # Set PyTorch seed
np.random.seed(seed=local_seed) # Set Numpy seed
random.seed(local_seed) # Set the Python seed

config.freezing = True
config.max_autotune = True
config.max_autotune_gemm_backends = "CPP,ATEN"

# M, N, K
shapes = (
    # input tokens 1024; output tokens 128; BS 1
    (4096, 4096, 4096),
    (4096, 11008, 4096),
    (4096, 4096, 11008),
    (4096, 32000, 4096),

    (4, 4096, 4096),
    (4, 11008, 4096),
    (4, 4096, 11008),
    (4, 32000, 4096),

    # input tokens 1024; output tokens 128; BS 2
    (8192, 4096, 4096),
    (8192, 11008, 4096),
    (8192, 4096, 11008),
    (8192, 32000, 4096),

    (8, 4096, 4096),
    (8, 11008, 4096),
    (8, 4096, 11008),
    (8, 32000, 4096),

    # input tokens 2016; output tokens 32; BS 1
    (8064, 4096, 4096),
    (8064, 11008, 4096),
    (8064, 4096, 11008),
    (8064, 32000, 4096),

    # input tokens 2016; output tokens 32; BS 2
    (16128, 4096, 4096),
    (16128, 11008, 4096),
    (16128, 4096, 11008),
    (16128, 32000, 4096),
)

shapes = (
    # (8, 4096, 4096),
    # (8, 11008, 4096),
    # (8, 4096, 11008),
    (8, 32000, 4096),
)

class Model(torch.nn.Module):
    def __init__(self, M, N, k):
        super().__init__()
        self.linear = torch.nn.Linear(K, N, True)

    def forward(self, x, x2):
        tmp = self.linear(x)
        return tmp
        # return tmp + x2

# skip_benchmark = True
skip_benchmark = False

def benchmark(m, input, input2):
    if skip_benchmark:
        return

    warm_up_iter = 50
    iter = 200

    for _ in range(warm_up_iter):
        m(input, input2)
    
    start = time.time()
    for _ in range(iter):
        m(input, input2)
    time_per_iter = (time.time() - start) / iter
    print("---- time is : {}".format(time_per_iter), flush=True)
    return time_per_iter

if __name__ == "__main__":
    results = []
    with open('result.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow(['shape', 'M', 'N', 'K', 'Inductor BF16 gemm template', 'IPEX WOQ INT8', 'Inductor WOQ INT8'])
        for shape in shapes:

            torch._dynamo.reset()

            result = [shape,]
            M, N, K = shape
            result.extend(shape)
            m = Model(M, N, K).eval()
            input = torch.randn(M, K).to(torch.bfloat16)
            input2 = torch.randn(M, N).to(torch.bfloat16)
            with torch.autocast(device_type="cpu"), torch.no_grad():

                # BF16 Run
                bf16_m = copy.deepcopy(m)
                bf16_cm = torch.compile(bf16_m)
                bf16_cm(input, input2)
                print("---- benchmark Inductor BF16 ----", flush=True)
                inductor_bf16_time_per_iter = benchmark(bf16_cm, input, input2)
                result.append(inductor_bf16_time_per_iter)


                # IPEX WOQ int8
                ipex_m = copy.deepcopy(m)
                qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping()
                prepared_model = prepare(m, qconfig, example_inputs=(input, input2), inplace=False)
                converted_model = convert(prepared_model)
                traced_model = torch.jit.trace(converted_model, (input, input2))
                traced_model = torch.jit.freeze(traced_model)
                traced_model(input, input2)
                print("---- benchmark IPEX WOQ INT8 ----", flush=True)
                ipex_woq_int8_time_per_iter = benchmark(traced_model, input, input2)
                result.append(ipex_woq_int8_time_per_iter)


                # Inductor WOQ int8
                quant_api.quantize_(m, quant_api.int8_weight_only(), set_inductor_config=False)
                unwrap_tensor_subclass(m)

                ref_res = m(input, input2)

                cm = torch.compile(m)
                res = cm(input, input2)
                print("---- benchmark Inductor WOQ INT8 ----", flush=True)
                inductor_woq_int8_time_per_iter = benchmark(cm, input, input2)
                result.append(inductor_woq_int8_time_per_iter)
                # print("ref_res is: {}".format(ref_res), flush=True)
                # print("res is: {}".format(res), flush=True)
                print(torch.allclose(ref_res, res, atol=1e-2, rtol=1e-2), flush=True)
            print("result is: {}".format(result), flush=True)
            spamwriter.writerow(result)
            results.append(result)

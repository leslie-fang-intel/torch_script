# https://github.com/pytorch/pytorch/issues/147596#issuecomment-2713199475

import os
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

import time
import torch
import torch.profiler
import oneccl_bindings_for_pytorch
from pyinstrument import Profiler

import torch
import os
import torch.backends.mkldnn
import torch.backends.openmp

print(f"Using {torch.get_num_threads()} threads (PyTorch)")
print(f"OMP_NUM_THREADS={os.getenv('OMP_NUM_THREADS')}")

# # Ensure PyTorch respects the OMP setting
# torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "56")))

# print(f"Now using {torch.get_num_threads()} threads after setting manually")



model_id = "meta-llama/Llama-3.1-8B-Instruct"

def main(is_tp, rank, world_size) -> None:
    backend = "ccl"
    print(is_tp)
    if is_tp:
        dist.init_process_group(backend)

    model_kwargs = dict(torch_dtype=torch.bfloat16)
    if is_tp:
        model_kwargs["tp_plan"] = "auto"
    else:
        model_kwargs["device_map"] = "cpu"

    # Retrieve tensor parallel model
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    print(model.dtype)

    # Prepare input tokens
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    prompt = "Can I help" * 200
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512).input_ids.to(model.device)
    print(f"inpu shape is {inputs.shape}")

    # model = torch.compile(model)
    # warm-up
    if is_tp:
        dist.barrier()
    for i in range(5):
        with torch.no_grad():
            outputs = model(inputs)

    if is_tp:
        dist.barrier()
    # profiler = Profiler()
    # profiler.start()

    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA,
    #     ],
    # ) as prof:
    for i in range(5):
        with torch.no_grad():
            start = time.time()
            outputs = model(inputs)
            end = time.time()
            print(f"time cost {(end-start)*1000} ms")
    
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    # profiler.stop()
    # with open(f"profile_tp_{is_tp}_backend_{backend}_rank_{rank}.html", "w") as f:
    #     f.write(profiler.output_html())

    # count = 0
    # # if rank == 0:
    # for name, parameter in model.named_parameters():
    #     if isinstance(parameter.data, torch.distributed.tensor.DTensor):
    #         if name == "model.layers.0.self_attn.q_proj.weight":
    #             print(f"name: {name}\nparameter: {parameter}", flush=True)
    #             original_shape = parameter.data.shape
    #             shape = parameter.data.to_local().shape
    #             print(f"paramater local shape is {shape}", flush=True)
    #             print(f"paramater original shape is {original_shape}", flush=True)
    #             count += 1
    #             if count > 2:
    #                 break

    print(outputs)
    print("---- finish rank is: {}----".format(rank), flush=True)


if __name__ == "__main__":
    rank = int(os.environ["RANK"]) if "RANK" in os.environ else 0
    world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    is_tp = "RANK" in os.environ
    main(is_tp, rank, world_size)
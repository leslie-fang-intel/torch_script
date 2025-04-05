import torch
from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer
from torchao.quantization import int4_weight_only
from torchao.dtypes import Int4CPULayout
import torch._inductor.config as config

config.freezing = True
# config.max_autotune = True

with torch.no_grad():
    model_name = "meta-llama/Meta-Llama-3-8B"

    ## Test 1: AutoQuant
    quantization_config = TorchAoConfig("autoquant")

    ## Test 2: WOQ INT4
    ## quant_config = int4_weight_only(group_size=128, layout=Int4CPULayout(), set_inductor_config=False)
    # quantization_config = TorchAoConfig(quant_type="int4_weight_only")

    quantized_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        quantization_config=quantization_config
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt = "Hi, how are you today?"
    inputs = tokenizer(prompt, return_tensors="pt")
    # Warm up run to record shapes for autoquant
    generate_ids = quantized_model.generate(inputs.input_ids, max_length=100, cache_implementation="static")
    quantized_model.finalize_autoquant()

    print("---- start the second run ----", flush=True)
    quantized_model.forward = torch.compile(quantized_model.forward)

    generate_ids = quantized_model.generate(inputs.input_ids, max_length=100, cache_implementation="static")

    print("---- start the thrid run ----", flush=True)
    generate_ids = quantized_model.generate(inputs.input_ids, max_length=100, cache_implementation="static")
    res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # print("res is: {}".format(res), flush=True)
    print("-------------------", flush=True)
    print(res, flush=True)
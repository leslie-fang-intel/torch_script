import torch
from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer
from torchao.quantization import int4_weight_only
from torchao.dtypes import Int4CPULayout

model_name = "meta-llama/Meta-Llama-3-8B"

## Test 1: AutoQuant
quantization_config = TorchAoConfig("autoquant", min_sqnr=None)

## Test 2: WOQ INT4
## quant_config = int4_weight_only(group_size=128, layout=Int4CPULayout(), set_inductor_config=False)
# quantization_config = TorchAoConfig(quant_type="int4_weight_only")

quantized_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    # device_map="cpu",
    quantization_config=quantization_config
)

# output_dir = "llama3-8b-int4wo-128"
# quantized_model.save_pretrained(output_dir, safe_serialization=False)

# ckpt_id = "llama3-8b-int4wo-128"  # or huggingface hub model id
# loaded_quantized_model = AutoModelForCausalLM.from_pretrained(ckpt_id, device_map="auto", torch_dtype="auto")


tokenizer = AutoTokenizer.from_pretrained(model_name)
prompt = "Hi, how are you today?"
inputs = tokenizer(prompt, return_tensors="pt")
generate_ids = quantized_model.generate(inputs.input_ids, max_length=100)
res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

# print("res is: {}".format(res), flush=True)
print("-------------------", flush=True)
print(res, flush=True)
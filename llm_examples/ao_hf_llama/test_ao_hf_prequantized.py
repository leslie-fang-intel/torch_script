import torch
from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer
from torchao.quantization import int4_weight_only
from torchao.dtypes import Int4CPULayout

# model_name = "jerryzh168/Meta-Llama-3-8B-torchao-int4_weight_only-gs_128"

model_name = "marksaroufim/Meta-Llama-3-8B-torchao-int8_weight_only"

model_name_tokenizer = "meta-llama/Meta-Llama-3-8B"

quantized_model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name_tokenizer)
prompt = "Hi, how are you today?"
inputs = tokenizer(prompt, return_tensors="pt")
generate_ids = quantized_model.generate(inputs.input_ids, max_length=100)
res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

# print("res is: {}".format(res), flush=True)
print("-------------------", flush=True)
print(res, flush=True)
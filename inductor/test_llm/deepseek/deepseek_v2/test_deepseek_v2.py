from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch._inductor.config as config

config.freezing = True


with torch.no_grad():
    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-Coder-V2-Lite-Base",
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-Coder-V2-Lite-Base",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        _attn_implementation="eager",)
    input_text = "#write a quick sort algorithm in C++"
    # input_text = "What's your version?"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    model.forward = torch.compile(model.forward)

    outputs = model.generate(**inputs, max_length=11)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

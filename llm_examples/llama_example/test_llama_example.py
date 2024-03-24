from transformers import pipeline

def test_task_CausalLM():
    from transformers import AutoModel, AutoTokenizer
    from transformers import LlamaForCausalLM, LlamaTokenizer
    # model = AutoModel.from_pretrained('meta-llama/Llama-2-7b-hf')
    model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', low_cpu_mem_usage=True, torchscript=True)
    tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')

    # prompt = "Hey, are you consciours? Can you talk to me?"
    
    prompt = "Hi, how are you today?"
    inputs = tokenizer(prompt, return_tensors="pt")
    generate_ids = model.generate(inputs.input_ids, max_length=100)
    res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # print("res is: {}".format(res), flush=True)
    print("-------------------", flush=True)
    print(res, flush=True)

if __name__ == "__main__":
    test_task_CausalLM()

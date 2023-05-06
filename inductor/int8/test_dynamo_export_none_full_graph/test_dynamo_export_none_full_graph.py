import torch
import transformers
from datasets import load_dataset
import copy
import torch._dynamo as torchdynamo
import torch._inductor
from torch.utils.data.sampler import BatchSampler, RandomSampler
import json
import pathlib
# from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer

# model = LlamaForCausalLM.from_pretrained("llama-7b", low_cpu_mem_usage=True, torch_dtype=torch.float32)
# tokenizer = LlamaTokenizer.from_pretrained("llama-7b")
# model = model.eval().to(torch.device("cpu"))
# model = model.to(memory_format=torch.channels_last)

# def dynamo_run(model, dataset):
#     current_path = pathlib.Path(__file__).parent.resolve()
#     with open(str(current_path) + '/prompt.json') as f:
#         prompt_pool = json.load(f)
#     prompt = prompt_pool[32]
#     batch_size = 1
#     prompt = [prompt] * batch_size
#     generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=4)

#     model.generate = torch.compile(model.generate, backend='inductor', dynamic=True)

#     num_iter = 10
#     for i in range(num_iter):
#         input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(torch.device("cpu"))
#         gen_tokens, latency_list = model.generate(input_ids, max_new_tokens=32, **generate_kwargs)
    
#     # dummy = next(iter(dataloader))
#     # example_inputs = [dummy["input_ids"], dummy["attention_mask"], dummy["decoder_input_ids"]]
#     # with torch.no_grad():
#     #     # m, guards = torchdynamo.export(m, *copy.deepcopy(example_inputs), aten_graph=True)

#     #     # print("data is: {}".format(data), flush=True)
#     #     # model(*example_inputs)
#     #     # print("---- Finish raw model run 222 ----", flush=True)
        
#     #     model = torch.compile(model, backend='inductor', dynamic=True)
#     #     print("---- Finish torch compiler ----", flush=True)
#     #     model(*example_inputs)
#     #     print("---- Finish compiled_m model run ----", flush=True)
#     return


def test_fnet_base():
    # model_name = 'gokuls/BERT-tiny-sst2'
    model_name = 'gchhablani/fnet-base-finetuned-sst2'
    model = transformers.AutoModel.from_pretrained(model_name).eval().to(torch.device("cpu")).to(memory_format=torch.channels_last)
    def get_example_inputs(model_name, dataset_name='sst2'): 
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        dataset = load_dataset(dataset_name, split='validation')
        text = dataset[0]['text'] if dataset_name=='lambada' else dataset[0]['sentence']
        example_inputs = tokenizer(text, padding='max_length', max_length=195, return_tensors='pt')
        print("example_inputs is: {}".format(example_inputs), flush=True)
        example_inputs = example_inputs['input_ids']
        return example_inputs
    data = get_example_inputs(model_name, dataset_name='sst2')
    example_inputs = (data, )
    with torch.no_grad():
        # m, guards = torchdynamo.export(m, *copy.deepcopy(example_inputs), aten_graph=True)

        # print("data is: {}".format(data), flush=True)
        # model(*example_inputs)
        # print("---- Finish raw model run 222 ----", flush=True)
        
        model = torch.compile(model, backend='inductor', dynamic=True)
        print("---- Finish torch compiler ----", flush=True)
        model(*example_inputs)
        print("---- Finish compiled_m model run ----", flush=True)

def test_bert_large_uncased():
    from transformers import BertTokenizer, BertModel
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    model = BertModel.from_pretrained("bert-large-uncased")
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)

    with torch.no_grad():
        # m, guards = torchdynamo.export(m, *copy.deepcopy(example_inputs), aten_graph=True)

        # print("data is: {}".format(data), flush=True)
        # model(*example_inputs)
        # print("---- Finish raw model run 222 ----", flush=True)
        
        model = torch.compile(model, backend='inductor', dynamic=True)
        print("---- Finish torch compiler ----", flush=True)
        model(**encoded_input)
        print("---- Finish compiled_m model run ----", flush=True)

def test_Llama():
    from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer

    model = LlamaForCausalLM.from_pretrained("llama-7b", low_cpu_mem_usage=True, torch_dtype=torch.float32)
    tokenizer = LlamaTokenizer.from_pretrained("llama-7b")
    model = model.eval().to(torch.device("cpu"))
    model = model.to(memory_format=torch.channels_last)

    current_path = pathlib.Path(__file__).parent.resolve()
    with open(str(current_path) + '/prompt.json') as f:
        prompt_pool = json.load(f)
    prompt = prompt_pool["32"]
    batch_size = 1
    prompt = [prompt] * batch_size
    generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=4)

    model.generate = torch.compile(model.generate, backend='inductor', dynamic=True)

    num_iter = 10
    for i in range(num_iter):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(torch.device("cpu"))
        gen_tokens, latency_list = model.generate(input_ids, max_new_tokens=32, **generate_kwargs)
    
    return

if __name__ == "__main__":
    # test_fnet_base()
    # test_bert_large_uncased()
    test_Llama()
    # dynamo_run(model, tokenized_dataset)
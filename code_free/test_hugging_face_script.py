import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

inputs = tokenizer("Hello world!", return_tensors="pt")

for i in range(10):
    print("Running step:{}".format(i), flush=True)
    if i == 5:
        with torch.profiler.profile(on_trace_ready=torch.profiler.tensorboard_trace_handler("./fp32_log")) as prof:
            outputs = model(**inputs)
        print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    else:
        outputs = model(**inputs)


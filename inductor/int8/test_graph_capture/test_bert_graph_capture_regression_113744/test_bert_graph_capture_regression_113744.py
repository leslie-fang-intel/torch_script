import torch
from torch._export import capture_pre_autograd_graph
from transformers import BertTokenizer, BertModel, AutoConfig

model_id = 'bert-base-uncased'
config = AutoConfig.from_pretrained(model_id, return_dict=False)
tokenizer = BertTokenizer.from_pretrained(model_id, return_dict=False)
model = BertModel.from_pretrained(model_id, config=config)

text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
# encoded_input is: {
# 'input_ids': tensor([[ 101, 5672, 2033, 2011, 2151, 3793, 2017, 1005, 1040, 2066, 1012,  102]]), 
# 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
# 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
# }

# encoded_input2 = {
#     'input_ids': torch.tensor([[ 101, 5672, 2033, 2011, 2151, 3793, 2017, 1005, 1040, 2066, 1012,  102]]), 
#     'token_type_ids': torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
#     'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
# }

with torch.no_grad():
    print("encoded_input is: {}".format(encoded_input), flush=True)
    # print("encoded_input2 is: {}".format(encoded_input2), flush=True)
    captured_model = capture_pre_autograd_graph(model, (), kwargs=encoded_input)
    print('Success')

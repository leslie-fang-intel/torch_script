import numpy as np
from numpy import testing
import torch
import random

local_seed = 2024

torch.manual_seed(local_seed) # Set PyTorch seed
np.random.seed(seed=local_seed) # Set Numpy seed
random.seed(local_seed) # Set the Python seed

DEVICE='cpu'

class Model0(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        max_1 = torch.max(args[1], args[0])
        max_2 = torch.max(max_1, args[2])
        flatten = max_2.flatten()
        matmul = torch.matmul(flatten, flatten)
        return (matmul,)

model_0 = Model0()
output_names_0 = ['v2_0']

class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        max_1 = torch.max(args[1], args[0])
        add = torch.add(max_1, args[0])
        cat = torch.cat((args[2], args[2]), dim = 0)
        max_2 = torch.max(max_1, args[2])
        flatten = max_2.flatten()
        matmul = torch.matmul(flatten, flatten)
        return (add, cat, matmul)

model_1 = Model1()
output_names_1 = ['v7_0', 'v10_0', 'v13_0']

# data_0 = np.array(6, dtype=np.uint8)
# data_1 = np.random.normal(5, 1, size=(43, 1, 32, 24)).astype(np.uint8)
# data_2 = np.random.normal(5, 1, size=(1, 1, 1)).astype(np.uint8)

data_0 = np.full((1), 6, dtype=np.uint8)
data_1 = np.full((43, 1, 32, 24), 6, dtype=np.uint8)
data_2 = np.full((1, 1, 1), 6, dtype=np.uint8)

# data_0 = np.array(6, dtype=np.int32)
# data_1 = np.random.normal(5, 1, size=(43, 1, 32, 24)).astype(np.int32)
# data_2 = np.random.normal(5, 1, size=(1, 1, 1)).astype(np.int32)

# print("data_0 is: {}".format(data_0), flush=True)
# print("data_1 is: {}".format(data_1), flush=True)
# print("data_2 is: {}".format(data_2), flush=True)
input_data_0 = [data_0,data_1,data_2,]

optmodel_0 = torch.compile(model_0, fullgraph=True, backend='inductor', mode=None)
model_out_0 = optmodel_0(*[torch.from_numpy(v).to(DEVICE) for v in input_data_0])
print("compiler model_out_0 is: {}".format(model_out_0), flush=True)
model_out_0 = [v.to(DEVICE).detach() for v in model_out_0] if isinstance(model_out_0, tuple) else [model_out_0.to(DEVICE).detach()]
model_out_0 = [v.cpu().resolve_conj().numpy() if v.is_conj() else v.cpu().numpy() for v in model_out_0]
output_0 = dict(zip(output_names_0, model_out_0))

input_data_1 = input_data_0

optmodel_1 = torch.compile(model_1, fullgraph=True, backend='inductor', mode=None)
model_out_1 = optmodel_1(*[torch.from_numpy(v).to(DEVICE) for v in input_data_1])
print("compiler model_out_1 is: {}".format(model_out_1[2]), flush=True)
model_out_1 = [v.to(DEVICE).detach() for v in model_out_1] if isinstance(model_out_1, tuple) else [model_out_1.to(DEVICE).detach()]
model_out_1 = [v.cpu().resolve_conj().numpy() if v.is_conj() else v.cpu().numpy() for v in model_out_1]
output_1 = dict(zip(output_names_1, model_out_1))
output_name_dict = {'v2_0': 'v13_0'}

print('=========================')
try:
    for tensor_name_0, tensor_name_1 in output_name_dict.items():
        testing.assert_allclose(output_0[tensor_name_0], output_1[tensor_name_1], atol=1e-6, err_msg=f'at {tensor_name_0}, {tensor_name_1}')
    print("torch_complie does not trigger assertion")
except AssertionError as e:
    print("torch_complie triggers assertion")
    print(e)
print('=========================')

model_out_0 = model_0(*[torch.from_numpy(v).to(DEVICE) for v in input_data_0])
print("eager model_out_0 is: {}".format(model_out_0), flush=True)
model_out_0 = [v.to(DEVICE).detach() for v in model_out_0] if isinstance(model_out_0, tuple) else [model_out_0.to(DEVICE).detach()]
model_out_0 = [v.cpu().resolve_conj().numpy() if v.is_conj() else v.cpu().numpy() for v in model_out_0]
output_0 = dict(zip(output_names_0, model_out_0))

model_out_1 = model_1(*[torch.from_numpy(v).to(DEVICE) for v in input_data_1])
print("eager model_out_1 is: {}".format(model_out_1[2]), flush=True)
model_out_1 = [v.to(DEVICE).detach() for v in model_out_1] if isinstance(model_out_1, tuple) else [model_out_1.to(DEVICE).detach()]
model_out_1 = [v.cpu().resolve_conj().numpy() if v.is_conj() else v.cpu().numpy() for v in model_out_1]
output_1 = dict(zip(output_names_1, model_out_1))
# print("output_1 is: {}".format(output_1), flush=True)

print('=========================')
try:
    for tensor_name_0, tensor_name_1 in output_name_dict.items():
        testing.assert_allclose(output_0[tensor_name_0], output_1[tensor_name_1], atol=1e-6, err_msg=f'at {tensor_name_0}, {tensor_name_1}')
    print("torch_eager does not trigger assertion")
except AssertionError as e:
    print("torch_eager triggers assertion")
    print(e)
print('=========================')
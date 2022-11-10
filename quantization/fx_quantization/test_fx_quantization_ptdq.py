import torch
from torch.ao.quantization import QConfigMapping
import torch.quantization.quantize_fx as quantize_fx
import copy

# class M(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv = torch.nn.Conv2d(1, 1, 1)
#         self.relu = torch.nn.ReLU()

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.relu(x)
#         return x

# # create a model instance
# model_fp = M()

import torch.nn.functional as F
class M(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(weight)

    def forward(self, x):
        return F.linear(x, self.weight)

model_fp = M(torch.rand(1, 1)).eval()

# post training dynamic/weight_only quantization

# we need to deepcopy if we still want to keep model_fp unchanged after quantization since quantization apis change the input model
model_to_quantize = copy.deepcopy(model_fp)
model_to_quantize.eval()

#qconfig_mapping = QConfigMapping().set_global(torch.quantization.default_dynamic_qconfig)

#qconfig = torch.quantization.get_default_qconfig("onednn")
qconfig = torch.quantization.default_dynamic_qconfig
qconfig_mapping = {"": qconfig}
# qconfig_mapping = {
#     "object_type": [
#         (torch.nn.Conv2d, torch.quantization.default_dynamic_qconfig),
#         (torch.nn.ReLU, torch.quantization.default_dynamic_qconfig)
#     ]
# }

torch.backends.quantized.engine = 'onednn'

#input_tensor = torch.randn(4, 1, 4, 4)
input_tensor = torch.rand(1, 1)

# a tuple of one or more example inputs are needed to trace the model
example_inputs = (input_tensor, )
# prepare
model_prepared = quantize_fx.prepare_fx(model_to_quantize, qconfig_mapping, example_inputs)

print("model_prepared is: {}".format(model_prepared))
print("model_prepared.graph is: {}".format(model_prepared.graph))

# no calibration needed when we only have dynamic/weight_only quantization
# quantize
model_quantized = quantize_fx.convert_fx(model_prepared, qconfig_mapping)

print("model_quantized is: {}".format(model_quantized))
print("model_quantized.graph: {}".format(model_quantized.graph))
res = model_quantized(input_tensor)

# print("res is: {}".format(res))

# res_ref = model_fp(input_tensor)
# print("res_ref is: {}".format(res_ref))
# print(torch.allclose(res, res_ref))

# def print_size_of_model(model):
#     import os
#     torch.save(model.state_dict(), "temp.p")
#     print('Size (MB):', os.path.getsize("temp.p")/1e6)
#     os.remove('temp.p')
# print_size_of_model(model_fp)
# print_size_of_model(model_quantized)
# print(model_quantized._scale_0)


# #
# # fusion
# #
# model_to_quantize = copy.deepcopy(model_fp)
# model_fused = quantize_fx.fuse_fx(model_to_quantize)

# print("model_fused is: {}".format(model_fused))

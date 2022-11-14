import torch
from torch.ao.quantization import QConfigMapping
import torch.quantization.quantize_fx as quantize_fx
import copy

# class M(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         # QuantStub converts tensors from floating point to quantized
#         # self.quant = torch.quantization.QuantStub()
#         self.conv = torch.nn.Conv2d(1, 1, 1)
#         self.relu = torch.nn.ReLU()
#         # DeQuantStub converts tensors from quantized to floating point
#         # self.dequant = torch.quantization.DeQuantStub()

#     def forward(self, x):
#         # manually specify where tensors will be converted from floating
#         # point to quantized in the quantized model
#         # x = self.quant(x)
#         x = self.conv(x)
#         x = self.relu(x)
#         # manually specify where tensors will be converted from quantized
#         # to floating point in the quantized model
#         # x = self.dequant(x)
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

# input_tensor = torch.randn(4, 1, 4, 4)
input_tensor = torch.rand(1, 1)

#
# quantization aware training for static quantization
#

torch.backends.quantized.engine = 'onednn'

example_inputs = (input_tensor, )

model_to_quantize = copy.deepcopy(model_fp)
qconfig_mapping = QConfigMapping().set_global(torch.quantization.get_default_qat_qconfig('onednn'))
model_to_quantize.train()
# prepare
model_prepared = quantize_fx.prepare_qat_fx(model_to_quantize, qconfig_mapping, example_inputs)
# training loop (not shown)
# quantize
model_quantized = quantize_fx.convert_fx(model_prepared)

model_quantized(input_tensor)

# #
# # fusion
# #
# model_to_quantize = copy.deepcopy(model_fp)
# model_fused = quantize_fx.fuse_fx(model_to_quantize)

# print(model_fused)

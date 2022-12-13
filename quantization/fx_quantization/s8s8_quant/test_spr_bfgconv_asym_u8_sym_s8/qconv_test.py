import torch
from torch import nn
import torch.nn.quantized as nnq
N = 1 # batch size
IC = 1 # input channel
IH = 5 # input height
IW = 5 # input width
ID = 2 # input depth for conv3d
OC = 1 # output channel
OD = 2 # output depth for conv3d
KD = 1 # kernel depth for conv3d
KH = 1 # kernel height
KW = 1 # kernal width
groups = 1
use_bias = False
conv2d_fns = (torch.ops.quantized.conv2d_prepack, torch.ops.quantized.conv2d, torch.ops.quantized.conv2d_unpack, False, False)
conv_transpose2d_fns = (torch.ops.quantized.conv_transpose2d_prepack, torch.ops.quantized.conv_transpose2d, torch.ops.quantized.conv_transpose2d_unpack, False, True)
conv3d_fns = (torch.ops.quantized.conv3d_prepack, torch.ops.quantized.conv3d, torch.ops.quantized.conv3d_unpack, True, False)
conv_transpose3d_fns = (torch.ops.quantized.conv_transpose3d_prepack, torch.ops.quantized.conv_transpose3d, torch.ops.quantized.conv_transpose3d_unpack, True, True)
func_lists = (conv2d_fns, conv_transpose2d_fns, conv3d_fns, conv_transpose3d_fns)
# Select op for test: 0=conv2d, 1=deconv2d, 2=conv3d, 3=deconv3d
prepack_fn, conv_fn, unpack_fn, is_3d, transposed = func_lists[0]
## Input
x_size = (N, IC, ID, IH, IW) if is_3d else (N, IC, IH, IW)
x = 2*torch.ones(x_size, dtype=torch.float32)#+1
x_scale = 1.0 #1.5
x_zero_point = 1
## weight
w_conv_size = (OC, IC//groups, OD, KH, KW) if is_3d else (OC, IC//groups, KH, KW)
w_deconv_size = (IC, OC//groups, KD, KH, KW) if is_3d else (IC, OC//groups, KH, KW)
w_size = w_deconv_size if transposed else w_conv_size
w = 1*torch.ones(w_size, dtype=torch.float)#+1
w_scale = 1.0
w_zero_point = 0
## bias
bias_float = 1*torch.randn(OC, dtype=torch.float) if use_bias else None
strides = [1, 1, 1] if is_3d else [1, 1]
pads = [1, 1, 1] if is_3d else [0, 0]
pads_out = [1, 1, 1] if is_3d else [0, 0]
dilations = [1, 1, 1] if is_3d else [1, 1]
y_scale = 1.0 #5.0
y_zero_point = 2
def prepack_weight(qw):
    return prepack_fn(qw, bias_float, strides, pads, pads_out, dilations, groups)\
        if transposed else prepack_fn(qw, bias_float, strides, pads, dilations, groups)
def run_conv(qx, w_packed):
    return conv_fn(qx, w_packed, y_scale, y_zero_point)
def unpack(w_packed):
    return unpack_fn(w_packed) # tuple of (W_unpacked, bias)
torch.backends.quantized.engine = 'fbgemm'
qx_fbgemm = torch.quantize_per_tensor(x, scale=x_scale, zero_point=x_zero_point, dtype=torch.quint8)
qw_fbgemm = torch.quantize_per_tensor(w, scale=w_scale, zero_point=w_zero_point, dtype=torch.qint8)
w_packed_fbgemm = prepack_weight(qw_fbgemm)
qy_fbgemm = run_conv(qx_fbgemm, w_packed_fbgemm)
torch.backends.quantized.engine = 'onednn'
qx_onednn = torch.quantize_per_tensor(x, scale=x_scale, zero_point=x_zero_point, dtype=torch.quint8)
qw_onednn = torch.quantize_per_tensor(w, scale=w_scale, zero_point=w_zero_point, dtype=torch.qint8)
w_packed_onednn = prepack_weight(qw_onednn)
qy_onednn = run_conv(qx_onednn, w_packed_onednn)
## Compare results
print("fbgemm qy == onednn qy ?", torch.equal(qy_fbgemm, qy_onednn))
if not torch.equal(qy_fbgemm, qy_onednn):
    print('fbgemm qy =')
    print(qy_fbgemm.int_repr())
    print('onednn qy =')
    print(qy_onednn.int_repr())
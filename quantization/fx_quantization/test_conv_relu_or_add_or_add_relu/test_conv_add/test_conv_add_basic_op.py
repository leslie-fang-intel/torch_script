import torch
import numpy as np
import random

local_seed = 2022
torch.manual_seed(local_seed) # Set PyTorch seed
np.random.seed(seed=local_seed) # Set Numpy seed
random.seed(local_seed) # Set the Python seed

def test_pytorch_op():
    x1 = torch.rand(1, 9, 2, 2)
    x2 = torch.rand(1, 3, 3, 3)
    # conv = torch.nn.Conv2d(3, 9, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    input_channel = 3
    output_channel = 9
    kernel_size = (3, 3)
    strides = (2, 2)
    paddings = (1, 1)
    dilations = (1, 1)
    groups = 1
    use_bias = False

    x2 = torch.rand(1, input_channel, 3, 3)
    W = torch.rand(output_channel, input_channel, 3, 3)

    conv = torch.nn.Conv2d(input_channel, output_channel, kernel_size,
        stride=strides, padding=paddings, bias=use_bias,
        dilation=dilations, groups=groups)

    qconv = torch.ops.quantized.conv2d
    qconv_prepack = torch.ops.quantized.conv2d_prepack
    qadd = torch.ops.quantized.add

    # Update weight of conv and calculate reference result
    conv.weight = torch.nn.Parameter(W, requires_grad=False)
    conv.bias = None
    res_ref = torch.add(conv(x2), x1)
    # print("res_ref is: {}".format(res_ref))

    torch.backends.quantized.engine = "onednn"
    activation_dtype = torch.quint8
    weight_dtype = torch.qint8

    # s8 for weight
    w_scale = 0.02
    w_zero_point = 0

    # u8 for activation
    x1_scale = 0.021
    x1_zero_point = 128

    x2_scale = 0.022
    x2_zero_point = 128

    # u8 for result
    Y_scale = 0.12
    Y_zero_point = 128   
    x1_q = torch.quantize_per_tensor(
        x1, scale=x1_scale, zero_point=x1_zero_point, dtype=activation_dtype)
    x2_q = torch.quantize_per_tensor(
        x2, scale=x2_scale, zero_point=x2_zero_point, dtype=activation_dtype)

    w_q = torch.quantize_per_tensor(
        W, scale=w_scale, zero_point=w_zero_point, dtype=weight_dtype)

    res_ref_q = torch.quantize_per_tensor(
        res_ref, scale=Y_scale, zero_point=Y_zero_point, dtype=activation_dtype)

    # Step2: Native Int8 path
    W_prepack = qconv_prepack(
        w_q, None, strides, paddings, dilations, groups)
    # Scale and Zero point should be the output's scale and zero point
    Y_q2 = qadd( 
        qconv(
        x2_q,
        W_prepack,
        Y_scale,
        Y_zero_point,),
        x1_q,
        Y_scale,
        Y_zero_point
    )
    # print("res_ref_q.int_repr() is: {}".format(res_ref_q.int_repr()))
    # print("Y_q2.int_repr() is: {}".format(Y_q2.int_repr()))
    # print(torch.allclose(res_ref_q.int_repr(), Y_q2.int_repr()))
    #assert torch.allclose(res_ref_q.int_repr(), Y_q.int_repr())
    import numpy as np
    np.testing.assert_array_almost_equal(res_ref_q.int_repr().cpu().numpy(),
        Y_q2.int_repr().cpu().numpy(),
        decimal=0)

    # Step3
    qconv_add = torch.ops.quantized.conv2d_add
    W_prepack_conv_add = qconv_prepack(
        w_q, None, strides, paddings, dilations, groups)
    print("Finish the qconv weight prepack", flush=True)
    # Scale and Zero point should be the output's scale and zero point
    Y_q_conv_add = qconv_add(
        x2_q, # used for conv
        x1_q,
        W_prepack_conv_add,
        Y_scale,
        Y_zero_point
    )
    print(torch.allclose(res_ref_q.int_repr(), Y_q_conv_add.int_repr()))

    # print("x1_q.int_repr() is: {}".format(x1_q.int_repr()))
    print("res_ref_q.int_repr() is: {}".format(res_ref_q.int_repr()))
    print("Y_q_conv_add.int_repr() is: {}".format(Y_q_conv_add.int_repr()))
    import numpy as np
    np.testing.assert_array_almost_equal(res_ref_q.int_repr().cpu().numpy(),
        Y_q_conv_add.int_repr().cpu().numpy(),
        decimal=0)
    print("Finish:{}".format(True))

if __name__ == "__main__":
    test_pytorch_op()
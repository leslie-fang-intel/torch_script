import torch
import numpy as np
import random
import torch.ao.quantization.fx._decomposed

local_seed = 2022
torch.manual_seed(local_seed) # Set PyTorch seed
np.random.seed(seed=local_seed) # Set Numpy seed
random.seed(local_seed) # Set the Python seed

class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(64, 128, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    def forward(self, x):
        return self.conv1(x)

def test_pytorch_op():
    with torch.no_grad():
        conv1_input_channel = 64
        conv1_output_channel = 128

        strides = (2, 2)
        paddings = (1, 1)
        dilations = (1, 1)
        groups = 1

        model = SimpleNet().eval()
        x = torch.rand(1, 64, 3, 3)

        conv1 = model.conv1
        qconv = torch.ops.quantized.conv2d
        qconv_prepack = torch.ops.quantized.conv2d_prepack

        torch.backends.quantized.engine = "x86"
        activation_dtype = torch.quint8
        weight_dtype = torch.qint8

        # u8 for activation
        x_scale = 0.007840843871235847
        x_zero_point = 128

        # u8 for result
        Y_scale = 0.0055459425784647465
        Y_zero_point = 128   

        x_q = torch.quantize_per_tensor(
            x, scale=x_scale, zero_point=x_zero_point, dtype=activation_dtype)
        # 这里因为w1_scale, w2_scale 只取了小数点后3位，所以精度不够
        # 我们应该直接从module去save 量化好的tensor,并load
        w1_q = torch.load('conv1_q_weight.pt')

        # Step1: Calculate the refer result
        model.conv1.weight = torch.nn.Parameter(torch.dequantize(w1_q))
        res_ref = conv1(torch.dequantize(x_q))
        res_ref = torch.dequantize(torch.quantize_per_tensor(res_ref, 
                scale=Y_scale, zero_point=Y_zero_point, dtype=activation_dtype))

        # Step2: Calculate the quant conv result
        W1_prepack = qconv_prepack(
            w1_q, None, strides, paddings, dilations, groups)

        conv1 = qconv(
            x_q,
            W1_prepack,
            Y_scale,
            Y_zero_point,)

        res = torch.dequantize(conv1)

        import numpy as np

        print("res_ref is: {}".format(res_ref))
        print("res is: {}".format(res))
        print(torch.allclose(res_ref, res, rtol=0.1, atol=0.1))
        
        np.testing.assert_array_almost_equal(res_ref.cpu().numpy(),
        res.cpu().numpy(),
        decimal=2)

def test_inductor_pytorch_op():
    with torch.no_grad():
        conv1_input_channel = 64
        conv1_output_channel = 128

        strides = (2, 2)
        paddings = (1, 1)
        dilations = (1, 1)
        groups = 1

        model = SimpleNet().eval()
        x = torch.rand(1, 64, 3, 3)
        x_shape = (1, 64, 3, 3)


        conv1 = model.conv1
        qconv = torch.ops.quantized_decomposed.conv_unary_inductor
        qconv_prepack = torch.ops.quantized.conv_prepack_cpu_tensor

        torch.backends.quantized.engine = "x86"
        activation_dtype = torch.quint8
        weight_dtype = torch.qint8

        # u8 for activation
        x_scale = 0.007840843871235847
        x_zero_point = 128

        # u8 for result
        Y_scale = 0.0055459425784647465
        Y_zero_point = 128   

        x_q = torch.quantize_per_tensor(
            x, scale=x_scale, zero_point=x_zero_point, dtype=activation_dtype)
        # 这里因为w1_scale, w2_scale 只取了小数点后3位，所以精度不够
        # 我们应该直接从module去save 量化好的tensor,并load
        w1_q = torch.load('conv1_q_weight.pt')

        # Step1: Calculate the refer result
        model.conv1.weight = torch.nn.Parameter(torch.dequantize(w1_q))
        res_ref = conv1(torch.dequantize(x_q))
        res_ref = torch.dequantize(torch.quantize_per_tensor(res_ref, 
                scale=Y_scale, zero_point=Y_zero_point, dtype=activation_dtype))

        # Step2: Calculate the quant conv result
        w1_scales = w1_q.q_per_channel_scales()
        print(w1_scales)
        w1_zp = w1_q.q_per_channel_zero_points()

        w1_q = w1_q.int_repr()
        # print(w1_q.size())
        # exit(-1)
        w_axis = 0
        packed_weight, packed_bias = qconv_prepack(w1_q, w1_scales, x_shape, x_scale, x_zero_point,
            None, strides, paddings, dilations, groups)

        x_q = x_q.int_repr()
        conv1 = qconv(
            x_q, torch.tensor(x_scale), torch.tensor(x_zero_point), packed_weight, w1_scales, w1_zp, w_axis,
            None, strides, paddings, dilations, groups, torch.tensor(Y_scale), torch.tensor(Y_zero_point), "none")
        res = torch.ops.quantized_decomposed.dequantize_per_tensor(conv1, torch.tensor(Y_scale), torch.tensor(Y_zero_point), 0, 255, torch.uint8)

        import numpy as np

        print("res_ref is: {}".format(res_ref))
        print("res is: {}".format(res))
        print(torch.allclose(res_ref, res, rtol=0.1, atol=0.1))
        
        np.testing.assert_array_almost_equal(res_ref.cpu().numpy(),
        res.cpu().numpy(),
        decimal=2)

if __name__ == "__main__":
    # test_pytorch_op()
    test_inductor_pytorch_op()
import torch
import numpy as np
import random

local_seed = 2022
torch.manual_seed(local_seed) # Set PyTorch seed
np.random.seed(seed=local_seed) # Set Numpy seed
random.seed(local_seed) # Set the Python seed

class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(64, 128, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.conv2 = torch.nn.Conv2d(64, 128, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # self.conv3 = torch.nn.Conv2d(128, 128, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        #3 self.relu = torch.nn.ReLU()
    def forward(self, x):
        # x = torch.add(x, x)
        x1 = torch.add(self.conv1(x), self.conv2(x))
        # Here we need conv2 to ensure the first conv's output is int8
        #print("x1 is: {}".format(x1))
        # return self.conv3(x1)
        return x1

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
        conv2 = model.conv2
        qconv = torch.ops.quantized.conv2d
        qconv_prepack = torch.ops.quantized.conv2d_prepack
        qadd = torch.ops.quantized.add

        # Update weight of conv and calculate reference result
        # conv.weight = torch.nn.Parameter(W, requires_grad=False)
        # conv.bias = None
        W1 = model.conv1.weight
        W2 = model.conv2.weight
        
        #x1 = torch.add(x, x)
        res_ref = torch.add(conv1(x), conv2(x))
        print("res_ref is: {}".format(res_ref))


        torch.backends.quantized.engine = "onednn"
        activation_dtype = torch.quint8
        weight_dtype = torch.qint8

        # u8 for activation
        x_scale = 0.007840843871235847
        x_zero_point = 128

        # s8 for weight
        w1_scale = torch.tensor([0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003,
        0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003,
        0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003,
        0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003,
        0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003,
        0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003,
        0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003,
        0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003,
        0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003,
        0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003,
        0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003,
        0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003,
        0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003,
        0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003,
        0.0003, 0.0003], dtype=torch.float64)
        w1_zero_point = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0])

        w2_scale = torch.tensor([0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003,
        0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003,
        0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003,
        0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003,
        0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003,
        0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003,
        0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003,
        0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003,
        0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003,
        0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003,
        0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003,
        0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003,
        0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003,
        0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003,
        0.0003, 0.0003], dtype=torch.float64)
        w2_zero_point = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0])

        # u8 for result
        Y_scale = 0.008548887446522713
        Y_zero_point = 128   

        # exit(-1)
        x_q = torch.quantize_per_tensor(
            x, scale=x_scale, zero_point=x_zero_point, dtype=activation_dtype)

        w1_q = torch.quantize_per_channel(
            W1, scales=w1_scale, zero_points=w1_zero_point, axis=0, dtype=weight_dtype)

        w2_q = torch.quantize_per_channel(
            W2, scales=w2_scale, zero_points=w2_zero_point, axis=0, dtype=weight_dtype)
        
        # 这里因为w1_scale, w2_scale 只取了小数点后3位，所以精度不够
        # 我们应该直接从module去save 量化好的tensor,并load
        w1_q = torch.load('conv1_q_weight.pt')
        w2_q = torch.load('conv2_q_weight.pt')
    
        # res_ref_q = torch.quantize_per_tensor(
        #     res_ref, scale=Y_scale, zero_point=Y_zero_point, dtype=activation_dtype)
        print(w1_q)
        print(w2_q)
        #exit(-1)
        # Step2: Native Int8 path
        W1_prepack = qconv_prepack(
            w1_q, None, strides, paddings, dilations, groups)
        W2_prepack = qconv_prepack(
            w2_q, None, strides, paddings, dilations, groups)
        # print(W1_prepack.unpack()[0])
        # print(W2_prepack.unpack()[0])

        #exit(-1)
        # Scale and Zero point should be the output's scale and zero point
        # first_add_scale = 0.0158
        # first_add_scale = 0.01578652672469616
        # first_add_zp = 128
        # add_1 = qadd(x_q, x_q, first_add_scale, first_add_zp)

        # print("add_1 is: {}".format(add_1))

        #----Done to Here----
        conv1 = qconv(
            x_q,
            W1_prepack,
            0.0055459425784647465,
            128,)
        
        print("conv1 is: {}".format(conv1))
        #exit(-1)
        conv2 = qconv(
            x_q,
            W2_prepack,
            0.005295046605169773,
            128,)
        # conv2 = qconv(
        #     add_1,
        #     W2_prepack,
        #     0.0154,
        #     126,)
        add_2 = qadd(conv2, conv1, Y_scale, Y_zero_point)

        res = torch.dequantize(add_2)

        import numpy as np

        print("res_ref is: {}".format(res_ref))
        print("res is: {}".format(res))
        print(torch.allclose(res_ref, res, rtol=0.1, atol=0.1))
        
        np.testing.assert_array_almost_equal(res_ref.cpu().numpy(),
        res.cpu().numpy(),
        decimal=2)

        #exit(-1)

        # Step3
        print("Start the step3", flush=True)
        qconv_add = torch.ops.quantized.conv2d_add

        qconv_add_res = qconv_add(
            x_q,
            conv1,
            W2_prepack,
            Y_scale,
            Y_zero_point,)
        res2 = torch.dequantize(qconv_add_res)
        print("res2 is: {}".format(res2))
        # np.testing.assert_array_almost_equal(res_ref.cpu().numpy(),
        # res2.cpu().numpy(),
        # decimal=2)
        np.testing.assert_array_almost_equal(add_2.int_repr().cpu().numpy(),
        qconv_add_res.int_repr().cpu().numpy(),
        decimal=0)
        print("add_2 int:{}".format(add_2.int_repr().cpu().numpy()), flush=True)
        print("qconv_add_res int:{}".format(qconv_add_res.int_repr().cpu().numpy()), flush=True)
        # print(add_2)
        np.testing.assert_array_almost_equal(res_ref.cpu().numpy(),
        res2.cpu().numpy(),
        decimal=2)
        print("res_ref is: {}".format(res_ref))
        print("res is: {}".format(res))
        print("res2 is: {}".format(res2))
        print("Finish:{}".format(True))      

if __name__ == "__main__":
    test_pytorch_op()
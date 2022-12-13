import torch
import numpy as np
import random

local_seed = 2022
torch.manual_seed(local_seed) # Set PyTorch seed
np.random.seed(seed=local_seed) # Set Numpy seed
random.seed(local_seed) # Set the Python seed

CALI = True

class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv = torch.nn.Conv2d(64, 64, (3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        # self.conv2 = torch.nn.Conv2d(64, 128, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # self.conv3 = torch.nn.Conv2d(128, 128, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        # x1 = self.conv(x)
        # Here we need conv2 to ensure the first conv's output is int8
        # return self.conv3(x1)
        return self.relu(self.conv(x))

def test_pytorch_op():
    x1 = torch.rand(116, 64, 56, 56)

    input_channel = 64
    output_channel = 64
    kernel_size = (3, 3)
    strides = (1, 1)
    paddings = (1, 1)
    dilations = (1, 1)
    groups = 1
    use_bias = False

    # W = torch.rand(output_channel, input_channel, 3, 3)
    w_q = torch.load('conv_q_weight.pt')
    W = w_q.dequantize()
    bias = torch.load('conv_bias.pt')

    relu = torch.nn.ReLU()
    conv = torch.nn.Conv2d(input_channel, output_channel, kernel_size,
        stride=strides, padding=paddings, bias=use_bias,
        dilation=dilations, groups=groups)

    qconv_relu = torch.ops.quantized.conv2d_relu.new
    qconv_prepack = torch.ops.quantized.conv2d_prepack

    # Update weight of conv and calculate reference result
    conv.weight = torch.nn.Parameter(W, requires_grad=False)
    conv.bias = bias
    res_ref = relu(conv(x1))
    print("res_ref is: {}".format(res_ref))
    
    torch.backends.quantized.engine = "onednn"
    activation_dtype = torch.quint8
    weight_dtype = torch.qint8

    # s8 for weight
    # scale = torch.load('conv_q_weight.pt')
    # zero_point = torch.load('conv1_q_weight.pt')
    # u8 for activation
    x1_scale = 0.003921568393707275
    x1_zero_point = 0

    # u8 for result
    Y_scale = 0.0058631692081689835
    Y_zero_point = 0
 
    x1_q = torch.quantize_per_tensor(
        x1, scale=x1_scale, zero_point=x1_zero_point, dtype=activation_dtype)

    # w_scale = torch.tensor(scale)
    # w_zero_point = torch.tensor(zero_point)
    # w_q = torch.quantize_per_channel(
    #     W, scales=w_scale, zero_points=w_zero_point, axis=0, dtype=weight_dtype)

    W_prepack = qconv_prepack(
        w_q, bias, strides, paddings, dilations, groups)
    # Scale and Zero point should be the output's scale and zero point
    Y_q = qconv_relu(
        x1_q,
        W_prepack,
        Y_scale,
        Y_zero_point)

    res_ref_q = torch.quantize_per_tensor(
        res_ref, scale=Y_scale, zero_point=Y_zero_point, dtype=activation_dtype)
    # print(res_ref_q.int_repr())
    # print("Y_q2.int_repr() is: {}".format(Y_q.int_repr()))
    # print(torch.allclose(res_ref_q.int_repr(), Y_q.int_repr()))
    # assert torch.allclose(res_ref_q.int_repr(), Y_q.int_repr())
    import numpy as np
    np.testing.assert_array_almost_equal(res_ref_q.int_repr().cpu().numpy(),
        Y_q.int_repr().cpu().numpy(),
        decimal=0)


def test_pytorch_module():
    from torch.ao.quantization import QConfigMapping
    import torch.quantization.quantize_fx as quantize_fx
    with torch.no_grad():
        batch_size = 116
        model = SimpleNet().eval()
        x = torch.rand(batch_size, 64, 56, 56)
        example_inputs = (x, )
        res_ref = model(x)
        torch.backends.quantized.engine = 'onednn'
        # qconfig_mapping = QConfigMapping().set_global(torch.quantization.get_default_qconfig('onednn'))
        qconfig = torch.ao.quantization.QConfig(activation=torch.ao.quantization.MinMaxObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.quint8),
                    weight=torch.ao.quantization.PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))
        qconfig_mapping = torch.ao.quantization.QConfigMapping().set_global(qconfig)
        backend_config = torch.ao.quantization.backend_config.onednn.get_onednn_backend_config()

        model_prepared = quantize_fx.prepare_fx(model, qconfig_mapping, example_inputs, backend_config=backend_config)
        # calibrate (not shown)
        # model_prepared(x)
        for i in range(1):
            images = torch.rand(batch_size, 64, 56, 56)
            model_prepared(images)
        # quantize
        model_quantized = quantize_fx.convert_fx(model_prepared, backend_config=backend_config)


        print(model_quantized.conv_input_scale_0.item())
        print(model_quantized.conv_input_zero_point_0.item())
        print("model_quantized is: {}".format(model_quantized), flush=True)

        torch.save(model_quantized.conv.weight(), 'conv_q_weight.pt')
        torch.save(model_quantized.conv.bias(), 'conv_bias.pt')

        print("model_quantized.conv.weight()")
    
        print("Finish Conver, before jit", flush=True)

        model_quantized = torch.jit.trace(model_quantized, x)
        model_quantized = torch.jit.freeze(model_quantized.eval())

        print("Finish the jit trace", flush=True)

        for i in range(3):
            print("step: {}".format(i), flush=True)
            model_quantized(x)
        print(model_quantized.graph_for(x), flush=True)

        print("Final run", flush=True)
        res = model_quantized(x)

        # print("res is:{}".format(res))
        
        print(torch.allclose(res_ref, res, rtol=0.08, atol=0.01))
        assert torch.allclose(res_ref, res, rtol=0.08, atol=0.01)
        
    
def test_ipex_module():
    import intel_extension_for_pytorch as ipex
    from intel_extension_for_pytorch.quantization import prepare, convert
    from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, QConfig
    batch_size = 116
    model = SimpleNet().eval()
    x = torch.rand(batch_size, 64, 56, 56)
    res_ref = model(x)  
    qconfig = QConfig(
            activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.quint8),
            weight= PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))
    prepared_model = ipex.quantization.prepare(model, qconfig, x, inplace=True)
    if CALI:
        with torch.no_grad():
            for i in range(1):
                images = torch.rand(batch_size, 64, 56, 56)
                prepared_model(images)
            prepared_model.save_qconf_summary("./ipex_cali.json")
            model = ipex.quantization.convert(prepared_model)
            model = torch.jit.trace(model, x)
            model = torch.jit.freeze(model.eval())
            for i in range(3):
                model(x)
            print(model.graph_for(x), flush=True)
            print("Finish Print the graph", flush=True)
            model(x)
    else:
        prepared_model.load_qconf_summary(qconf_summary="./ipex_cali.json")
        model = ipex.quantization.convert(prepared_model)
        with torch.no_grad():
            model = torch.jit.trace(model, x)
            model = torch.jit.freeze(model.eval())

            for i in range(3):
                model(x)
            print(model.graph_for(x), flush=True)
            print("Finish Print the graph", flush=True)
            res = model(x)
            print(torch.allclose(res_ref, res, rtol=0.08, atol=0.01))
            assert torch.allclose(res_ref, res, rtol=0.08, atol=0.01)

if __name__ == "__main__":
    test_pytorch_op()
    #test_pytorch_module()
    # test_ipex_module()
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
        self.conv = torch.nn.Conv2d(3, 9, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.conv2 = torch.nn.Conv2d(64, 128, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.conv3 = torch.nn.Conv2d(128, 128, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        # x1 = self.conv(x)
        # Here we need conv2 to ensure the first conv's output is int8
        # return self.conv3(x1)
        return self.conv(x)

def test_pytorch_op():
    x1 = torch.rand(1, 64, 3, 3)

    input_channel = 64
    output_channel = 128
    kernel_size = (3, 3)
    strides = (2, 2)
    paddings = (1, 1)
    dilations = (1, 1)
    groups = 1
    use_bias = False

    W = torch.rand(output_channel, input_channel, 3, 3)

    conv = torch.nn.Conv2d(input_channel, output_channel, kernel_size,
        stride=strides, padding=paddings, bias=use_bias,
        dilation=dilations, groups=groups)

    qconv = torch.ops.quantized.conv2d
    qconv_prepack = torch.ops.quantized.conv2d_prepack

    # Update weight of conv and calculate reference result
    conv.weight = torch.nn.Parameter(W, requires_grad=False)
    conv.bias = None
    res_ref = conv(x1)
    print("res_ref is: {}".format(res_ref))
    
    torch.backends.quantized.engine = "onednn"
    activation_dtype = torch.quint8
    weight_dtype = torch.qint8

    # s8 for weight
    # w_scale = 0.02
    # w_zero_point = 0
    scale = [
        0.00032623024890199304,
        0.00032610056223347783,
        0.0003258503566030413,
        0.0003267894790042192,
        0.0003266672429163009,
        0.0003267731808591634,
        0.0003262246318627149,
        0.0003262299869675189,
        0.00032614244264550507,
        0.0003267397696617991,
        0.00032667434425093234,
        0.0003260831581428647,
        0.00032632940565235913,
        0.00032626892789267004,
        0.000325956498272717,
        0.0003265547857154161,
        0.00032646561157889664,
        0.00032667466439306736,
        0.0003264119441155344,
        0.0003264650877099484,
        0.0003259920049458742,
        0.000326441164361313,
        0.0003266309795435518,
        0.000326590146869421,
        0.00032615632517263293,
        0.00032667384948581457,
        0.00032630370697006583,
        0.0003263168619014323,
        0.00032666255719959736,
        0.0003266658750362694,
        0.00032587905297987163,
        0.0003260243101976812,
        0.00032658822601661086,
        0.0003267765569034964,
        0.00032679378637112677,
        0.00032570853363722563,
        0.0003255243937019259,
        0.00032678042771294713,
        0.0003263588878326118,
        0.0003261262027081102,
        0.0003256305353716016,
        0.00032613336225040257,
        0.00032616526004858315,
        0.0003260298690292984,
        0.00032622599974274635,
        0.00032509947777725756,
        0.0003267516731284559,
        0.00032631191425025463,
        0.0003266413405071944,
        0.0003259998338762671,
        0.0003264543483965099,
        0.00032653840025886893,
        0.0003267934371251613,
        0.0003267768188379705,
        0.00032611057395115495,
        0.0003265859850216657,
        0.0003267605497967452,
        0.0003246756095904857,
        0.0003254342300351709,
        0.00032676546834409237,
        0.00032573798671364784,
        0.00032648674095980823,
        0.0003266383719164878,
        0.00032673272653482854,
        0.0003263896214775741,
        0.00032669861684553325,
        0.0003264891274739057,
        0.00032569983159191906,
        0.0003265064733568579,
        0.00032675519469194114,
        0.00032580349943600595,
        0.000326719309668988,
        0.00032613310031592846,
        0.0003267042920924723,
        0.00032472138991579413,
        0.00032325286883860826,
        0.0003266300482209772,
        0.000325141561916098,
        0.00032664998434484005,
        0.0003240394580643624,
        0.00032485753763467073,
        0.0003260478551965207,
        0.0003265526902396232,
        0.0003264379920437932,
        0.0003260138037148863,
        0.0003254131297580898,
        0.0003252144088037312,
        0.00032566318986937404,
        0.00032670435030013323,
        0.00032591292983852327,
        0.0003260201192460954,
        0.0003263389808125794,
        0.0003261452657170594,
        0.0003261197707615793,
        0.0003264539991505444,
        0.0003267527208663523,
        0.0003264001861680299,
        0.0003263349353801459,
        0.00032562727574259043,
        0.00032580364495515823,
        0.0003251698799431324,
        0.00032672483939677477,
        0.00032665792969055474,
        0.000326089357258752,
        0.00032497019856236875,
        0.0003248398716095835,
        0.0003265540290158242,
        0.0003259743098169565,
        0.0003263389808125794,
        0.0003259332734160125,
        0.0003264256229158491,
        0.00032635845127515495,
        0.0003267133724875748,
        0.00032550899777561426,
        0.000326614361256361,
        0.00032577765523456037,
        0.00032631741487421095,
        0.0003264259430579841,
        0.00032635495881550014,
        0.0003265782434027642,
        0.0003244909457862377,
        0.00032619189005345106,
        0.00032631048816256225,
        0.00032651223591528833,
        0.0003264077240601182,
        0.0003266475978307426,
        0.0003247523563914001,
        0.0003266796702519059
    ]
    zero_point = [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
    ]
    # u8 for activation
    x1_scale = 0.007839311845600605
    x1_zero_point = 128

    # u8 for result
    Y_scale = 0.005602228455245495
    Y_zero_point = 128
 
    x1_q = torch.quantize_per_tensor(
        x1, scale=x1_scale, zero_point=x1_zero_point, dtype=activation_dtype)

    w_scale = torch.tensor(scale)
    w_zero_point = torch.tensor(zero_point)
    w_q = torch.quantize_per_channel(
        W, scales=w_scale, zero_points=w_zero_point, axis=0, dtype=weight_dtype)

    print("---- start to do prepack ----", flush=True)
    W_prepack = qconv_prepack(
        w_q, None, strides, paddings, dilations, groups)
    print("---- finish the prepack ----", flush=True)
    # Scale and Zero point should be the output's scale and zero point
    Y_q = qconv(
        x1_q,
        W_prepack,
        Y_scale,
        Y_zero_point)

    print("---- start step2 calculation ----", flush=True)
    _ = qconv(
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
        batch_size = 1
        model = SimpleNet().eval()
        x = torch.rand(batch_size, 3, 3, 3)
        example_inputs = (x, )
        res_ref = model(x)
        torch.backends.quantized.engine = 'onednn'
        qconfig = torch.ao.quantization.QConfig(
                        activation=torch.ao.quantization.MinMaxObserver.with_args(
                            reduce_range=False,
                            dtype=torch.quint8),
                        weight=torch.ao.quantization.default_per_channel_weight_observer)
        # qconfig_mapping = QConfigMapping().set_global(torch.quantization.get_default_qconfig('onednn'))
        qconfig_mapping = QConfigMapping().set_global(qconfig)
        backend_config = torch.ao.quantization.backend_config.onednn.get_onednn_backend_config()
        model_prepared = quantize_fx.prepare_fx(model, qconfig_mapping, example_inputs, backend_config=backend_config)
        # calibrate (not shown)
        # model_prepared(x)
        for i in range(1):
            # images = torch.rand(batch_size, 64, 3, 3)
            model_prepared(x)
        # quantize
        model_quantized = quantize_fx.convert_fx(model_prepared, backend_config=backend_config)

        model_quantized = torch.jit.trace(model_quantized, x)
        model_quantized = torch.jit.freeze(model_quantized.eval())

        for i in range(3):
            model_quantized(x)
        print(model_quantized.graph_for(x), flush=True)

        print("Final run")
        res = model_quantized(x)

        print("res_ref is:{}".format(res_ref), flush=True)
        print("res is:{}".format(res), flush=True)
        
        print(torch.allclose(res_ref, res, rtol=0.08, atol=0.01))
        assert torch.allclose(res_ref, res, rtol=0.08, atol=0.01)

if __name__ == "__main__":
    # test_pytorch_op()
    test_pytorch_module()
import torch
import numpy as np
import random

local_seed = 2022
torch.manual_seed(local_seed) # Set PyTorch seed
np.random.seed(seed=local_seed) # Set Numpy seed
random.seed(local_seed) # Set the Python seed

CALI = False

class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv = torch.nn.Conv2d(64, 128, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.conv2 = torch.nn.Conv2d(64, 128, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.conv3 = torch.nn.Conv2d(128, 128, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x1 = self.relu(self.conv(x))
        # Here we need conv2 to ensure the first conv's output is int8
        return x1

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

    relu = torch.nn.ReLU()
    conv = torch.nn.Conv2d(input_channel, output_channel, kernel_size,
        stride=strides, padding=paddings, bias=use_bias,
        dilation=dilations, groups=groups)

    qconv_relu = torch.ops.quantized.conv2d_relu.new
    qconv_prepack = torch.ops.quantized.conv2d_prepack

    # Update weight of conv and calculate reference result
    conv.weight = torch.nn.Parameter(W, requires_grad=False)
    conv.bias = None
    res_ref = relu(conv(x1))
    print("res_ref is: {}".format(res_ref))
    
    torch.backends.quantized.engine = "onednn"
    activation_dtype = torch.quint8
    weight_dtype = torch.qint8

    # s8 for weight
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

    W_prepack = qconv_prepack(
        w_q, None, strides, paddings, dilations, groups)
    # Scale and Zero point should be the output's scale and zero point
    Y_q = qconv_relu(
        x1_q,
        W_prepack,
        Y_scale,
        Y_zero_point)

    print("start step2 calculation")
    _ = qconv_relu(
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
        x = torch.rand(batch_size, 64, 3, 3)
        example_inputs = (x, )
        res_ref = model(x)
        torch.backends.quantized.engine = 'onednn'
        # qconfig_mapping = QConfigMapping().set_global(torch.quantization.get_default_qconfig('onednn'))
        qconfig_mapping = torch.ao.quantization.get_default_qconfig_mapping(backend="onednn")
        backend_config = torch.ao.quantization.backend_config.onednn.get_onednn_backend_config()
        model_prepared = quantize_fx.prepare_fx(model, qconfig_mapping, example_inputs, backend_config=backend_config)
        # calibrate (not shown)
        # model_prepared(x)
        print("model_prepared is: {}".format(model_prepared), flush=True)
        print(type(model_prepared.conv))
        for i in range(1):
            images = torch.rand(batch_size, 64, 3, 3)
            model_prepared(images)
        # quantize
        model_quantized = quantize_fx.convert_fx(model_prepared, backend_config=backend_config)

        print("model_quantized is: {}".format(model_quantized), flush=True)

        model_quantized = torch.jit.trace(model_quantized, x)
        model_quantized = torch.jit.freeze(model_quantized.eval())

        for i in range(3):
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
    batch_size = 1
    model = SimpleNet().eval()
    x = torch.rand(batch_size, 64, 3, 3)
    res_ref = model(x)  
    qconfig = QConfig(
            activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.quint8),
            weight= PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))
    prepared_model = ipex.quantization.prepare(model, qconfig, x, inplace=True)
    if CALI:
        with torch.no_grad():
            for i in range(1):
                images = torch.rand(batch_size, 64, 3, 3)
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
    # test_pytorch_op()
    test_pytorch_module()
    # test_ipex_module()
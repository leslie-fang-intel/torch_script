import torch
import numpy as np
import random

local_seed = 2022
torch.manual_seed(local_seed) # Set PyTorch seed
np.random.seed(seed=local_seed) # Set Numpy seed
random.seed(local_seed) # Set the Python seed

CALI = True

class SimpleNet(torch.nn.Module):
    def __init__(self, with_bias=False):
        super(SimpleNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(64, 128, (3, 3), stride=(2, 2), padding=(1, 1), bias=with_bias)
        self.conv2 = torch.nn.Conv2d(64, 128, (3, 3), stride=(2, 2), padding=(1, 1), bias=with_bias)
        # self.conv3 = torch.nn.Conv2d(128, 128, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.relu = torch.nn.ReLU()

        self.bn = torch.nn.BatchNorm2d(128)
    def forward(self, x):
        #x = torch.add(x, x)
        
        #x1 = self.relu(torch.add(self.conv1(x), self.conv2(x)))
        #x1 = torch.add(self.conv1(x), self.conv2(x))
        
        
        # # Here we need conv2 to ensure the first conv's output is int8
        # print("x1 is: {}".format(x1))
        # # return self.conv3(x1)
        
        
        x1 = self.bn(self.conv1(x)) + self.conv2(x)
        #x1 = self.conv1(x) + self.conv2(x)
        return x1
        #return torch.add(x, x)

def test_pytorch_module():
    from torch.ao.quantization import QConfigMapping
    import torch.quantization.quantize_fx as quantize_fx
    import itertools
    options = itertools.product(
        [True, False], # bias
        [True, False], # 对称或者非对称量化
    )
    options = itertools.product(
        [False], # bias
        [True], # 对称或者非对称量化
    )
    for (with_bias, sys_quant) in options:
        print("Start new test case", flush=True)
        print("with_bias is: {}".format(with_bias), flush=True)
        print("sys_quant is: {}".format(sys_quant), flush=True)
        with torch.no_grad():
            batch_size = 1
            model = SimpleNet(with_bias).eval()

            x = torch.rand(batch_size, 64, 3, 3)
            example_inputs = (x, )
            res_ref = model(x)

            torch.backends.quantized.engine = 'onednn'
            # input default oneDNN observer
            # qconfig_mapping = QConfigMapping().set_global(torch.quantization.get_default_qconfig('onednn'))
            if sys_quant:
                # input MinMaxObserver 对称量化
                qconfig = torch.ao.quantization.QConfig(activation=torch.ao.quantization.MinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.quint8),
                            weight=torch.ao.quantization.PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))
                qconfig_mapping = QConfigMapping().set_global(qconfig)
            else:
                # input MinMaxObserver 非对称量化
                qconfig = torch.ao.quantization.QConfig(activation=torch.ao.quantization.MinMaxObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.quint8),
                        weight=torch.ao.quantization.PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))
                qconfig_mapping = QConfigMapping().set_global(qconfig)

            # # qconfig_mapping = torch.ao.quantization.get_default_qconfig_mapping('onednn')
            backend_config = torch.ao.quantization.backend_config.onednn.get_onednn_backend_config()

            # torch.backends.quantized.engine = 'fbgemm'
            # qconfig = torch.ao.quantization.QConfig(activation=torch.ao.quantization.MinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.quint8),
            #                weight=torch.ao.quantization.PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))
            # qconfig_mapping = QConfigMapping().set_global(qconfig)
            # # Default use HistogramObserver for activation, I find accuracy not good for calbriation only 1 iteration
            # # qconfig_mapping = torch.ao.quantization.get_default_qconfig_mapping('fbgemm')
            # backend_config = torch.ao.quantization.backend_config.fbgemm.get_fbgemm_backend_config()

            print("model before prepare is: {}".format(model), flush=True)
            model_prepared = quantize_fx.prepare_fx(model, qconfig_mapping, example_inputs, backend_config=backend_config)
            
            
            # calibrate (not shown)
            print("model_prepared is: {}".format(model_prepared), flush=True)

            res_prepared = model_prepared(x)
            # print(torch.allclose(res_ref, res_prepared, rtol=0.08, atol=0.01), flush=True)
            for i in range(10):
                #images = torch.rand(batch_size, 64, 3, 3)
                model_prepared(x)
            # quantize
            model_quantized = quantize_fx.convert_fx(model_prepared, backend_config=backend_config)

            print("model_quantized is: {}".format(model_quantized), flush=True)

            res_quantized = model_quantized(x)
            # print(torch.allclose(res_ref, res_quantized, rtol=0.08, atol=0.01), flush=True)
            import numpy as np
            np.testing.assert_array_almost_equal(res_ref.cpu().numpy(),
            res_quantized.cpu().numpy(), decimal=2)

            exit(-1)
            # print("res_quantized is: {}".format(res_quantized), flush=True)
            
            model_quantized = torch.jit.trace(model_quantized, x)
            model_quantized = torch.jit.freeze(model_quantized.eval())

            for i in range(3):
                model_quantized(x)
            print(model_quantized.graph_for(x), flush=True)

            # print("Final run", flush=True)
            res_quantized_jit = model_quantized(x)

            # print(torch.allclose(res_ref, res_quantized_jit, rtol=0.08, atol=0.01), flush=True)
            # assert torch.allclose(res_ref, res, rtol=0.08, atol=0.01)
            import numpy as np
            np.testing.assert_array_almost_equal(res_ref.cpu().numpy(),
                res_quantized_jit.cpu().numpy(),
                decimal=2)
            # print("res_ref is: {}".format(res_ref), flush=True)
            # print("res_quantized_jit is: {}".format(res_quantized_jit), flush=True)
    print("Finish:{}".format(True), flush=True)


if __name__ == "__main__":
    test_pytorch_module()

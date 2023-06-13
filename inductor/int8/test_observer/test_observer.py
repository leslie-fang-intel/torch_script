import torch
from torch.ao.quantization.observer import HistogramObserver

def test1():
    pass


def test2():
    obser = HistogramObserver.with_args(reduce_range=False)()
    mask = torch.tensor([-3.4028234663852886 * 10**38, -0.0])
    mask2 = torch.tensor([-0.0, -3.4028234663852886 * 10**38])

    obser(mask)
    obser(mask2)

    mask3 = torch.tensor([-3.4028234663852886 * 10**38, 1.0, 2])
    obser(mask3)
    scale, zp = obser.calculate_qparams()
    print("scale is: {}".format(scale))
    print("zp is: {}".format(zp))


    input_mask = torch.tensor([-3.4028234663852886 * 10**38, 0.0, 0.0])
    input = torch.tensor([2.1, 3.2, 4.1])
    ref_result = torch.softmax(input + input_mask, dim=0)
    print("ref_result is: {}".format(ref_result))

    # print("ref_result is: {}".format(torch.softmax(torch.tensor([-3.4028234663852886 * 10**38, 3.2, 4.1]), dim=0)))

    print("mask is: {}".format(mask))

    # scale = 3.4028234663852886 * 10**38 / 255
    # zp = 255

    quant_tensor = torch.quantize_per_tensor(input_mask, scale, zp, torch.quint8)
    print("quant_tensor is: {}".format(quant_tensor))
    dequant_tensor = quant_tensor.dequantize()
    print("dequant_tensor is: {}".format(dequant_tensor))


    result = torch.softmax(input + dequant_tensor, dim=0)
    print("result is: {}".format(result))

if __name__ == "__main__":
    test1()
    test2()
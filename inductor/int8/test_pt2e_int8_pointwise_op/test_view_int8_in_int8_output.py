import torch
import torch.ao.quantization.fx._decomposed

def raw_test():
    x = torch.randint(3, 5, (2, 3), dtype=torch.uint8)
    print("x is: {}".format(x), flush=True)
    y = torch.ops.aten.view.default(x, (1, 6))
    print("y is: {}".format(y), flush=True)

def quant_test():
    x = torch.rand((2, 3), dtype=torch.float)
    print("x is: {}".format(x), flush=True)

    y = torch.ops.aten.view.default(x, (1, 6))

    _scale_0 = 0.01
    _zero_point_0 = 123
    quantize_x = torch.ops.quantized_decomposed.quantize_per_tensor(x, _scale_0, _zero_point_0, 0, 255, torch.uint8)
    print("quantize_x is: {}".format(quantize_x), flush=True)
    quantize_y = torch.ops.aten.view.default(quantize_x, (1, 6))
    print("quantize_y is: {}".format(quantize_y), flush=True)
    dequant_y = torch.ops.quantized_decomposed.dequantize_per_tensor(quantize_y, _scale_0, _zero_point_0, 0, 255, torch.uint8)

    print("y is: {}".format(y), flush=True) 
    print("dequant_y is: {}".format(dequant_y), flush=True)

    print(torch.allclose(y, dequant_y, rtol=0.01, atol=0.01))

if __name__ == "__main__":
    # raw_test()
    quant_test()

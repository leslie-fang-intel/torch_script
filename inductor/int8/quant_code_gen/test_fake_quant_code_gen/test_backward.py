import torch
from torch.library import Library, impl
import torch.ao.quantization.fx._decomposed

def qdq(input, scales, zero_points, axis, quant_min, quant_max, dtype, label):
    res = torch.ops.quantized_decomposed.test_backward(
        input, scales, zero_points, axis, quant_min, quant_max, dtype
    )
    # res = input * scales + zero_points
    res.sum().backward()
    # loss_fn = torch.nn.MSELoss()
    # loss = loss_fn(res, label)
    # loss.backward()

    return res, input.grad

def test_eager():
    print("--- start test_eager----", flush=True)
    device = torch.device("cpu")
    input = torch.randn(1, 3, 224, 224, requires_grad=True)
    scales = torch.ones((1, 3, 224, 224,))
    zero_points = torch.zeros((1, 3, 224, 224,))
    # scales = torch.ones((3,))
    # zero_points = torch.zeros((3,))
    axis = 1
    quant_min = -128
    quant_max = 127
    dtype = torch.int8
    label = torch.randn(1, 3, 224, 224)
    res, input_grad = qdq(input, scales, zero_points, axis, quant_min, quant_max, dtype, label)
    print("input_grad is: {}".format(input_grad), flush=True)

def test_compile():
    print("--- start test_compile----", flush=True)
    device = torch.device("cpu")
    input = torch.randn(1, 3, 224, 224, requires_grad=True)
    scales = torch.ones((1, 3, 224, 224,))
    zero_points = torch.zeros((1, 3, 224, 224,))
    # scales = torch.ones((3,))
    # zero_points = torch.zeros((3,))
    axis = 1
    quant_min = -128
    quant_max = 127
    dtype = torch.int8
    label = torch.randn(1, 3, 224, 224)
    compiled_qdq = torch.compile(qdq)
    res, input_grad = compiled_qdq(input, scales, zero_points, axis, quant_min, quant_max, dtype, label)
    print("input_grad is: {}".format(input_grad), flush=True)

if __name__ == "__main__":
    test_eager()  # Success
    test_compile()  # Failed

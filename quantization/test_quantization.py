import torch

if __name__ == "__main__":
    float_tensor = torch.randn(2, 2, 3)
    # print("float_tensor is: {}".format(float_tensor))

    scale, zero_point = 1.0, 1
    dtype = torch.qint8
    q_per_tensor = torch.quantize_per_tensor(float_tensor, scale, zero_point, dtype)
    # print("q_per_tensor is: {}".format(q_per_tensor.dtype))
    # print("q_per_tensor is: {}".format(q_per_tensor.dequantize()))
    # print(q_per_tensor[1, 1, 2])
    # print(type(q_per_tensor))
    # import pdb;pdb.set_trace()
    print(q_per_tensor)

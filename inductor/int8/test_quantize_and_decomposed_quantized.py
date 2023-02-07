import torch
import torch.ao.quantization.fx._decomposed

if __name__ == "__main__":
    input_tensor = torch.randn(1, 3, 2, 2)
    scale = 0.25
    zp = 0
    dtype = torch.uint8

    # import pdb;pdb.set_trace()
    q_tensor = torch.quantize_per_tensor(input_tensor, scale, zp, torch.quint8)
    decomposed_q_tensor = torch.ops.quantized_decomposed.quantize_per_tensor(input_tensor, scale, zp, 0, 255, torch.uint8)

    torch._make_per_tensor_quantized_tensor(q_tensor.int_repr(), 1.2, 1)

    print(type(q_tensor), flush=True)
    print(type(decomposed_q_tensor), flush=True)


    print(q_tensor, flush=True)
    print(q_tensor.int_repr(), flush=True)
    print(decomposed_q_tensor, flush=True)
    # q_tensor.int_repr() should be same as decomposed_q_tensor


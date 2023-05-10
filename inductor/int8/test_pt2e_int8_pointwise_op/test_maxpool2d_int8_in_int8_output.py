import torch
import torch.ao.quantization.fx._decomposed

def raw_test():
    x = torch.randint(3, 5, (1, 2, 14, 14), dtype=torch.uint8)
    print("x is: {}".format(x), flush=True)
    y = torch.ops.aten.max_pool2d_with_indices.default(x, [1, 1], [1, 1])[0]
    # RuntimeError: "max_pool2d" not implemented for 'Byte'
    # maxpool should be supported by oneDNN
    print("y is: {}".format(y), flush=True)

if __name__ == "__main__":
    raw_test()
    # quant_test()

import torch

if __name__ == "__main__":
    # tensor creation with `torch.zeros`
    x = torch.zeros(4, dtype=torch.float8_e4m3fn)
    print(x)

    # conversion to and from float8 dtypes
    # conversion between float32 and float8 uses the kernels copied from fbgemm
    # conversion between other dtypes and float8 converts to float32 as an intermediate step
    x = torch.randn(4, dtype=torch.float)
    x_float8 = x.to(torch.float8_e4m3fn)
    x_float32 = x_float8.to(torch.float32)

    # printing out the tensor for debugging
    # note: the values are converted to float32 before being printed
    print(x_float8)


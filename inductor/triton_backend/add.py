import torch

def fn(a, b):
    return a + b

if __name__ == "__main__":
    cfn = torch.compile(fn)
    a = torch.randn(2048)
    b = torch.randn(2048)
    cfn(a,b)
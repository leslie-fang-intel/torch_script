import torch

def test(x):
    tokens_per_expert = x.sum(dim=0)
    if tokens_per_expert>10:
        return x * x
    else:
        return x - x/2

if __name__ == "__main__":
    # test()
    cfn = torch.compile(test)
    x = torch.randn(12)
    # x2 = torch.randn(8)
    # print(x.size(), flush=True)
    y = cfn(x)
    # y = cfn(x2)
    print("y is: {}".format(y), flush=True)
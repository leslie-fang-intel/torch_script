import torch

def qdq(input,):
    res = torch.ge(input, -128)
    return res

if __name__ == "__main__":
    input = torch.randn(2, 3, 224, 224)
    res_ref = qdq(input)
    c_fn = torch.compile(qdq)

    res = c_fn(input)

    # print("res_ref is: {}".format(res_ref), flush=True)
    # print("res is: {}".format(res), flush=True)

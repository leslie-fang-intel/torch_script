import torch

def test_pytorch_round():
    input = torch.ones((16)) * 4.4
    res = torch.round(input)
    print("input is: {}; round res is: {}".format(input, res), flush=True)
    input = torch.ones((16)) * 4.6
    res = torch.round(input)
    print("input is: {}; round res is: {}".format(input, res), flush=True)
    input = torch.ones((16)) * 4.5
    res = torch.round(input)
    print("input is: {}; round res is: {}".format(input, res), flush=True)
    input = torch.ones((16)) * 5.5
    res = torch.round(input)
    print("input is: {}; round res is: {}".format(input, res), flush=True)

def test_pytorch_to_int32():
    input = torch.ones((16)) * 4.4
    res = input.to(torch.int32)
    print("input is: {}; round res is: {}".format(input, res), flush=True)
    input = torch.ones((16)) * 4.6
    res = input.to(torch.int32)
    print("input is: {}; round res is: {}".format(input, res), flush=True)
    input = torch.ones((16)) * 4.5
    res = input.to(torch.int32)
    print("input is: {}; round res is: {}".format(input, res), flush=True)
    input = torch.ones((16)) * 5.5
    res = input.to(torch.int32)
    print("input is: {}; round res is: {}".format(input, res), flush=True)

def test_pytorch_to_uint8():
    input = torch.ones((16)) * 4.4
    res = input.to(torch.uint8)
    print("input is: {}; round res is: {}".format(input, res), flush=True)
    input = torch.ones((16)) * 4.6
    res = input.to(torch.uint8)
    print("input is: {}; round res is: {}".format(input, res), flush=True)
    input = torch.ones((16)) * 4.5
    res = input.to(torch.uint8)
    print("input is: {}; round res is: {}".format(input, res), flush=True)
    input = torch.ones((16)) * 5.5
    res = input.to(torch.uint8)
    print("input is: {}; round res is: {}".format(input, res), flush=True)

if __name__ == "__main__":
    # test_pytorch_round
    # test_pytorch_to_int32()
    test_pytorch_to_uint8()
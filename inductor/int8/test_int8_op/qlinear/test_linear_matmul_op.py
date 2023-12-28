# CMD: python test_linear_matmul_op.py

import torch

if __name__ == "__main__":
    linear = torch.nn.Linear(320, 1024)
    weight = linear.weight
    print("weight.size is: {}".format(weight.size()), flush=True)
    input = torch.randn((2, 320, 320))

    res_linear = torch.nn.functional.linear(input, weight)
    print("res_linear.size() is: {}".format(res_linear.size()), flush=True)
    
    # matmul_tensor2 = weight.t().view(1, 320, 1024).expand(2, -1, -1)
    matmul_tensor2 = weight.t().expand(2, 320, 1024)
    
    
    res_matmul = torch.matmul(input, matmul_tensor2)# 
    print("res_matmul.size() is: {}".format(res_matmul.size()), flush=True)
    print(torch.allclose(res_linear, res_matmul))


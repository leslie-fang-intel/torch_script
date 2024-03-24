
import torch

def matrix_inner_product(_matrix_a, _matrix_b):
    matrix_c = torch.zeros(_matrix_a.size(0), _matrix_a.size(1))
    for i in range(16):
        for j in range(16):
            accum = 0
            for k in range(16):
                accum += _matrix_a[i, k] * _matrix_b[k, j]
            matrix_c[i, j] = accum
    return matrix_c


def matrix_outer_product(_matrix_a, _matrix_b):
    matrix_c = torch.zeros(_matrix_a.size(0), _matrix_a.size(1))
    for k in range(16):
        v1 = _matrix_a[:, k] # 矩阵 A 的一列
        v2 = _matrix_b[k, :] # 矩阵 B 的一行
        matrix_c += torch.outer(v1, v2)  # 两者做外积
    return matrix_c

if __name__ == "__main__":

    matrix_a = torch.randint(1, 5, (16, 16))
    matrix_b = torch.randint(1, 5, (16, 16))

    matrix_c_ref = torch.matmul(matrix_a, matrix_b).to(torch.float32)

    matrix_c_inner_product = matrix_inner_product(matrix_a, matrix_b)
    matrix_c_outer_product = matrix_inner_product(matrix_a, matrix_b)

    print(torch.allclose(matrix_c_ref, matrix_c_inner_product), flush=True)
    print(torch.allclose(matrix_c_ref, matrix_c_outer_product), flush=True)
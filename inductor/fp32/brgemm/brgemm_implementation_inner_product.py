
import torch

def naive_batch_reduced_gemm(_matrix_a, _matrix_b):
    # refer to https://arxiv.org/abs/1906.06440 的 伪代码1
    matrix_c = torch.zeros(_matrix_a.size(0), _matrix_b.size(1))
    assert _matrix_a.size(1) == _matrix_b.size(0)
    # m * n 组成了 dst 一个tile 的大小
    # m * k 组成了 matrix_a 一个tile 的大小
    # k * n 组成了 matrix_b 一个tile 的大小
    n = 16 # 8
    m = 16 # 4
    k = 16 # 8
    N = _matrix_a.size(1) // k
    for i_n in range(0, _matrix_a.size(0), n):
         for i_m in range(0, _matrix_b.size(1), m):
            # 一次计算 dst 矩阵的 一个16X16小块 的结果
            acc_regs = matrix_c[(i_n) : (i_n + n), (i_m):(i_m + m)]
            for i in range(int(N)):
                acc_regs += torch.matmul(
                    _matrix_a[(i_n):(i_n + n), (i * k) : (i * k + k)],
                    _matrix_b[(i * k) : (i * k + k), (i_m):(i_m + m)],
                )
            matrix_c[(i_n):(i_n + n), (i_m):(i_m + m)] = acc_regs
    return matrix_c

if __name__ == "__main__":
    matrix_a_shapes = [(16, 16), (32, 16), (32, 16), (32, 32), (32, 64), (128, 32)]
    matrix_b_shapes = [(16, 16), (16, 16), (16, 32), (32, 32), (64, 128), (32, 32)]

    for matrix_a_shape, matrix_b_shape in zip(matrix_a_shapes, matrix_b_shapes):
        matrix_a = torch.randint(1, 5, matrix_a_shape)
        matrix_b = torch.randint(1, 5, matrix_b_shape)
        refer_matrix_c = torch.matmul(matrix_a, matrix_b).to(torch.float32)
        matrix_c = naive_batch_reduced_gemm(matrix_a, matrix_b)
        # print("refer_matrix_c is: {}".format(refer_matrix_c), flush=True)
        # print("matrix_c is: {}".format(matrix_c), flush=True)
        print(torch.allclose(refer_matrix_c, matrix_c), flush=True)
## 矩阵内积与外积
矩阵相乘可以通过 内积 或者 外积的方式来实现
参考： test_matrix_inner_outer_product.py 关于 内积和外积的实现

## BRGEMM 实现
* 参考论文 https://arxiv.org/abs/1906.06440

### 基本思路
* 将大矩阵分块
    * 比如一个32X32 矩阵A 和 32X32 矩阵B 相乘得到 32X32 的矩阵C
        * 16X16 为1个小矩阵来划分
        * 所以退化成
            ```
            matmul ( (A11，A12)  (B11，B12))
                     (A21，A22), (B21，B22)
            ```
            Aij， Bij 都是 16X16的矩阵
            * 外部大矩阵可以用内积也可用外积表达
                用内积实现表达为
                * `C11 = matmul(A11, B11) + matmul(A12, B21)`
                * `C12 = matmul(A11, B12) + matmul(A12, B22)` 
                * `C21 = matmul(A21, B11) + matmul(A22, B21)` 
                * `C22 = matmul(A21, B12) + matmul(A22, B22)`
            * 内部小矩阵相乘`matmul(A11, B11)`, 也可以用内积或者外积来实现
* 小矩阵的乘法可以利用外积，也可以用内积 (两者等价)
    * 论文中用的是外积，实现参考：brgemm_implementation_outer_product.py
    * 也可以用内积来实现（实际AMX的指令用的是内积实现），实现参考：brgemm_implementation_inner_product.py
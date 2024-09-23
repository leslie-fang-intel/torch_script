## Used to tune the correcness of Certain GEMM implementation
* u8s8f32
    * since weight is symmetric quant, the weight zero point is 0 
* Reference result 1: Doing matmul with FP32
* Reference result 2: Doing u8s8f32 with inner product
    * We can compare the new gemm implement with the int32 output of inner product
* How to tune
    * Change the M,N,K in this test file
    * Copy new GEMM pack and implementation (corresponding to this M,N,K) and test the result

```
rm -rf /tmp/torchinductor_leslie/* && clear && numactl -C 56-111 -m 1 python int8_linear_test_case.py
```

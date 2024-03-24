// Build: /usr/local/cuda-12.1/bin/nvcc test_hello_world.cu
#include <stdio.h>
#include <cuda.h>

#define N 1024

__global__ void dkernel() {

    printf("dkernel thread is: %d \n", threadIdx.x);

}

__global__ void dkernel2() {

    printf("dkernel2 thread is: %d \n", threadIdx.x);

}

int main() {

    cudaStream_t s1, s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);

    // 第四个 参数表示为 stream number
    // 这样，Kernel1 和 Kernel2 的执行会有重叠
    dkernel<<<1, N, 0, s1>>>();
    dkernel2<<<1, N, 0, s2>>>();

    cudaDeviceSynchronize();

    return 0;
}

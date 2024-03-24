// Build: /usr/local/cuda-12.1/bin/nvcc test_hello_world.cu
#include <stdio.h>
#include <cuda.h>

#define N 100

__global__ void dkernel() {
    printf("%d \n", threadIdx.x);
}

int main() {
    // 每个线程 都会 执行一遍 同一个函数
    dkernel<<<1,N>>>();
    cudaDeviceSynchronize();
    return 0;
}

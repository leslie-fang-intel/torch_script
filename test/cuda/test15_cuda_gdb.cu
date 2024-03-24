// Build: /usr/local/cuda-12.1/bin/nvcc -g -G test_hello_world.cu
#include <stdio.h>
#include <cuda.h>

#define N 1024

__global__ void dkernel() {

    // share memory 大小 会动态变化
    // 大小定义在 kernel launch的第三个参数上
    extern __shared__ int s[];

    s[threadIdx.x] = threadIdx.x;
    // 一个block内所有线程都会barrier到这里执行
    __syncthreads();
    printf("s is: %d \n", s[threadIdx.x]);

}

int main() {

    // unsigned* dm;
    // cudaMalloc(&dm, N * sizeof(unsigned));

    // 第三个 参数表示为 dynamic shared memory
    dkernel<<<1, N, N * sizeof(int)>>>();

    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    printf("%d, %s, %s \n", error, cudaGetErrorName(error), cudaGetErrorString(error));

    return 0;
}

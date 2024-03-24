// Build: /usr/local/cuda-12.1/bin/nvcc test_hello_world.cu
#include <stdio.h>
#include <cuda.h>

#define N 1024

__global__ void dkernel(unsigned* dm) {

    // 一个block 内 所有的线程 share 同一个 s
    // 不同 block 的线程 会有不同的 s copy
    // s 的初始化值 是未知的
    __shared__ unsigned s;

    // 一个注意点 是 0， 1 线程在一个 warp 内部
    // 所以前面两句话的执行结果 是有保证的
    if (threadIdx.x == 0) s = 0;
    if (threadIdx.x == 1) s += 1;

    // 一个block内所有线程都会barrier到这里执行
    __syncthreads();

    if (threadIdx.x == 100) s += 2;

    __syncthreads();

    if (threadIdx.x == 0) printf("s is: %d \n", s);

}

int main() {

    unsigned* dm;
    cudaMalloc(&dm, N * sizeof(unsigned));

    dkernel<<<1, N>>>(dm);

    cudaDeviceSynchronize();

    return 0;
}

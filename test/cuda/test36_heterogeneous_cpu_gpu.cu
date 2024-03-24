// Build: /usr/local/cuda-12.1/bin/nvcc -g -G test_hello_world.cu
#include <stdio.h>
#include <cuda.h>

#define N 1024

__global__ void dkernel(int* dm) {
    *dm += 1;
    printf("Hello, World! %d\n", *dm);
}

int main() {

    int* dm;
    // Pinned Memory
    // 在CPU 的内存中，分配一块页锁定的内存
    // 页锁定 意味着 Linux 不会将这块内存交换到硬盘上
    cudaHostAlloc(&dm, sizeof(int), 0);

    do {
        dkernel<<<1, 1>>>(dm);
        cudaDeviceSynchronize();
    } while (*dm < 10);


    // Pinned Memory 释放
    // 什么是页锁定
    cudaFreeHost(dm);

    return 0;
}

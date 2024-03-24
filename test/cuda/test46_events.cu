// Build: /usr/local/cuda-12.1/bin/nvcc test_hello_world.cu
#include <stdio.h>
#include <cuda.h>

#define N 1

__global__ void dkernel(int* d) {

    printf("dkernel thread is: %d \n", *d);

}

__global__ void dkernel2() {

    printf("dkernel2 thread is: %d \n", threadIdx.x);

}

int main() {

    int* da;
    int* db;
    cudaMalloc(&da, N * sizeof(int));
    cudaMemset(da, 0, N * sizeof(int));
    cudaMemset(db, 1, N * sizeof(int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaStream_t s1, s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);

    // Event 可以实现 stream 之间的通信
    dkernel<<<1, N, 0, s1>>>(da);
    cudaEventRecord(start, s1);
    dkernel<<<1, N, 0, s1>>>(db);

    // s2 等待 event 执行之后才会执行
    cudaStreamWaitEvent(s2, start, 0);
    dkernel2<<<1, N, 0, s2>>>();

    cudaDeviceSynchronize();

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

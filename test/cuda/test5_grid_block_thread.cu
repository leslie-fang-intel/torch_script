// Build: /usr/local/cuda-12.1/bin/nvcc test_hello_world.cu
#include <stdio.h>
#include <cuda.h>

#define N 100

__global__ void dkernel() {
    // printf();

    unsigned threadId  =
        threadIdx.x
        + threadIdx.y * blockDim.x
        + threadIdx.z * blockDim.x * blockDim.y
        + blockIdx.x * blockDim.x * blockDim.y * blockDim.z
        + blockIdx.y * blockDim.x * blockDim.y * blockDim.z * gridDim.x
        + blockIdx.z * blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y;
    if (threadId == 5000) {
        printf("ThreadID is: %d \n", threadId);
        printf("%d,%d,%d,%d,%d,%d,  \n", blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
    }

}

int main() {

    // A kernel will use a grid of threads to compute
    // Define a grid with 2*3*4 blocks
    dim3 grid(2, 3, 4);
    // Define a block with 5*6*7 threads
    dim3 block(5, 6, 7);

    dkernel<<<grid, block>>>();

    cudaDeviceSynchronize();

    return 0;
}

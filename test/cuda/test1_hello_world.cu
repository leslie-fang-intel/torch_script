// Build: /usr/local/cuda-12.1/bin/nvcc test_hello_world.cu
#include <stdio.h>
#include <cuda.h>

__global__ void dkernel() {
    printf("Hello World. \n");
}

int main() {
    dkernel<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}

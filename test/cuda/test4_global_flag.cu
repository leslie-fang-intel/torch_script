// Build: /usr/local/cuda-12.1/bin/nvcc test_hello_world.cu
#include <stdio.h>
#include <cuda.h>

#define N 100

__global__ void dkernel(char* msg, int len) {
    // printf(msg);
    unsigned id = threadIdx.x;
    if (id < len) {
        ++msg[id];
    }
}

int main() {
    char msg[] = "Hello World.";
    char* gpu_msg;

    cudaMalloc(&gpu_msg, (1 + strlen(msg)) * sizeof(char));

    cudaMemcpy(gpu_msg, msg, (1 + strlen(msg)) * sizeof(char), cudaMemcpyHostToDevice);

    dkernel<<<1, N>>>(gpu_msg, strlen(msg));

    cudaMemcpy(msg, gpu_msg, (1 + strlen(msg)) * sizeof(char), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    printf(msg);

    return 0;
}

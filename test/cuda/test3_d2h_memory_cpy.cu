// Build: /usr/local/cuda-12.1/bin/nvcc test_hello_world.cu
#include <stdio.h>
#include <cuda.h>

#define N 100

__global__ void dkernel(int* da) {
    da[threadIdx.x] = threadIdx.x;
}

int main() {
    // allocate memory on Device for calculation
    // https://blog.csdn.net/bendanban/article/details/8151335
    int* da;
    cudaMalloc(&da, N * sizeof(int));
    dkernel<<<1, N>>>(da);

    // Cpy result from device to host
    int a[N];
    // No need of cudaDeviceSynchronize with cudaMemcpy
    cudaMemcpy(a, da, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result on host
    for (int i=0; i<N; i++) {
        printf("%d \n", a[i]);
    }
    return 0;
}

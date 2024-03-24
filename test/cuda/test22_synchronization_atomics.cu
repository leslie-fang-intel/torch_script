// Build: /usr/local/cuda-12.1/bin/nvcc -g -G test_hello_world.cu
#include <stdio.h>
#include <cuda.h>

#define N 1024

__global__ void dkernel(unsigned int* dm) {

    // Atomic operator in CUDA
    // https://blog.csdn.net/QLeelq/article/details/127534493
    atomicInc(dm, N);

}

int main() {

    unsigned int* dm;
    cudaMalloc(&dm, 1 * sizeof(unsigned int));

    unsigned int* hm = (unsigned int *)malloc(1 * sizeof(unsigned int));
    
    // init the data
    *hm = 0;
    cudaMemcpy(dm, hm, sizeof(unsigned int), cudaMemcpyHostToDevice);


    // 第三个 参数表示为 dynamic shared memory
    dkernel<<<1, N>>>(dm);

    cudaDeviceSynchronize();

    cudaMemcpy(hm, dm, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();


    printf("s is: %d \n", *hm);


    return 0;
}

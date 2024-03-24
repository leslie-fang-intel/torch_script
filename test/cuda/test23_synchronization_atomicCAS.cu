// Build: /usr/local/cuda-12.1/bin/nvcc -g -G test_hello_world.cu
#include <stdio.h>
#include <cuda.h>

#define N 1024

__global__ void dkernel(int* dm) {
    int old = atomicCAS(dm, 0, threadIdx.x + 1);
    if (old == 0) {
        printf("Thread %d \n", threadIdx.x);
    }
    old = atomicCAS(dm, 0, threadIdx.x + 1);
    if (old == 0) {
        printf("2 Thread %d \n", threadIdx.x);
    }
    old = atomicCAS(dm, threadIdx.x, -1);
    if (old == threadIdx.x) {
        printf("3 Thread %d \n", threadIdx.x);
    }

}

int main() {

    int* dm;
    cudaMalloc(&dm, 1 * sizeof(int));
    cudaMemset(&dm, 0, sizeof(int));


    // 第三个 参数表示为 dynamic shared memory
    dkernel<<<1, N>>>(dm);

    cudaDeviceSynchronize();

    return 0;
}

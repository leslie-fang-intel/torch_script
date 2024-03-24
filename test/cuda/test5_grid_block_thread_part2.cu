// Build: /usr/local/cuda-12.1/bin/nvcc test_hello_world.cu
#include <stdio.h>
#include <cuda.h>

#define N 5
#define M 6

__global__ void dkernel(unsigned* da) {
    unsigned threadId  =
        threadIdx.x
        + threadIdx.y * blockDim.x;
    // printf("threadId is %d \n", threadId);
    da[threadId] = threadId;
}

int main() {

    // A kernel will use a grid of threads to compute
    unsigned* da;
    unsigned* ha = (unsigned *)malloc(N * M * sizeof(unsigned));
    cudaMalloc(&da, N * M * sizeof(unsigned));

    dim3 block(N, M, 1);

    dkernel<<<1, block>>>(da);

    cudaMemcpy(ha, da, N * M * sizeof(unsigned), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    for (int i=0; i<N; i++) {
        for (int j=0; j<M; j++) {
            printf("%2d ", ha[i*M + j]);
        }
        printf("\n");
    }

    return 0;
}

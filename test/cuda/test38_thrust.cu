// Build: /usr/local/cuda-12.1/bin/nvcc -g -G test_hello_world.cu
#include <stdio.h>
#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>


#define N 1024

// __global__ void dkernel(thrust::device_vector<int>& D) {
//     // for (int i = 0; i < D.size(); i++) {
//     //         printf("Hello, World! %d\n", D[i]);
//     // }
//     printf("Hello, World! %d\n", D[0]);
// }

int main() {

    // H存储了四个整数 on CPU
    thrust::host_vector<int> H(4);
    H[0] = 1;
    H[1] = 2;
    H[2] = 3;
    H[3] = 4;

    // Copy memory from CPU to GPU
    thrust::device_vector<int> D = H;
    // dkernel<<<1, 1>>>(D);
    // cudaDeviceSynchronize();
    std::cout<<D[0]<<std::endl;

    return 0;
}

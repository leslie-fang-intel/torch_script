#include <ATen/AccumulateType.h>
#include <torch/library.h>
#include <cuda.h>

namespace at {
namespace native {

__global__ void _extended_add_kernel(float * a_ptr, float * b_ptr, float * out_ptr) {
    auto idx = blockIdx.x;
    auto y_len = blockDim.x;
    auto idy = threadIdx.x;
    out_ptr[idx*y_len + idy] = a_ptr[idx*y_len + idy] + b_ptr[idx*y_len + idy];
}

Tensor extended_add_kernel(Tensor a, Tensor b, Tensor out) {

    auto a_ptr = a.data_ptr();
    auto b_ptr = b.data_ptr();
    auto out_ptr = out.data_ptr();
    auto N1 = a.size(0);
    auto N2 = a.size(1);

    //TODO<leslie> assert the scalar_type is float, and the input tensor is 2D

    _extended_add_kernel<<<N1,N2>>>((float*)a_ptr, (float*)b_ptr, (float*)out_ptr);

    return out;
}

} // namespace native
} // namespace at

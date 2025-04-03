#include <ATen/ATen.h>
#include <torch/library.h>
#include <iostream>
#include <Python.h>
#include <torch/torch.h>

namespace torchll {

at::Tensor toy_test_ll(
    at::Tensor   _in_feats)
{
    std::cout<<"---- run into cpu 2 ----"<<std::endl;
    return _in_feats;

}

TORCH_LIBRARY_IMPL(torchll, CPU, m) {
  std::cout << "Inside TORCH_LIBRARY_IMPL2 ..." << std::endl;
  // m.impl("torchll::toy_test_ll", &toy_test_ll);
  m.impl("toy_test_ll", &toy_test_ll);
}

} // namespace torchll
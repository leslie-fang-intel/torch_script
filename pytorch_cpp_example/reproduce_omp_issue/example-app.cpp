#include <torch/torch.h>
#include <omp.h>

int main()
{
  std::cout<<"omp_get_max_threads() is: "<<omp_get_max_threads()<<std::endl;

  return 0;
}
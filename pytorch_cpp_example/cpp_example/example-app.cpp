#include <torch/torch.h>
#include <torch/script.h>
#include <c10/util/Exception.h>
#include <iostream>
#include <omp.h>

int main()
{
  torch::jit::script::Module container = torch::jit::load("../scores3.pt");

  // Load values by name
  torch::Tensor batch_scores = container.attr("scores").toTensor();

  auto compare_result = at::nonzero(batch_scores[15].squeeze(0).slice(1, 1, 2).squeeze(1) > 0.05).squeeze(1);
  std::cout<<"single thread result at BS 16 and class 1: "<<compare_result.sizes()<<std::endl;

  auto nbatch = batch_scores.size(0); // number of batches 16
  auto ndets = batch_scores.size(1); // number of boxes 15130
  auto nscore = batch_scores.size(2); // number of labels 81

  auto nbatch_x_nscore = nbatch * nscore; // (number of batches) * (number of labels)

  //at::set_num_threads(at::intraop_default_num_threads());
  int64_t grain_size = std::min(at::internal::GRAIN_SIZE / nbatch_x_nscore, (int64_t)1);
  at::parallel_for(0, nbatch_x_nscore, grain_size, [&](int64_t begin, int64_t end){
    for (int index = begin; index < end; index++) {
      //omp_set_num_threads(28);
      auto bs = index / nscore;
      auto i = index % nscore;
      at::Tensor scores = batch_scores[bs].squeeze(0); // scores for boxes per image: (num_bbox, 81); For example: (15130, 81)
      at::Tensor score = scores.slice(1, i, i+1).squeeze(1); // score for boxes per image per class: (num_bbox); For example: (15130)

      at::Tensor mask_index = at::nonzero(score > 0.05).squeeze(1);

      if (bs == 15 && i == 1) {
        std::cout<<"Multi threads result at BS 16 and class 1: "<<mask_index.sizes()<<std::endl;
        auto pointer1 = compare_result.data<int64_t>();
        auto pointer2 = mask_index.data<int64_t>();
        for(int j =0; j<mask_index.sizes()[0];j++ ){
          std::cout<<pointer1[j]<<std::endl;
          std::cout<<pointer2[j]<<std::endl;
        }
        
      }
    }
  });

  return 0;
}
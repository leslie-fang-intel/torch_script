#include <iostream>
#include <torch/script.h>
#include <memory>
#include "csrc/jit/fusion_pass.h"
#include "csrc/quantization/auto_opt_config.hpp"
#include <torch/csrc/jit/passes/pass_manager.h>

using namespace torch::jit;
using namespace torch_ipex;

int main() {
    torch::jit::registerPrePass([](std::shared_ptr<Graph>& g) {
        if (AutoOptConfig::singleton().get_jit_fuse()) {
            torch::jit::FusionPass(g);
        }
    });

    torch::jit::script::Module module;
    try {
        module = torch::jit::load("../scriptmodule.pt");
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }
    std::vector<torch::jit::IValue> inputs;
    // make sure input data are converted to channels last format
    inputs.push_back(torch::ones({1, 3, 224, 224}).to(c10::MemoryFormat::ChannelsLast));

    // Warm up to trigger the fusion path
    for(int i=0;i<10;i++){
        at::Tensor output = module.forward(inputs).toTensor();
    }
    // std::cout<<output<<std::endl;
    return 0;
}
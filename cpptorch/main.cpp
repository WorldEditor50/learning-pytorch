#include <iostream>
#include <torch/script.h>
#include <memory>
#include <vector>
#include "tensor_op.h"

int main()
{
#if 0
    torch::jit::script::Module module;
    module = torch::jit::load("test.pth");
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({1, 3, 224, 224}));
    at::Tensor output = module.forward(inputs).toTensor();
    std::cout<<output.slice(/*dim*/1, /*start*/3, /*end*/5)<<std::endl;
#endif
    //test::createTensor();
    //test::transformation();
    //test::slice();
    //test::accumulate();
    test::op();
    return 0;
}

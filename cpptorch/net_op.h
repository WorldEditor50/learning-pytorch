#ifndef NET_OP_H
#define NET_OP_H
#include <torch/torch.h>
#include <memory>
#include <iostream>

class Net : public torch::nn::Module
{
private:
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
    torch::nn::Linear fc3{nullptr};
public:
    Net()
    {
        fc1 = register_module("fc1", torch::nn::Linear(784, 64));
        fc2 = register_module("fc2", torch::nn::Linear(64, 32));
        fc3 = register_module("fc3", torch::nn::Linear(32, 10));
    }
    torch::Tensor forward(const torch::Tensor &x)
    {
        auto y = torch::relu(fc1->forward(x.reshape({x.size(0), 784})));
        y = torch::dropout(y, 0.5, is_training());
        y = torch::relu(fc2->forward(y));
        y = torch::relu(fc3->forward(y));
        return torch::log_softmax(y, 1);
    }
    static void test()
    {
        auto net = std::make_shared<Net>();
        auto dataLoader = torch::data::make_data_loader(
                    torch::data::datasets::MNIST("/home/eigen/Downloads/mnist1").map(
                        torch::data::transforms::Stack<>()), 64);
        torch::optim::SGD optimizer(net->parameters(), 0.01);
        for (int epoch = 0; epoch < 10; epoch++) {
            std::size_t index = 0;
            for (auto& x : *dataLoader) {
                optimizer.zero_grad();
                auto y = net->forward(x.data);
                torch::Tensor loss = torch::nll_loss(y, x.target);
                loss.backward();
                optimizer.step();
                if (index%100 == 0) {
                    std::cout<<"Epoch: "<<epoch<<" | Batch: "<<index
                            <<" | Loss: "<<loss.item<float>()<<std::endl;
                    torch::save(net, "net.pth");
                }
                index++;
            }
        }
        return;
    }
};

#endif // NET_OP_H

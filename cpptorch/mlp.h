#ifndef MLP_H
#define MLP_H
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>
#include <string>

class LayerImpl : public torch::nn::Module
{
private:
    torch::nn::Linear linear;
    torch::nn::BatchNorm1d batchNorm;
public:
    explicit LayerImpl(int inputDim, int outputDim)
    {
        linear = register_module("linear",
                                 torch::nn::Linear(torch::nn::LinearOptions(inputDim, outputDim)));
        batchNorm = register_module("batchnorm", torch::nn::BatchNorm1d(outputDim));
    }
    torch::Tensor forward(const torch::Tensor &x)
    {
        auto out1 = torch::relu(linear->forward(x));
        return batchNorm->forward(out1);
    }
};
TORCH_MODULE(Layer);

class MLP : public torch::nn::Module
{
protected:
    Layer layer1;
    Layer layer2;
    Layer layer3;
    torch::nn::Linear outLayer;
public:
    explicit MLP(int inputDim, int hiddenDim, int outputDim)
    {
        layer1 = Layer(inputDim, hiddenDim);
        layer2 = Layer(hiddenDim, hiddenDim);
        layer3 = Layer(hiddenDim, hiddenDim);
        outLayer = torch::nn::Linear(hiddenDim, outputDim);
        layer1 = register_module("layer1", layer1);
        layer2 = register_module("layer2", layer2);
        layer2 = register_module("layer3", layer3);
        outLayer = register_module("out", outLayer);
    }
    torch::Tensor forward(const torch::Tensor &x)
    {
        auto out1 = layer1->forward(x);
        auto out2 = layer2->forward(out1);
        auto out3 = layer3->forward(out2);
        return outLayer->forward(out3);
    }
};

#endif // MLP_H

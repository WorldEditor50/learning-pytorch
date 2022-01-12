#ifndef TENSOR_OP_H
#define TENSOR_OP_H
#include <torch/script.h>
#include <vector>
#include <iostream>

namespace test {

void createTensor()
{
    std::cout<<"zeros:"<<std::endl;
    auto t1 = torch::zeros({2, 3});
    std::cout<<t1<<std::endl;
    std::cout<<"ones:"<<std::endl;
    auto t2 = torch::ones({2, 2, 3});
    std::cout<<t2<<std::endl;
    std::cout<<"eye:"<<std::endl;
    auto t3 = torch::eye(4);
    std::cout<<t3<<std::endl;
    std::cout<<"full:"<<std::endl;
    auto t4 = torch::full({2, 3, 3}, 9);
    std::cout<<t4<<std::endl;
    std::cout<<"rand:"<<std::endl;
    auto t5 = torch::rand({3, 3});
    std::cout<<t5<<std::endl;
    std::cout<<"randint:"<<std::endl;
    auto t6 = torch::randint(0, 9, {3, 3});
    std::cout<<t6<<std::endl;
    std::cout<<"from_blob:"<<std::endl;
    int a1[3] = {1, 2, 4};
    auto t7 = torch::from_blob(a1, {3}, torch::kInt);
    std::cout<<t7<<std::endl;
    std::vector<float> a2 = {1, 2, 3};
    auto t8 = torch::from_blob(a2.data(), {3}, torch::kFloat);
    std::cout<<t8<<std::endl;
    std::cout<<"like:"<<std::endl;
    auto t9 = torch::Tensor(t1);
    std::cout<<t9<<std::endl;
    auto t10 = torch::zeros_like(t1);
    std::cout<<t10<<std::endl;
    auto t11 = torch::rand_like(t1);
    std::cout<<t11<<std::endl;
    auto t12 = t1.clone();
    std::cout<<t12<<std::endl;
    auto t13 = torch::ones_like(t1);
    std::cout<<t13<<std::endl;
    std::cout<<"distribution:"<<std::endl;
    auto t14 = torch::normal(0, 1, {3, 3});
    std::cout<<t14<<std::endl;
    return;
}

void transformation()
{
    std::cout<<"origin:"<<std::endl;
    auto t = torch::randint(0, 10, {2, 2, 2});
    std::cout<<t<<std::endl;
    std::cout<<"transpose:1,2"<<std::endl;
    auto t1 = t.transpose(1, 2);
    std::cout<<t1<<std::endl;
    std::cout<<"permute:1,2,0"<<std::endl;
    auto t2 = t.permute({1, 2, 0});
    std::cout<<t2<<std::endl;
    std::cout<<"reshape:1, 8, 1"<<std::endl;
    auto t3 = t.reshape({1, 8, 1});
    std::cout<<t3<<std::endl;
    std::cout<<"view:8, 1, 1"<<std::endl;
    auto t4 = t.view({8, 1, 1});
    std::cout<<t4<<std::endl;
    return;
}

void slice()
{
    std::cout<<"operator[]:"<<std::endl;
    auto t = torch::rand({4, 3, 28, 28});
    std::cout<<t.sizes()<<std::endl;
    std::cout<<t[0].sizes()<<std::endl;
    std::cout<<t[0][0].sizes()<<std::endl;
    std::cout<<t[0][0][0].sizes()<<std::endl;
    std::cout<<t[0][0][0][0]<<std::endl;
    std::cout<<"index_select:"<<std::endl;
    auto t1 = t.index_select(0, torch::tensor({0, 3, 3}));
    std::cout<<t1.sizes()<<std::endl;
    auto t2 = t.index_select(1, torch::tensor({0, 2}));
    std::cout<<t2.sizes()<<std::endl;
    auto t3 = t.index_select(2, torch::arange(0, 8));
    std::cout<<t3.sizes()<<std::endl;
    std::cout<<"narrow:"<<std::endl;
    auto t4 = t.narrow(0, 1, 2);
    std::cout<<t4.sizes()<<std::endl;
    std::cout<<"select:"<<std::endl;
    auto t5 = t.select(0, 1);
    std::cout<<t5.sizes()<<std::endl;
    std::cout<<"mask:"<<std::endl;
    auto c = torch::rand({3, 4});
    std::cout<<c<<std::endl;
    auto mask = torch::zeros({3, 4});
    mask[0][0] = 1;
    auto t6 = c.index({mask.to(torch::kBool)});
    std::cout<<t6<<std::endl;
    mask[0][2] = 1;
    auto t7 = c.index_put_({mask.to(torch::kBool) }, t6+1.5);
    std::cout<<t7<<std::endl;
    return;
}

void accumulate()
{
    auto t1 = torch::rand({2, 3});
    auto t2 = torch::rand({2, 3});
    std::cout<<"t1:"<<t1<<std::endl;
    std::cout<<"t2:"<<t2<<std::endl;
    std::cout<<"cat:"<<std::endl;
    auto t3 = torch::cat({t1, t2}, 1);
    std::cout<<"t3:"<<t3<<std::endl;
    std::cout<<"stack:"<<std::endl;
    auto t4 = torch::stack({t1, t2}, 1);
    std::cout<<"t4:"<<t4<<std::endl;
    return;
}

void op()
{
    auto t1 = torch::randint(0, 9, {2, 3});
    auto t2 = torch::randint(0, 9, {2, 3});
    std::cout<<"t1:"<<t1<<std::endl;
    std::cout<<"t2:"<<t2<<std::endl;
    std::cout<<"+:"<<std::endl;
    auto t3 = t1 + t2;
    std::cout<<t3<<std::endl;
    std::cout<<"*:"<<std::endl;
    auto t4 = t1 * t2;
    std::cout<<t4<<std::endl;
    std::cout<<"mm:"<<std::endl;
    auto t5 = t1.mm(t2.t());
    std::cout<<t5<<std::endl;
    return;
}

}
#endif // TENSOR_OP_H

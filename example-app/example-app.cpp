#include <iostream>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>

using namespace std;

int main(int argc, char* argv[])
{
    cout<<"argc:"<<argc<<endl;
    for(int i = 0; i < argc;i++)
        cout<<"argv:"<<argv[i]<<endl;

    cout << "Hello pytorch C++!" << endl;

    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;

    at::Tensor a = at::ones({2, 2}, at::kInt);
    at::Tensor b = at::randn({2, 2});
    auto c = a + b.to(at::kInt);

    cout<<"add:"<<endl;
    cout<<a<<endl<<c<<endl;

    torch::Tensor t1 = torch::ones({2, 2}, torch::requires_grad());
    torch::Tensor t2 = torch::randn({2, 2});
    auto out = t1 + t2;

    cout<<"backward:"<<endl;
    cout<< t1<<endl<<t2<<endl<<out<<endl;
    out.backward();

    cout<< t1<<endl<<t2<<endl<<out<<endl;

    cout<<"t1.grad:"<<endl<<t1.grad()<<endl;
    return 0;
}

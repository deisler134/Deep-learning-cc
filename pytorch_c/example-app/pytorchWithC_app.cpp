//============================================================================
// Name        : pytorchWithC_app.cpp
// Author      : Deisler
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <ATen/ATen.h>
#include <torch/csrc/api/include/torch/torch.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>

using namespace std;

void show_pytorch_tensor_C(){

	//show tenor variable
    torch::Tensor tensor = torch::rand({2, 3});
    cout <<"torch variable torch::rand({2,3}):"<<endl << tensor <<endl;

	//show ATen variable
	at::Tensor a = at::ones({2, 2}, at::kInt);
	cout<<"ATen variable:"<<endl
			<<"at::Tensor a = at::ones(({2,2}, at::kInt))"<<endl<<a<<endl;

	//show ATen calculate
    at::Tensor b = at::randn({2, 2});
    cout<<"at::Tensor b = at::randn({2, 2})"<<endl
    		<<b<<endl;
    auto c = a + b.to(at::kInt);
    cout<<"c = a + b.to(at::kInt)"<<endl
    		<<c<<endl;

    //show torch Tensor backward:
    cout<<" show torch Tensor backward in C++:"<<endl;
    torch::Tensor t1 = torch::ones({2, 2}, torch::requires_grad());
    cout<<"torch::Tensor t1 = torch::ones({2, 2}, torch::requires_grad())"<<endl
    		<<t1<<endl;
    torch::Tensor t2 = torch::randn({2, 2});
    cout<<"torch::Tensor t2 = torch::randn({2, 2})"<<endl
    		<<t2<<endl;
    auto out = t1 + t2;
    cout<<"auto out = t1 + t2"<<endl
    		<<out<<endl;

    out.backward();
    cout<<"out.backward()"<<endl;
    cout<<"t1.grad():"<<endl<<t1.grad()<<endl;

    //show backward with tensor variable:
    torch::Tensor t3 = torch::ones({2,2},torch::requires_grad());
    cout<<"torch::Tensor t3 = torch::ones({2,2},torch::requires_grad())"<<endl
    		<<t3<<endl;
    torch::Tensor t4 = t3*t3*2;
    cout<<"torch::Tensor t4 = t3*t3*2"<<endl
    		<<t4<<endl;
    torch::Tensor out1 = at::mean(t4);
    cout<<"torch::Tensor out1 = at::mean(t4)"<<endl
    		<<out1<<endl;

    torch::Tensor param1 = torch::ones({1,1})*2;	//note: must be variable!
    cout<<"torch::Tensor param1 = torch::ones({1,1}) * 2"<<endl
    		<<param1<<endl;

    cout<<"param1:"<<param1<<endl;

    out1.backward(param1);

    cout<<"t3.grad():"<<t3.grad()<<endl;
    cout<<"out1.backward(param1)"<<endl;
    cout<<"t3.grad():"<<endl<<t3.grad()<<endl;

}

int main(int argc, char* argv[]) {

	//show main parameters
    cout<<"argc:"<<argc<<endl;
    for(int i = 0; i < argc;i++)
        cout<<"argv:"<<argv[i]<<endl;

	cout << "!!!Hello pytorch c++!!!" << endl; // prints !!!Hello pytorch c++!!!

	show_pytorch_tensor_C();

	return 0;
}

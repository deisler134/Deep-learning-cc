/*
 * pytorch_C_frontend.cpp
 *
 *  Created on: May 2, 2019
 *      Author: Deisler
 */

// Comparison pytorch between python and c++

/*
// python create pytorch module

import torch

class Net(torch.nn.Module):
	def __init__(self, N, M):
		super(Net, self).__init__()
		self.W = torch.nn.Parameter(torch.randn(N, M))
		self.b = torch.nn.Parameter(torch.randn(M))

	def forward(self, input):
		retrun torch.addmm(self.b, input, self.W)

*/


#include <iostream>
#include <ATen/ATen.h>
#include <torch/csrc/api/include/torch/torch.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>

// C++ create pytorch module

struct Net1 : torch::nn::Module{
	Net1(int64_t N, int64_t M){
		W = register_parameter("W", torch::randn({N, M}));
		b = register_parameter("b", torch::randn(M));
	}

	torch::Tensor forward(torch::Tensor input){
		return torch::addmm(b, input, W);
	}

	torch::Tensor W, b;
};

/*
// python create module with submodule and traversing

class Net(torch.nn.module):
	def __init__(self, N, M):
		super(Net, self).__init__()
		self.linear = torch.nn.Linear(N, M)
		self.another_bias = torch.randn(M)

	def forward(self, input):
		return self.linear + self.another_bias

//>>> net = Net(4, 5)
//>>> print(list(net.parameters()))
//[Parameter containing:
//tensor([0.0808, 0.8613, 0.2017, 0.5206, 0.5353], requires_grad=True), Parameter containing:
//tensor([[-0.3740, -0.0976, -0.4786, -0.4928],
//        [-0.1434,  0.4713,  0.1735, -0.3293],
//        [-0.3467, -0.3858,  0.1980,  0.1986],
//        [-0.1975,  0.4278, -0.1831, -0.2709],
//        [ 0.3730,  0.4307,  0.3236, -0.0629]], requires_grad=True), Parameter containing:
//tensor([ 0.2038,  0.4638, -0.2023,  0.1230, -0.0516], requires_grad=True)]

*/

// C++ create pytorch module with submodule

struct Net2 : torch::nn::Module{
	Net2(int64_t N, int64_t M)
	: linear( register_module("linear", torch::nn::Linear((N, M)))){
		another_bias = register_parameter("b", torch::randn(M));
	}

	torch::Tensor forward(torch::Tensor input){
		return linear(input) + another_bias;
	}
	torch::nn::Linear linear;
	torch::Tensor another_bias;
};




/*
 * test.cpp
 * implement custom operator c++
 *  Created on: May 14, 2019
 *      Author: deisler
 */

#include <iostream>
#include <vector>
#include <torch/script.h>

using namespace std;


torch::Tensor datat2tensor(int* a){

	torch::Tensor out = torch::from_blob(a,{8});

	return out.clone();

}

static auto registry =
  torch::jit::RegisterOperators("my_ops::datat2tensor", &datat2tensor);


/*
 * mnist2.cpp
 *
 *  Created on: May 17, 2019
 *      Author: deisler
 */

#include <torch/torch.h>

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

using namespace torch;

// Where to find the MNIST dataset.
const char* kDataRoot = "./data";

// The batch size for training.
const int64_t kTrainBatchSize = 64;

// The batch size for testing.
const int64_t kTestBatchSize = 1000;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 10;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;


//template <typename DataLoader>
//void train(
//    int32_t epoch,
//	MnistNet& model,
////	torch::nn::Sequential& model,
////	Net& model,
//    torch::Device device,
//    DataLoader& data_loader,
//    torch::optim::Optimizer& optimizer,
//    size_t dataset_size) {
//  model->train();
//  size_t batch_idx = 0;
//  for (auto& batch : data_loader) {
//    auto data = batch.data.to(device), targets = batch.target.to(device);
//    optimizer.zero_grad();
//    auto output = model->forward(data);
//    auto loss = torch::nll_loss(output, targets);
//    AT_ASSERT(!std::isnan(loss.template item<float>()));
//    loss.backward();
//    optimizer.step();
//
//    if (batch_idx++ % kLogInterval == 0) {
//      std::printf(
//          "\rTrain Epoch: %d [%5ld/%5ld] Loss: %.4f",
//          epoch,
//          batch_idx * batch.data.size(0),
//          dataset_size,
//          loss.template item<float>());
//    }
//  }
//}
//
//template <typename DataLoader>
//void test(
//	MnistNet& model,
////	torch::nn::Sequential& model,
////	Net& model,
//    torch::Device device,
//    DataLoader& data_loader,
//    size_t dataset_size) {
//  torch::NoGradGuard no_grad;
//  model->eval();
//  double test_loss = 0;
//  int32_t correct = 0;
//  for (const auto& batch : data_loader) {
//    auto data = batch.data.to(device), targets = batch.target.to(device);
//    auto output = model->forward(data);
//    test_loss += torch::nll_loss(
//                     output,
//                     targets,
//                     /*weight=*/{},
//                     Reduction::Sum)
//                     .template item<float>();
//    auto pred = output.argmax(1);
//    correct += pred.eq(targets).sum().template item<int64_t>();
//  }
//
//  test_loss /= dataset_size;
//  std::printf(
//      "\nTest set: Average loss: %.4f | Accuracy: %.3f\n",
//      test_loss,
//      static_cast<double>(correct) / dataset_size);
//}

int main(int argc, const char* argv[]) {
  torch::manual_seed(1);

  // Create the device we pass around based on whether CUDA is available.
  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available! Training on GPU." << std::endl;
    device = torch::Device(torch::kCUDA);
  }

  nn::Sequential MnistNet(
		nn::Conv2d(
				  nn::Conv2dOptions(1, 10, 5).with_bias(false)),
		nn::Functional(torch::max_pool2d,2,2,0,1,false),
		nn::Functional(torch::relu),
		nn::Conv2d(
				nn::Conv2dOptions(10, 20, 5).with_bias(false)),
		nn::Dropout(),
		nn::Functional(torch::max_pool2d,2,2,0,1,false),
		nn::Functional(torch::relu),
		nn::Functional(torch::flatten,1,-1),
		nn::Linear(nn::LinearOptions(320, 50).with_bias(false)),
		nn::Functional(torch::dropout,0.5, true),
		nn::Linear(nn::LinearOptions(50, 10).with_bias(false))//,nn::Functional(torch::log_softmax,1)
  );
  MnistNet->to(device);

  auto train_dataset = torch::data::datasets::MNIST(kDataRoot)
                           .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                           .map(torch::data::transforms::Stack<>());
  const size_t train_dataset_size = train_dataset.size().value();
  auto train_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(train_dataset), kTrainBatchSize);
  torch::optim::SGD optimizer(
		  MnistNet->parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));

  for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {

	  MnistNet->train();
	  size_t batch_idx = 0;

	  for (auto& batch : *train_loader) {
	    auto data = batch.data.to(device), targets = batch.target.to(device);
	    optimizer.zero_grad();
	    auto output = torch::log_softmax(MnistNet->forward(data),1);
	    auto loss = torch::nll_loss(output, targets);
	    AT_ASSERT(!std::isnan(loss.template item<float>()));
	    loss.backward();
	    optimizer.step();

	    if (batch_idx++ % kLogInterval == 0) {
	      std::printf(
	          "\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f",
	          epoch,
	          batch_idx * batch.data.size(0),
			  train_dataset_size,
	          loss.template item<float>());
	    }
	  }
  }

}



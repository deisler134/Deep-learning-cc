'''
Created on May 14, 2019

    create script to load custom C++

@author: deisler
'''

import torch
import torch.utils.cpp_extension

print(torch.__version__)
print(torch.utils.cpp_extension.check_compiler_abi_compatibility("g++"))
print(torch.utils.cpp_extension.check_compiler_abi_compatibility("gcc"))

torch.ops.load_library("/home/deisler/Downloads/resource/project/workspace/eclipse/src/custom_C_plus/build/libData2Tensor.so")

print(torch.ops.my_ops.data2tensor)
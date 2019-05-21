/*
 * hello_tf.c
 *
 *  Created on: May 21, 2019
 *      Author: deisler
 */

#include <stdio.h>
#include <tensorflow/c/c_api.h>

int main() {
  printf("Hello from TensorFlow C library version %s\n", TF_Version());
  return 0;
}




#include <cuda_runtime_api.h>
#include "allocator.h"
#include <cstdio>

extern "C" {

cudaError_t cudaMalloc(void **devPtr, size_t size) {
  // printf("alloc size: %lu\n", size);
  pytorch_malloc::Allocator *allocator = pytorch_malloc::Allocator::Instance();
  // printf("allocated\n");
  return allocator->malloc(devPtr, size);
}

cudaError_t cudaFree(void *devPtr) {
  pytorch_malloc::Allocator *allocator = pytorch_malloc::Allocator::Instance();
  return allocator->free(devPtr);
}

cudaError_t cudaMemGetInfo (size_t* free, size_t* total) {
  printf("cudagetInfo called\n");
// ​cudaError_t cudaMemGetInfo (size_t* free, size_t* total) {
  *free = BUFFER_SIZE;
  *total = BUFFER_SIZE;
  return cudaSuccess;
}

}

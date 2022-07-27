#include <cuda_runtime_api.h>
#include "allocator.h"

extern "C" {

cudaError_t cudaMalloc(void **devPtr, size_t size) {
  pytorch_malloc::Allocator *allocator = pytorch_malloc::Allocator::Instance();
  // printf("alloc size: %lu\n", size);
  return allocator->malloc(devPtr, size);
}

cudaError_t cudaFree(void *devPtr) {
  pytorch_malloc::Allocator *allocator = pytorch_malloc::Allocator::Instance();
  return allocator->free(devPtr);
}

cudaError_t cudaMemGetInfo (size_t* free, size_t* total) {
// ​cudaError_t cudaMemGetInfo (size_t* free, size_t* total) {
  *free = (size_t) 4 * 1024 * 1024 * 1024;
  *total = (size_t) 4 * 1024 * 1024 * 1024;
  return cudaSuccess;
}

}

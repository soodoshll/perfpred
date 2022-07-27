#include <stdio.h>
#include <dlfcn.h>
#include <assert.h> 

#include <iostream>

#include "allocator.h"
#include "cnmem/cnmem.h"

#define DTYPE unsigned long long
constexpr size_t BUFFER_SIZE = (size_t)4 * 1024 * 1024 * 1024;

namespace pytorch_malloc {

cudaError_t cnmemStatus2cudaError(cnmemStatus_t status) {
  switch (status) {
    case CNMEM_STATUS_SUCCESS:
      return cudaSuccess;
    case CNMEM_STATUS_CUDA_ERROR:
      return cudaErrorUnknown;
    case CNMEM_STATUS_INVALID_ARGUMENT:
      return cudaErrorInvalidValue;
    case CNMEM_STATUS_NOT_INITIALIZED:
      return cudaErrorInitializationError;
    case CNMEM_STATUS_OUT_OF_MEMORY:
      return cudaErrorMemoryAllocation;
    case CNMEM_STATUS_UNKNOWN_ERROR:
      return cudaErrorUnknown;
    default:
      return cudaErrorUnknown;
  }
}

Allocator::Allocator() {
  cnmemDevice_t device;
  device.device = 0;
  device.size = 0;
  device.numStreams = 0;
  device.streams = NULL;
  *(device.streamSizes) = 0;
  cnmemInit(1, &device, 0);
  cnmemStatus_t status = cnmemMalloc(&(this->devPtr_), BUFFER_SIZE, NULL);
  cudaMemset(devPtr_, 0, BUFFER_SIZE);
  assert(status == CNMEM_STATUS_SUCCESS);
}

Allocator::~Allocator() {
  // printf("max mem: %.3f GB\n", alloc_max_ / 1e9);
  cnmemFree(this->devPtr_, NULL);
  cnmemFinalize();
}

cudaError_t Allocator::malloc(void **devPtr, size_t size) {
  // cnmemStatus_t status = cnmemMalloc(devPtr, size, NULL);
  alloc_cur_ += size;
  if (alloc_cur_ > alloc_max_)
    alloc_max_ = alloc_cur_;
  // printf("%.3f GB | alloc: %lu\n", alloc_cur_ / 1e9, size);
  *devPtr = (DTYPE *)this->devPtr_ + alloc_num_;
  cudaMemset(*devPtr, 0, size);
  size_[alloc_num_] = size;
  alloc_num_++;
  return cudaSuccess;
}


cudaError_t Allocator::free(void *devPtr) {
  // cnmemStatus_t status = cnmemFree(devPtr, NULL);
  // return cnmemStatus2cudaError(status);
  int diff = (DTYPE *)devPtr - (DTYPE *)devPtr_;
  auto size = size_[diff];
  alloc_cur_ -= size;
  // printf("%.3f GB | free: %lu\n", alloc_cur_ / 1e9, size);
  return cudaSuccess;
}

}  // end pytorch_malloc

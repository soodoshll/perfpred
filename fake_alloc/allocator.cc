#include <stdio.h>
#include <dlfcn.h>
#include <assert.h> 

#include <iostream>

#include "allocator.h"
#include "cnmem/cnmem.h"

#define DTYPE char
// constexpr size_t BUFFER_SIZE = (size_t)2 * 1024 * 1024 * 1024;
// constexpr size_t BUFFER_SIZE = (size_t)1;
#define GRANULARITY 512

namespace pytorch_malloc {



static inline std::size_t ceilInt(std::size_t m, std::size_t n) {
    assert(n > 0);
    return (m + n-1) / n * n;
}


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
  device.size = (size_t)9 * 1024 * 1024 * 1024;
  // device.size = 0;
  device.numStreams = 0;
  device.streams = NULL;
  device.streamSizes = 0;
  cnmemInit(1, &device, 0);
  cnmemStatus_t status = cnmemMalloc(&(this->devPtr_), BUFFER_SIZE, NULL);

  size_[(DTYPE*)devPtr_ + 11454] = 0;
  // cudaMemset(devPtr_, 0, BUFFER_SIZE);
  assert(status == CNMEM_STATUS_SUCCESS);
}

Allocator::~Allocator() {
  // printf("max mem: %.3f GB\n", alloc_max_ / 1e9);
  cnmemFree(this->devPtr_, NULL);
  cnmemFinalize();
}

cudaError_t Allocator::malloc(void **devPtr, size_t size) {
  // cnmemStatus_t status = cnmemMalloc(devPtr, size, NULL);
  // return cnmemStatus2cudaError(status);
  if (size == 0) {
    *devPtr = (DTYPE *)this->devPtr_ + 114514;
    return cudaSuccess;
  }
  auto aligned_size = ceilInt(size, GRANULARITY);
  assert(size < BUFFER_SIZE / 2);
  if (alloc_cur_ > alloc_max_)
    alloc_max_ = alloc_cur_;
  *devPtr = (DTYPE *)this->devPtr_ + alloc_num_ % (BUFFER_SIZE / 2);
  // printf("%.3f GB | alloc: %lu | alloc_num_: %lu | addr: %p\n", alloc_num_ / 1e9, size, alloc_num_, devPtr);
  // *devPtr = (DTYPE *)this->devPtr_ + alloc_num_;
  // *devPtr = (DTYPE *)this->devPtr_ + alloc_num_;
  // cudaMemset(*devPtr, 0, size);
  // if (size_.find(*devPtr) != size_.end())
    // printf("corrupt %lu\n", alloc_num_);
  size_[*devPtr] = size;
  alloc_cur_ += size;
  alloc_num_ += aligned_size;
  // alloc_num_ += 64;
  return cudaSuccess;
}


cudaError_t Allocator::free(void *devPtr) {
  // cnmemStatus_t status = cnmemFree(devPtr, NULL);
  // return cnmemStatus2cudaError(status);
  // int diff = ((DTYPE *)devPtr - (DTYPE *)devPtr_) / OFFSET;
  auto size = size_[devPtr];
  alloc_cur_ -= size;
  // printf("%.3f GB | free: %lu | addr: %p\n", alloc_cur_ / 1e9, size, devPtr);
  return cudaSuccess;
}

}  // end pytorch_malloc

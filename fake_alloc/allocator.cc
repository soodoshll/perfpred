#include <stdio.h>
#include <dlfcn.h>
#include <assert.h> 

#include <iostream>
#include <mutex>

#include "allocator.h"
#include "cnmem/cnmem.h"

#define DTYPE char
// constexpr size_t BUFFER_SIZE = (size_t)2 * 1024 * 1024 * 1024;
// constexpr size_t BUFFER_SIZE = (size_t)1;
#define GRANULARITY 64

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
  device.size = BUFFER_SIZE;
  device.numStreams = 0;
  device.streams = NULL;
  device.streamSizes = 0;
  cnmemInit(1, &device, 0);
  cnmemStatus_t status = cnmemMalloc(&(this->devPtr_), BUFFER_SIZE, NULL);

  size_[(DTYPE*)devPtr_ + 11454] = 0;
  assert(status == CNMEM_STATUS_SUCCESS);
}

Allocator::~Allocator() {
  cnmemFree(this->devPtr_, NULL);
  cnmemFinalize();
}

cudaError_t Allocator::malloc(void **devPtr, size_t size) {
  std::lock_guard<std::mutex> guard(mutex_);
  auto aligned_size = ceilInt(size, GRANULARITY);
  alloc_num_ %= (BUFFER_SIZE / 2);
  if (alloc_num_ + aligned_size >= BUFFER_SIZE) {
    alloc_num_ = 0;
  }
  *devPtr = (DTYPE *)this->devPtr_ + alloc_num_ ;
  while (size_.find(*devPtr) != size_.end()) {
    *devPtr = (DTYPE*)(*devPtr) + GRANULARITY;
  }
  size_[*devPtr] = size;
  alloc_cur_ += size;
  alloc_num_ += aligned_size;
  if (alloc_cur_ > alloc_max_)
    alloc_max_ = alloc_cur_;
  return cudaSuccess;
}


cudaError_t Allocator::free(void *devPtr) {
  std::lock_guard<std::mutex> guard(mutex_);
  auto size = size_[devPtr];
  alloc_cur_ -= size;
  size_.erase(devPtr);
  return cudaSuccess;
}

}  // end pytorch_malloc

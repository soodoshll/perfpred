#include <stdio.h>
#include <dlfcn.h>
#include <assert.h> 

#include "allocator.h"

#define DTYPE char
#define GRANULARITY 1024

namespace pytorch_malloc {

static inline std::size_t ceilInt(std::size_t m, std::size_t n) {
    assert(n > 0);
    if (m == 0) return n;
    return (m + n-1) / n * n;
}

Allocator::Allocator() {
  // cudaMallocManaged(&devPtr_, BUFFER_SIZE);
  // cudaMalloc(&devPtr_, BUFFER_SIZE);
  // printf("%p %u\n", (DTYPE*)devPtr_, sizeof(DTYPE));
  // size_[(DTYPE*)devPtr_ + 11454] = 0;
  devPtr_ = 0;
}

Allocator::~Allocator() {
}

cudaError_t Allocator::malloc(void **devPtr, size_t size) {
  std::lock_guard<std::mutex> guard(mutex_);
  auto aligned_size = ceilInt(size, GRANULARITY);
  // if (target_mem_limit >= 0 && alloc_cur_ + aligned_size >= target_mem_limit) {
  //   return cudaErrorMemoryAllocation; 
  // }
  // alloc_num_ %= (BUFFER_SIZE / 2);
  // if (alloc_num_ + aligned_size >= BUFFER_SIZE) {
  //   alloc_num_ = 0;
  // }
  // *devPtr = (DTYPE *)this->devPtr_ + alloc_num_ ;
  // while (size_.find(*devPtr) != size_.end()) {
  //   *devPtr = (DTYPE*)(*devPtr) + GRANULARITY;
  // }
  // if (*devPtr >= devPtr + BUFFER_SIZE)
  //   return cudaErrorMemoryAllocation; 
  // size_[*devPtr] = size;
  alloc_cur_ += size;
  // alloc_num_ += aligned_size;
  // if (alloc_cur_ > alloc_max_)
  //   alloc_max_ = alloc_cur_;
  *devPtr = (DTYPE*)devPtr_ + alloc_num_;
  alloc_num_ += aligned_size;
  return cudaSuccess;
}


cudaError_t Allocator::free(void *devPtr) {
  std::lock_guard<std::mutex> guard(mutex_);
  // auto size = size_[devPtr];
  // alloc_cur_ -= size;
  // size_.erase(devPtr);
  return cudaSuccess;
}

}  // end pytorch_malloc

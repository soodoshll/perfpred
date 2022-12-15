#include <stdio.h>
#include <dlfcn.h>
#include <assert.h> 

#include "allocator.h"

#define DTYPE char
#define GRANULARITY 512
#define BASE 1024

namespace pytorch_malloc {

void Allocator::set_target_mem_limit(size_t size) {
  // printf("[Fake allocator] Initialized, size %lu\n", size);
  assert(this->pool.empty());
  this->pool.emplace_back(0, size-INIT_USAGE, false);
  this->target_mem_limit = size;
}

static inline std::size_t ceilInt(std::size_t m, std::size_t n) {
    assert(n > 0);
    if (m == 0) return n;
    return (m + n-1) / n * n;
}

Allocator::Allocator() {
}

Allocator::~Allocator() {
}

cudaError_t Allocator::malloc(void **devPtr, size_t size) {
  std::lock_guard<std::mutex> guard(mutex_);
  if (this->pool.empty()) set_target_mem_limit(this->target_mem_limit);

  auto aligned_size = ceilInt(size, GRANULARITY);
  // auto ptr = reinterpret_cast<size_t>(devPtr);
  for (auto itr = pool.begin() ; itr != pool.end() ; itr++) {
    if (!itr->used && itr->length() >= aligned_size) {
      // split
      // printf("[split][before] %lx %lx \n", itr->start, itr->end);
      size_t new_start = itr->start + aligned_size;
      auto itr_next = itr; itr_next++;
      pool.insert(itr_next, PoolNode(new_start, itr->end, false));
      itr->end = new_start;
      itr->used = true;
      // printf("[split][after] %lx %lx \n", itr->start, itr->end);
      *devPtr = reinterpret_cast<void*>(itr->start + BASE);
      alloc_ += aligned_size;
      if (alloc_ > alloc_max_) alloc_max_ = alloc_;
      // printf("[low][malloc] %ld -- %ld, %lu\n", itr->start, itr->end, size);
      return cudaSuccess;
    }
  }
  return cudaErrorMemoryAllocation;
}


cudaError_t Allocator::free(void *devPtr) {
  std::lock_guard<std::mutex> guard(mutex_);
  // printf("[start free]\n");
  auto ptr = reinterpret_cast<size_t>(devPtr) - BASE;
  auto itr = pool.begin();
  for ( ; itr != pool.end() ; itr++) {
    if (ptr == itr->start) {
      itr->used = false;
      break;  
    }  
  }
  assert (itr != pool.end());
  auto size = itr->length();
  alloc_ -= itr->length();
  // printf("[free] %ld, %ld -- %ld\n", ptr, itr->start, itr->end);
  auto itr_next = itr; itr_next++;
  while (itr_next != pool.end() && !itr_next->used) {
    itr->end = itr_next->end;
    pool.erase(itr_next);
    itr_next = itr;
    itr_next ++;
  }
  auto itr_prev = itr; itr_prev--;
  while (itr_prev != pool.begin() && !itr_prev->used) {
    itr->start = itr_prev->start;
    pool.erase(itr_prev);
    itr_prev = itr;
    itr_prev --;
  }
  return cudaSuccess;
}

}  // end pytorch_malloc

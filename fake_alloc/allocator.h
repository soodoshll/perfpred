#ifndef ALLOCATOR_H
#define ALLOCATOR_H
#include <cuda_runtime_api.h>
#include <unordered_map>
#include <mutex>

constexpr size_t BUFFER_SIZE = (size_t)8 * 1024 * 1024 * 1024;
namespace pytorch_malloc {

class Allocator {
 public:
  Allocator();
  ~Allocator();

  static Allocator *Instance() {
    static Allocator allocator;
    return &allocator;
  }

  cudaError_t malloc(void **devPtr, size_t size);
  cudaError_t free(void *devPtr);

  size_t max_mem_allocated() {
    return alloc_max_;
  }
  
  void init_max_mem() {
    alloc_max_ = alloc_cur_;
  }

  void set_target_mem_limit(size_t x) {
    target_mem_limit = x;
  }

  size_t get_mem_limit() {
    return target_mem_limit;
  }

  size_t get_free_space() {
    return target_mem_limit - alloc_cur_;
  }

 private:
  void *devPtr_ = nullptr;
  size_t alloc_num_ = 1024;
  long long alloc_cur_ = 0;
  long long alloc_max_ = 0;
  std::unordered_map<void*, size_t> size_;
  // size_t target_mem_limit = -1;
  size_t target_mem_limit = (size_t)1024 * 1024 * 1024 * 128;
  std::mutex mutex_;
 };

}  // namespace pytorch_malloc

#endif  // ALLOCATOR_H

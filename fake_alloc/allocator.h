#ifndef ALLOCATOR_H
#define ALLOCATOR_H
#include <cuda_runtime_api.h>
#include <unordered_map>
// #include <vector>
// 
// enum cudaError_t : int;

constexpr size_t BUFFER_SIZE = (size_t)9 * 1024 * 1024 * 1024;
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

 private:
  void *devPtr_ = nullptr;
  size_t alloc_num_ = 0;
  // size_t size_[10000];
  long long alloc_cur_ = 0;
  long long alloc_max_ = 0;
  // std::vector<size_t> size_;
  std::unordered_map<void*, size_t> size_;
};

}  // namespace pytorch_malloc

#endif  // ALLOCATOR_H
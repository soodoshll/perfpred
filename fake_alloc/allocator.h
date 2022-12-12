#ifndef ALLOCATOR_H
#define ALLOCATOR_H
#include <cuda_runtime_api.h>
#include <unordered_map>
#include <mutex>
#include <list>

constexpr size_t BUFFER_SIZE = (size_t)8 * 1024 * 1024 * 1024;
namespace pytorch_malloc {

struct PoolNode {
  size_t start, end;
  bool used = false;
  PoolNode(size_t start, size_t end, bool used) : start(start), end(end), used(used) {}
  size_t length() { return end - start;}
};

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
    alloc_max_ = alloc_;
  }

  void set_target_mem_limit(size_t x);

  size_t get_mem_limit() {
    return target_mem_limit;
  }

  size_t get_free_space() {
    return target_mem_limit - alloc_;
  }

 private:
  // void *devPtr_ = nullptr;
  size_t alloc_max_ = 0;
  size_t alloc_ = 0;
  std::list<PoolNode> pool;
  std::unordered_map<void*, size_t> size_;
  size_t target_mem_limit = (size_t)1024 * 1024 * 1024 * 128;
  std::mutex mutex_;
 };

}  // namespace pytorch_malloc

#endif  // ALLOCATOR_H

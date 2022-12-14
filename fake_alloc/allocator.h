#ifndef ALLOCATOR_H
#define ALLOCATOR_H
#include <cuda_runtime_api.h>
#include <unordered_map>
#include <mutex>
#include <list>

constexpr size_t BUFFER_SIZE = (size_t)8 * 1024 * 1024 * 1024;
constexpr size_t INIT_USAGE = 0;
namespace pytorch_malloc {

}  // namespace pytorch_malloc

#endif  // ALLOCATOR_H

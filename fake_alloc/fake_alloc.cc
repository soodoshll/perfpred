#include <pybind11/pybind11.h>
#include "allocator.h"

namespace py = pybind11;
size_t max_mem_allocated() {
    auto allocator = pytorch_malloc::Allocator::Instance();
    return allocator->max_mem_allocated();
}

PYBIND11_MODULE(fake_alloc, m) {
    m.doc() = "utilities for the fake allocator";
    m.def("max_mem_allocated", &max_mem_allocated);
}
#include <pybind11/pybind11.h>
#include "allocator.h"

namespace py = pybind11;
size_t max_mem_allocated() {
    auto allocator = pytorch_malloc::Allocator::Instance();
    return allocator->max_mem_allocated();
}

void init_max_mem() {
    auto allocator = pytorch_malloc::Allocator::Instance();
    allocator->init_max_mem();
}

PYBIND11_MODULE(fake_alloc, m) {
    m.doc() = "utilities for the fake allocator";
    m.def("max_mem_allocated", &max_mem_allocated);
    m.def("init_max_mem", &init_max_mem);
}
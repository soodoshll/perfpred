c++ -O3 -Wall -shared -std=c++11 -fPIC -I/usr/local/cuda/include $(python3 -m pybind11 --includes) fake_alloc.cc -o ../fake_alloc.so
nvcc cudart.cc allocator.cc cnmem/cnmem.cpp --compiler-options '-fPIC' -shared --cudart=none -o ../fake_libcudart.so

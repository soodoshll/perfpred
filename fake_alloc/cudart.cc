#include <cuda_runtime_api.h>
#include "allocator.h"
#include <cstdio>
// #include <cudnn.h>

extern "C" {

cudaError_t cudaMalloc(void **devPtr, size_t size) {
  pytorch_malloc::Allocator *allocator = pytorch_malloc::Allocator::Instance();
  return allocator->malloc(devPtr, size);
}

cudaError_t cudaFree(void *devPtr) {
  pytorch_malloc::Allocator *allocator = pytorch_malloc::Allocator::Instance();
  return allocator->free(devPtr);
}

cudaError_t cudaMemGetInfo (size_t* free, size_t* total) {
  *free = BUFFER_SIZE;
  *total = BUFFER_SIZE;
  return cudaSuccess;
}

__host__ cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream) {
  return cudaSuccess;
}

__host__ cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind) {
  return cudaSuccess;
}

__host__ cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream) {
  return cudaSuccess;
}


/*
qidong: I tried to overwrite the cudnn calls to avoid useless computation but didn't succeed.
*/

// cudnnStatus_t cudnnConvolutionForward(
//     cudnnHandle_t                       handle,
//     const void                         *alpha,
//     const cudnnTensorDescriptor_t       xDesc,
//     const void                         *x,
//     const cudnnFilterDescriptor_t       wDesc,
//     const void                         *w,
//     const cudnnConvolutionDescriptor_t  convDesc,
//     cudnnConvolutionFwdAlgo_t           algo,
//     void                               *workSpace,
//     size_t                              workSpaceSizeInBytes,
//     const void                         *beta,
//     const cudnnTensorDescriptor_t       yDesc,
//     void                               *y) {
//       // printf("you're fucked, conv2d forward\n");
//       return CUDNN_STATUS_SUCCESS;
//     }

// cudnnStatus_t cudnnConvolutionBackwardBias(
//     cudnnHandle_t                    handle,
//     const void                      *alpha,
//     const cudnnTensorDescriptor_t    dyDesc,
//     const void                      *dy,
//     const void                      *beta,
//     const cudnnTensorDescriptor_t    dbDesc,
//     void                            *db) {
//       // printf("you're fucked, conv2d backward bias\n");
//       return CUDNN_STATUS_SUCCESS; 
//     }

// cudnnStatus_t cudnnConvolutionBackwardFilter(
//     cudnnHandle_t                       handle,
//     const void                         *alpha,
//     const cudnnTensorDescriptor_t       xDesc,
//     const void                         *x,
//     const cudnnTensorDescriptor_t       dyDesc,
//     const void                         *dy,
//     const cudnnConvolutionDescriptor_t  convDesc,
//     cudnnConvolutionBwdFilterAlgo_t     algo,
//     void                               *workSpace,
//     size_t                              workSpaceSizeInBytes,
//     const void                         *beta,
//     const cudnnFilterDescriptor_t       dwDesc,
//     void                               *dw) {
//       // printf("you're fucked, conv2d backward filter\n");
//       return CUDNN_STATUS_SUCCESS;       
//     }

// cudnnStatus_t cudnnConvolutionBiasActivationForward(
//     cudnnHandle_t                       handle,
//     const void                         *alpha1,
//     const cudnnTensorDescriptor_t       xDesc,
//     const void                         *x,
//     const cudnnFilterDescriptor_t       wDesc,
//     const void                         *w,
//     const cudnnConvolutionDescriptor_t  convDesc,
//     cudnnConvolutionFwdAlgo_t           algo,
//     void                               *workSpace,
//     size_t                              workSpaceSizeInBytes,
//     const void                         *alpha2,
//     const cudnnTensorDescriptor_t       zDesc,
//     const void                         *z,
//     const cudnnTensorDescriptor_t       biasDesc,
//     const void                         *bias,
//     const cudnnActivationDescriptor_t   activationDesc,
//     const cudnnTensorDescriptor_t       yDesc,
//     void                               *y) {
//       // printf("you're fucked, conv2d bias activation forward\n");
//       return CUDNN_STATUS_SUCCESS;
//     }

}


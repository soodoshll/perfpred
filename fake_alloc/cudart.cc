#include <cuda_runtime_api.h>
#include <cuda.h>
#include "allocator.h"
#include <cstdio>
#include <cudnn.h>

extern "C" {

// cudaError_t cudaMalloc(void **devPtr, size_t size) {
//   // printf("alloc\n");
//   pytorch_malloc::Allocator *allocator = pytorch_malloc::Allocator::Instance();
//   return allocator->malloc(devPtr, size);
// }

// cudaError_t cudaFree(void *devPtr) {
//   pytorch_malloc::Allocator *allocator = pytorch_malloc::Allocator::Instance();
//   return allocator->free(devPtr);
// }

// __host__ cudaError_t cudaMemGetInfo (size_t* free, size_t* total) {
//   pytorch_malloc::Allocator *allocator = pytorch_malloc::Allocator::Instance();
//   *free = allocator->get_free_space();
//   *total = allocator->get_mem_limit();
//   // printf("[meminfo low] %lu %lu\n", *free, *total);
//   return cudaSuccess;
// }

__host__ cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream) {
  return cudaSuccess;
}

CUresult cuLaunchKernel ( CUfunction f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, unsigned int  sharedMemBytes, CUstream hStream, void** kernelParams, void** extra ) {
  return CUDA_SUCCESS;
}

// cudnnStatus_t cudnnCreateTensorDescriptor(
//     cudnnTensorDescriptor_t *tensorDesc) {
//       void *ptr;
//       ptr = malloc(256);
//       *tensorDesc = (cudnnTensorDescriptor_t)ptr;
//       return CUDNN_STATUS_SUCCESS;
//     }

// __host__ cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind) {
//   return cudaSuccess;
// }

// __host__ cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream) {
//   return cudaSuccess;
// }

// __host__ cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
//   return cudaSuccess;
// }

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

// cudnnStatus_t cudnnConvolutionBackwardData(
//     cudnnHandle_t                       handle,
//     const void                         *alpha,
//     const cudnnFilterDescriptor_t       wDesc,
//     const void                         *w,
//     const cudnnTensorDescriptor_t       dyDesc,
//     const void                         *dy,
//     const cudnnConvolutionDescriptor_t  convDesc,
//     cudnnConvolutionBwdDataAlgo_t       algo,
//     void                               *workSpace,
//     size_t                              workSpaceSizeInBytes,
//     const void                         *beta,
//     const cudnnTensorDescriptor_t       dxDesc,
//     void                               *dx) {
//       printf("you're fucked, conv2d backward data\n");
//       return CUDNN_STATUS_SUCCESS;      
//     }

//   cudnnStatus_t cudnnBatchNormalizationBackwardEx (
//       cudnnHandle_t                       handle,
//       cudnnBatchNormMode_t                mode,
//       cudnnBatchNormOps_t                 bnOps,
//       const void                          *alphaDataDiff,
//       const void                          *betaDataDiff,
//       const void                          *alphaParamDiff,
//       const void                          *betaParamDiff,
//       const cudnnTensorDescriptor_t       xDesc,
//       const void                          *xData,
//       const cudnnTensorDescriptor_t       yDesc,
//       const void                          *yData,
//       const cudnnTensorDescriptor_t       dyDesc,
//       const void                          *dyData,
//       const cudnnTensorDescriptor_t       dzDesc,
//       void                                *dzData,
//       const cudnnTensorDescriptor_t       dxDesc,
//       void                                *dxData,
//       const cudnnTensorDescriptor_t       dBnScaleBiasDesc,
//       const void                          *bnScaleData,
//       const void                          *bnBiasData,
//       void                                *dBnScaleData,
//       void                                *dBnBiasData,
//       double                              epsilon,
//       const void                          *savedMean,
//       const void                          *savedInvVariance,
//       const cudnnActivationDescriptor_t   activationDesc,
//       void                                *workspace,
//       size_t                              workSpaceSizeInBytes,
//       void                                *reserveSpace,
//       size_t                              reserveSpaceSizeInBytes) {
//         // printf("you're fucked, bn backward\n");
//         return CUDNN_STATUS_SUCCESS;          
//       }
  // cudnnStatus_t cudnnIm2Col(
  //   cudnnHandle_t                   handle,
  //   cudnnTensorDescriptor_t         srcDesc,
  //   const void                      *srcData,
  //   cudnnFilterDescriptor_t         filterDesc,   
  //   cudnnConvolutionDescriptor_t    convDesc,
  //   void                            *colBuffer) {

  //       // printf("you're fucked, im2col\n");
  //       return CUDNN_STATUS_SUCCESS;          
  //   }
}


#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

using gyblasHandle_t = cublasHandle_t;
using gyblas_status_t = cublasStatus_t;

template <class T>
void wrapGemm(cublasHandle_t handle, int m, int n, int k,
    T alpha, const T* A, const T* B, T beta, T* C) 
{
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    m, n, k,
                    &alpha,
                    A, m,
                    B, k,
                    &beta,
                    C, m);
}

template <>
void wrapGemm<float>(cublasHandle_t handle, int m, int n, int k,
    float alpha, const float* A, const float* B, float beta, float* C){

        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    m, n, k,
                    &alpha,
                    A, m,
                    B, k,
                    &beta,
                    C, m);
}

inline
void gyCheckBlasError(cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << msg << ": cuBLAS error" << std::endl;
        exit(EXIT_FAILURE);
    }
}


inline cublasStatus_t gyblasCreate(cublasHandle_t* p_handle) {
    return cublasCreate(p_handle);
}


inline cublasStatus_t gyblasDestroy(cublasHandle_t handle) {
    return cublasDestroy(handle);
}

inline cublasStatus_t gyblasEnableTensorCore(cublasHandle_t handle) {
    return cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
}

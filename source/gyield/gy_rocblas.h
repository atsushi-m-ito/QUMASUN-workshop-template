#pragma once

#include <rocblas/rocblas.h>

using gyblas_status = rocblas_status;
using gyblasHandle_t = rocblas_handle;

inline
void gyCheckBlasError(rocblas_status status, const char* msg) {
    if (status != rocblas_status_success) {
        std::cerr << msg << ": rocBLAS error" << std::endl;
        exit(EXIT_FAILURE);
    }
}


template <class T>
void wrapGemm(rocblas_handle handle, int m, int n, int k,
    T alpha, const T* A, const T* B, T beta, T* C) 
{
    rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_none,
                  m, n, k,
                  &alpha,
                  A, m,
                  B, k,
                  &beta,
                  C, m);
}

template <>
void wrapGemm<float>(rocblas_handle handle, int m, int n, int k,
    float alpha, const float* A, const float* B, float beta, float* C) {

    rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none,
                  m, n, k,
                  &alpha,
                  A, m,
                  B, k,
                  &beta,
                  C, m);
}


inline gyblas_status gyblasCreate(rocblas_handle* p_handle) {
    return rocblas_create_handle(p_handle);
}


inline gyblas_status gyblasDestroy(rocblas_handle handle) {
    return rocblas_destroy_handle(handle);
}


inline gyblas_status gyblasEnableTensorCore(rocblas_handle handle) {
    return rocblas_status_success;
}

#pragma once
#include "gyield.h"

#include "gy_cusolverDn.h"
#include <cuda_runtime.h>



namespace gy {

    namespace lapack {


        template<typename MYCOMPLEX>
        int ZHEGVD(char UPLO, const MYCOMPLEX* A, const MYCOMPLEX* B, double* eigen_values, MYCOMPLEX* eigen_vectors, MYCOMPLEX* triangle_factor, int N) {



            int info = 0;

            gysolverDnHandle_t handle = SingletonHandle::Get();


            //Because lapack input matrix and output eigen vector is same pointor, A should be copied to eiven_vectors buffer//
            cuDoubleComplex* a = (cuDoubleComplex*)eigen_vectors;
            gyMemcpyAsync(a, A, N * N * sizeof(cuDoubleComplex), gyMemcpyDeviceToDevice);

            cuDoubleComplex* b = (cuDoubleComplex*)triangle_factor;
            if (b == nullptr) {
                gyMalloc((void**)&b, N * N * sizeof(cuDoubleComplex));
            }
            gyMemcpyAsync(b, B, N * N * sizeof(cuDoubleComplex), gyMemcpyDeviceToDevice);


            int LWORK = 0;
            //workサイズ取得//
            cusolverDnZhegvd_bufferSize(
                handle,
                CUSOLVER_EIG_TYPE_1,
                CUSOLVER_EIG_MODE_VECTOR,
                (UPLO == 'U') ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER,
                N,
                a, N, b, N,
                eigen_values,
                &LWORK
            );
            gyDeviceSynchronize();

            cuDoubleComplex* WORK = nullptr;
            gyMalloc((void**)&WORK, LWORK * sizeof(cuDoubleComplex));
            int* dINFO;
            gyMalloc((void**)&dINFO, sizeof(int));

            cusolverDnZhegvd(
                handle,
                CUSOLVER_EIG_TYPE_1,
                CUSOLVER_EIG_MODE_VECTOR,
                (UPLO == 'U') ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER,
                N,
                a, N, b, N,
                eigen_values,
                WORK,
                LWORK,
                dINFO);


            gyMemcpyAsync(&info, dINFO, sizeof(int), gyMemcpyDeviceToHost);
            gyDeviceSynchronize();

            if (info != 0) {
                printf(" info=%d\n", info);

            }

            gyFree(WORK);
            gyFree(dINFO);
            if (triangle_factor == nullptr) {
                gyFree(b);
            }

            return info;
        }
    }
}

template<typename MYCOMPLEX>
int lapack_ZHEGVD(char UPLO, const MYCOMPLEX* A, const MYCOMPLEX* B, double* eigen_values, MYCOMPLEX* eigen_vectors, MYCOMPLEX* triangle_factor, int N) {
    return gy::lapack::ZHEGVD< MYCOMPLEX >(UPLO, A, B, eigen_values, eigen_vectors, triangle_factor, N);
}


template<typename MYCOMPLEX>
int lapack_ZHEGVD(char UPLO, const MYCOMPLEX* A, const MYCOMPLEX* B, double* eigen_values, MYCOMPLEX* eigen_vectors, int N) {
    return gy::lapack::ZHEGVD< MYCOMPLEX >(UPLO, A, B, eigen_values, eigen_vectors, nullptr, N);
}


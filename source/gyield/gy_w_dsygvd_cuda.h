#pragma once
#include "gyield.h"

#include "gy_cusolverDn.h"
#include <cuda_runtime.h>



namespace gy {

    namespace lapack {



        inline
        int DSYGVD(char UPLO, const double* A, const double* B, double* eigen_values, double* eigen_vectors, double* triangle_factor, int N)
        {


            int info = 0;

            gysolverDnHandle_t handle = SingletonHandle::Get();



            //Because lapack input matrix and output eigen vector is same pointor, A should be copied to eiven_vectors buffer//
            double* a = eigen_vectors;
            gyMemcpyAsync(a, A, N * N * sizeof(double), gyMemcpyDeviceToDevice);

            double* b = triangle_factor;
            if (b == nullptr) {
                gyMalloc((void**)&b, N * N * sizeof(double));
            }
            gyMemcpyAsync(b, B, N * N * sizeof(double), gyMemcpyDeviceToDevice);

            int LWORK = 0;
            //workサイズ取得//
            cusolverDnDsygvd_bufferSize(
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

            double* WORK = nullptr;
            gyMalloc((void**)&WORK, LWORK * sizeof(double));
            int* dINFO;
            gyMalloc((void**)&dINFO, sizeof(int));

            cusolverDnDsygvd(
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

inline
int lapack_DSYGVD(char UPLO, const double* A, const double* B, double* eigen_values, double* eigen_vectors, double* triangle_factor, int N) {
    return gy::lapack::DSYGVD(UPLO, A, B, eigen_values, eigen_vectors, triangle_factor, N);
}


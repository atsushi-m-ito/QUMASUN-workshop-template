#pragma once

#include "gyield.h"
#include "gy_cusolverDn.h"
#include <cuda_runtime.h>


namespace gy {

    namespace lapack {

        inline
        int DPOTRF(char UPLO, const double* A, int N, double* triangle_factor) {

            int info = 0;

            gysolverDnHandle_t handle = SingletonHandle::Get();

            

            if (A != triangle_factor) {
                gyMemcpy(triangle_factor, A, sizeof(double) * N * N, gyMemcpyDeviceToDevice);
            }

            int LWORK = 0;            
            cusolverDnDpotrf_bufferSize(
                handle,
                (UPLO == 'U') ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER,
                N, triangle_factor, N, &LWORK);
            cudaDeviceSynchronize();

            double* WORK;
            cudaMalloc((void**)&WORK, sizeof(double) * (LWORK));
            int* dINFO;
            cudaMalloc((void**)&dINFO, sizeof(int));

            cusolverDnDpotrf(
                handle,
                (UPLO == 'U') ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER,
                N, triangle_factor, N, WORK, LWORK, dINFO);

            gyDeviceSynchronize();
            gyMemcpyAsync(&info, dINFO, sizeof(int), gyMemcpyDeviceToHost);
            gyDeviceSynchronize();

            if (info != 0) {
                printf(" info=%d\n", info);

            }
    
            gyFree(WORK);
            gyFree(dINFO);

            return info;
        }

        template<typename MYCOMPLEX>
        int ZPOTRF(char UPLO, const MYCOMPLEX* A, int N, MYCOMPLEX* triangle_factor) {

            int info = 0;

            gysolverDnHandle_t handle = SingletonHandle::Get();


            if (A != triangle_factor) {
                gyMemcpy(triangle_factor, A, sizeof(MYCOMPLEX) * N * N, gyMemcpyDeviceToDevice);
            }

            
            int LWORK = 0;            
            cusolverDnZpotrf_bufferSize(
                handle,
                (UPLO == 'U') ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER,
                N, (cuDoubleComplex*)triangle_factor, N, &LWORK);
            cudaDeviceSynchronize();

            cuDoubleComplex* WORK;
            cudaMalloc((void**)&WORK, sizeof(cuDoubleComplex ) * (LWORK));
            int* dINFO;
            cudaMalloc((void**)&dINFO, sizeof(int));

            cusolverDnZpotrf(
                handle,
                (UPLO == 'U') ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER,
                N, (cuDoubleComplex*)triangle_factor, N, WORK, LWORK, dINFO);
            gyDeviceSynchronize();
            gyMemcpyAsync(&info, dINFO, sizeof(int), gyMemcpyDeviceToHost);
            gyDeviceSynchronize();




            if (info != 0) {
                printf(" info=%d\n", info);

            }

            gyFree(WORK);
            gyFree(dINFO);
            
            return info;
        }

    }
}

inline
int lapack_DPOTRF(char UPLO, int N, const double* A, double* triangle_factor) {
    return gy::lapack::DPOTRF(UPLO,  A,  N, triangle_factor);
}


template<typename MYCOMPLEX>
int lapack_ZPOTRF(char UPLO, int N, const MYCOMPLEX* A, MYCOMPLEX* triangle_factor) {
    
    return gy::lapack::ZPOTRF(UPLO, A, N, triangle_factor);
}
#pragma once

#include "gyield.h"
#include "gy_rocsolver.h"
#include <hip/hip_runtime.h>    


namespace gy {

    namespace lapack {

        inline
        int DPOTRF(char UPLO, const double* A, int N, double* triangle_factor) {

            int info = 0;

            gyblasHandle_t handle = SingletonHandle::Get();

            

            if (A != triangle_factor) {
                gyMemcpy(triangle_factor, A, sizeof(double) * N * N, gyMemcpyDeviceToDevice);
            }

            int* dINFO;
            gyMalloc((void**)&dINFO, sizeof(int));

            rocsolver_dpotrf(
                handle,
                (UPLO == 'U') ? rocblas_fill_upper : rocblas_fill_lower,
                N, triangle_factor, N, dINFO);

            gyMemcpyAsync(&info, dINFO, sizeof(int), gyMemcpyDeviceToHost);
            gyDeviceSynchronize();

            if (info != 0) {
                printf(" info=%d\n", info);

            }

            gyFree(dINFO);

            return info;
        }

        template<typename MYCOMPLEX>
        int ZPOTRF(char UPLO, const MYCOMPLEX* A, int N, MYCOMPLEX* triangle_factor) {

            int info = 0;

            gyblasHandle_t handle = SingletonHandle::Get();


            if (A != triangle_factor) {
                gyMemcpy(triangle_factor, A, sizeof(MYCOMPLEX) * N * N, gyMemcpyDeviceToDevice);
            }

            int* dINFO;
            gyMalloc((void**)&dINFO, sizeof(int));

            rocsolver_zpotrf(
                handle,
                (UPLO == 'U') ? rocblas_fill_upper : rocblas_fill_lower,
                N, (rocblas_double_complex*)triangle_factor, N, dINFO);
            
            gyMemcpyAsync(&info, dINFO, sizeof(int), gyMemcpyDeviceToHost);
            gyDeviceSynchronize();

            if (info != 0) {
                printf(" info=%d\n", info);

            }

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
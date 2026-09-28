#pragma once

#include "gyield.h"
#include "gy_rocsolver.h"
#include <hip/hip_runtime.h>    


namespace gy {

    namespace lapack {


        template<typename MYCOMPLEX>
        int ZHEGVD(char UPLO, const MYCOMPLEX* A, const MYCOMPLEX* B, double* eigen_values, MYCOMPLEX* eigen_vectors, MYCOMPLEX* triangle_factor, int N) {



            int info = 0;

            gyblasHandle_t handle = SingletonHandle::Get();

            char JOBZ = ((eigen_vectors) ? 'V' : 'N');
            int LWORK = 0;

            double* WORK = nullptr;
            gyMalloc((void**)&WORK, N * sizeof(double));

            int* dINFO;
            gyMalloc((void**)&dINFO, sizeof(int));


            //Because lapack input matrix and output eigen vector is same pointor, A should be copied to eiven_vectors buffer//
            rocblas_double_complex* a = (rocblas_double_complex*)eigen_vectors;
            gyMemcpyAsync(a, A, N * N * sizeof(rocblas_double_complex), gyMemcpyDeviceToDevice);

            rocblas_double_complex* b = (rocblas_double_complex*)triangle_factor;
            if (b == nullptr) {
                gyMalloc((void**)&b, N * N * sizeof(rocblas_double_complex));
            }
            gyMemcpyAsync(b, B, N * N * sizeof(rocblas_double_complex), gyMemcpyDeviceToDevice);


            rocsolver_zhegvd(
                handle,
                rocblas_eform_ax,
                rocblas_evect_original,
                (UPLO == 'U') ? rocblas_fill_upper : rocblas_fill_lower,
                N,
                a, N, b, N,
                eigen_values,
                WORK,
                dINFO
            );

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

        template<int LOOP, typename MYCOMPLEX >
        int ZHEGVD_loop(char UPLO, const MYCOMPLEX* A, const MYCOMPLEX* B, double* eigen_values, MYCOMPLEX* eigen_vectors, MYCOMPLEX* triangle_factor, int N) {



            int info = 0;

            gyblasHandle_t handle = SingletonHandle::Get();

            char JOBZ = ((eigen_vectors) ? 'V' : 'N');
            int LWORK = 0;

            double* WORK = nullptr;
            gyMalloc((void**)&WORK, N * sizeof(double));

            int* dINFO;
            gyMalloc((void**)&dINFO, sizeof(int));


            //Because lapack input matrix and output eigen vector is same pointor, A should be copied to eiven_vectors buffer//
            rocblas_double_complex* a = (rocblas_double_complex*)eigen_vectors;
            gyMemcpyAsync(a, A, N * N * sizeof(rocblas_double_complex), gyMemcpyDeviceToDevice);

            rocblas_double_complex* b = (rocblas_double_complex*)triangle_factor;
            if (b == nullptr) {
                gyMalloc((void**)&b, N * N * sizeof(rocblas_double_complex));
            }
            
            for (int istep = 0; istep < LOOP; ++istep) {

                gyMemcpyAsync(a, A, N * N * sizeof(rocblas_double_complex), gyMemcpyDeviceToDevice);
                gyMemcpyAsync(b, B, N * N * sizeof(rocblas_double_complex), gyMemcpyDeviceToDevice);

                rocsolver_zhegvd(
                    handle,
                    rocblas_eform_ax,
                    rocblas_evect_original,
                    (UPLO == 'U') ? rocblas_fill_upper : rocblas_fill_lower,
                    N,
                    a, N, b, N,
                    eigen_values,
                    WORK,
                    dINFO
                );

                gyMemcpyAsync(&info, dINFO, sizeof(int), gyMemcpyDeviceToHost);
                gyDeviceSynchronize();
            }
            

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
int lapack_ZHEGVD(char UPLO, const MYCOMPLEX* A, const MYCOMPLEX* B, double* eigen_values, MYCOMPLEX* eigen_vectors, MYCOMPLEX* triangle_factor, int N){
    return gy::lapack::ZHEGVD< MYCOMPLEX >(UPLO, A, B, eigen_values, eigen_vectors, triangle_factor, N);
}


template<typename MYCOMPLEX>
int lapack_ZHEGVD(char UPLO, const MYCOMPLEX* A, const MYCOMPLEX* B, double* eigen_values, MYCOMPLEX* eigen_vectors, int N) {
    return gy::lapack::ZHEGVD<MYCOMPLEX >(UPLO, A, B, eigen_values, eigen_vectors, nullptr, N);
}

template<int LOOP, typename MYCOMPLEX>
int lapack_ZHEGVD_loop(char UPLO, const MYCOMPLEX* A, const MYCOMPLEX* B, double* eigen_values, MYCOMPLEX* eigen_vectors, int N) {
    return gy::lapack::ZHEGVD_loop<LOOP, MYCOMPLEX >(UPLO, A, B, eigen_values, eigen_vectors, nullptr, N);
}


#pragma once
#include "gyield.h"

#include "gy_cusolverDn.h"
#include <cuda_runtime.h>



namespace gy {

    namespace lapack {



        //DSYEVD for symmetric matrix
        inline
        int DSYEVD(char UPLO, double* A, double* eigen_values, double* eigen_vectors, int N)
        {

            /*
                DSYEVD

                input:  N;
                input:  A[n][n];  matrix A
                output: eigen_vectors[n][n];  eigevectors
                output: eigen_values[n];    eigenvalues
            */

            gysolverDnHandle_t handle = SingletonHandle::Get();

            //Because lapack input matrix and output eigen vector is same pointor, A should be copied to eiven_vectors buffer//
            double* a = eigen_vectors;
            cudaMemcpyAsync(a, A, N * N * sizeof(double), cudaMemcpyDeviceToDevice);


            int LDA = N;
            int LWORK = 0;
            //workサイズ取得//
            cusolverDnDsyevd_bufferSize(
                handle,
                CUSOLVER_EIG_MODE_VECTOR, // 固有ベクトルも計算
                (UPLO == 'U') ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER,
                N,
                a, LDA,
                eigen_values,
                &LWORK
            );
            cudaDeviceSynchronize();

            double* WORK;
            cudaMalloc((void**)&WORK, sizeof(double) * (LWORK));
            int* dINFO;
            cudaMalloc((void**)&dINFO, sizeof(int));



            //eigen vector is returned in a//
            cusolverDnDsyevd(
                handle,
                CUSOLVER_EIG_MODE_VECTOR,
                (UPLO == 'U') ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER,
                N,
                a, LDA,
                eigen_values,
                WORK, LWORK,
                dINFO
            );

            int info = 0;
            cudaMemcpyAsync(&info, dINFO, sizeof(int), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            if (info != 0) {
                printf(" info=%d\n", info);

            }

            cudaFree(dINFO);
            cudaFree(WORK);

            return info;
        }
    }
}

inline
int lapack_DSYEVD(char UPLO, double* A, double* eigen_values, double* eigen_vectors, int N) {
    return gy::lapack::DSYEVD(UPLO, A, eigen_values, eigen_vectors, N);
}

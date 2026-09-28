#pragma once
#include "gyield.h"

#include "gy_cusolverDn.h"
#include <cuda_runtime.h>



namespace gy {

    namespace lapack {


        template<typename MYCOMPLEX>
        int ZHEEVD(char UPLO, const MYCOMPLEX* A, double* eigen_values, MYCOMPLEX* eigen_vectors, int N) {



            int info = 0;

            gysolverDnHandle_t handle = SingletonHandle::Get();


            //Because lapack input matrix and output eigen vector is same pointor, A should be copied to eiven_vectors buffer//
            cuDoubleComplex* a = (cuDoubleComplex*)eigen_vectors;
            if(eigen_vectors != A){
                gyMemcpyAsync(a, A, N * N * sizeof(cuDoubleComplex), gyMemcpyDeviceToDevice);
            }
            
            
            int LWORK = 0;
            //workサイズ取得//
            cusolverDnZheevd_bufferSize(
                handle,
                CUSOLVER_EIG_MODE_VECTOR,
                (UPLO == 'U') ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER,
                N,
                a, N,
                eigen_values,
                &LWORK
            );
            gyDeviceSynchronize();

            cuDoubleComplex* WORK = nullptr;
            gyMalloc((void**)&WORK, LWORK * sizeof(cuDoubleComplex));
            int* dINFO;
            gyMalloc((void**)&dINFO, sizeof(int));

            cusolverDnZheevd(
                handle,
                CUSOLVER_EIG_MODE_VECTOR,
                (UPLO == 'U') ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER,
                N,
                a, N, 
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
            
            return info;
        }
    }
}

template<typename MYCOMPLEX>
int lapack_ZHEEVD(char UPLO, const MYCOMPLEX* A, double* eigen_values, MYCOMPLEX* eigen_vectors,  int N) {
    return gy::lapack::ZHEEVD< MYCOMPLEX >(UPLO, A, eigen_values, eigen_vectors, N);
}




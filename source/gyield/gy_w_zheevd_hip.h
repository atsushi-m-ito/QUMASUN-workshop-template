#pragma once

#include "gyield.h"
#include "gy_rocsolver.h"
#include <hip/hip_runtime.h>    


namespace gy {

    namespace lapack {


        template<typename MYCOMPLEX>
        int ZHEEVD(char UPLO, const MYCOMPLEX* A, double* eigen_values, MYCOMPLEX* eigen_vectors, int N) {



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
            if(eigen_vectors != A){
                gyMemcpyAsync(a, A, N * N * sizeof(rocblas_double_complex), gyMemcpyDeviceToDevice);
            }

            rocsolver_zheevd(
                handle,
                rocblas_evect_original,
                (UPLO == 'U') ? rocblas_fill_upper : rocblas_fill_lower,
                N,
                a, N, 
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
            

            return info;
        }


    }
}

template<typename MYCOMPLEX>
int lapack_ZHEEVD(char UPLO, const MYCOMPLEX* A, double* eigen_values, MYCOMPLEX* eigen_vectors, int N){
    return gy::lapack::ZHEEVD< MYCOMPLEX >(UPLO, A, eigen_values, eigen_vectors, N);
}



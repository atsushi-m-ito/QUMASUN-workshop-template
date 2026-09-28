#pragma once

#include "gyield.h"
#include "gy_rocsolver.h"
#include <hip/hip_runtime.h>    


namespace gy {

    namespace lapack {

        

        inline
        int DSYGVD(char UPLO, const double* A, const double* B, double* eigen_values, double* eigen_vectors, double* triangle_factor, int N)
        {


        int info = 0;

        gyblasHandle_t handle = SingletonHandle::Get();

        char JOBZ = ((eigen_vectors) ? 'V' : 'N');
        int LWORK = 0;
        
        double *WORK = nullptr;        
        gyMalloc((void**)&WORK, N*sizeof(double));

        int *dINFO;
        gyMalloc((void**)&dINFO, sizeof(int));


        //Because lapack input matrix and output eigen vector is same pointor, A should be copied to eiven_vectors buffer//
        double* a = eigen_vectors;        
        gyMemcpyAsync(a, A, N*N*sizeof(double), gyMemcpyDeviceToDevice);

        double* b = triangle_factor;
        if (b == nullptr) {
            gyMalloc((void**)&b, N * N * sizeof(double));
        }
        gyMemcpyAsync(b, B, N * N * sizeof(double), gyMemcpyDeviceToDevice);

        
        rocsolver_dsygvd(
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
            printf(" info=%d\n",info);

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
int lapack_DSYGVD(char UPLO, const double* A, const double* B, double* eigen_values, double* eigen_vectors, double* triangle_factor, int N){
    return gy::lapack::DSYGVD(UPLO, A, B, eigen_values, eigen_vectors, triangle_factor, N);
}



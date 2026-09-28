#pragma once

#include "gyield.h"
#include "gy_rocsolver.h"
#include <hip/hip_runtime.h>    


namespace gy {

    namespace lapack {

        //note: rocsolver has DSYEVDJ routine, which is hybrid methods of Jacobi and divide-and-conquer methods. It will be tried in future.

        //DSYEVD for symmetric matrix
        inline
        int DSYEVD(char UPLO, double *A, double *eigen_values, double *eigen_vectors, int N)
        {

        /* 
            DSYEVD
        
            input:  N;
            input:  A[n][n];  matrix A
            output: eigen_vectors[n][n];  eigevectors
            output: eigen_values[n];    eigenvalues 
        */
            
        gyblasHandle_t handle = SingletonHandle::Get();

        char JOBZ = ((eigen_vectors) ? 'V' : 'N');
        int LDA=N;
        int LDZ=N;
        int LWORK = 0;
        
        double *WORK = nullptr;        
        gyMalloc((void**)&WORK, N*sizeof(double));

        int *dINFO;
        gyMalloc((void**)&dINFO, sizeof(int));


        //Because lapack input matrix and output eigen vector is same pointor, A should be copied to eiven_vectors buffer//
        double* a = eigen_vectors;        
        gyMemcpyAsync(a, A, N*N*sizeof(double), gyMemcpyDeviceToDevice);

#if 1
        rocsolver_dsyevd(
            handle,
            rocblas_evect_original,
            (UPLO == 'U') ? rocblas_fill_upper : rocblas_fill_lower, 
            N,
            a, LDA,
            eigen_values,
            WORK,
            dINFO
        );
#elif 0
        rocsolver_dsyevdj(
            handle,
            rocblas_evect_original,
            (UPLO == 'U') ? rocblas_fill_upper : rocblas_fill_lower, 
            N,
            a, LDA,
            eigen_values,
            //WORK,
            dINFO
        );
#else

        rocsolver_dsyev(
            handle,
            rocblas_evect_original,
            (UPLO == 'U') ? rocblas_fill_upper : rocblas_fill_lower, 
            N,
            a, LDA,
            eigen_values,
            WORK,
            dINFO
        );
#endif

        int info = 0;
        gyMemcpyAsync(&info, dINFO, sizeof(int), gyMemcpyDeviceToHost);
        gyDeviceSynchronize();
        
        if (info != 0) {
            printf(" info=%d\n",info);

        }
        
        gyFree(WORK);
        gyFree(dINFO);
        

        return info;
        }
    }
}


inline
int lapack_DSYEVD(char UPLO, double* A, double* eigen_values, double* eigen_vectors, int N) {
    return gy::lapack::DSYEVD(UPLO, A, eigen_values, eigen_vectors, N);
}

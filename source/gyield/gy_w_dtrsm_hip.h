#pragma once

#include "gyield.h"
#include "gy_rocblas.h"
#include <hip/hip_runtime.h>    


namespace gy {

    namespace blas {


        int DTRSM(char SIDE, char UPLO, char TRANS, char DIAG, int N, const double* triangle_factor, const double* A, double* RESULT) {


            gyblasHandle_t handle = SingletonHandle::Get();


            if (A != RESULT) {
                gyMemcpy(RESULT, A, sizeof(double) * N * N, gyMemcpyDeviceToDevice);
            }


            double ONE = 1.0;

            rocblas_dtrsm(handle, 
                (SIDE == 'L') ? rocblas_side_left : rocblas_side_right,
                (UPLO == 'U') ? rocblas_fill_upper : rocblas_fill_lower,
                (TRANS == 'N') ? rocblas_operation_none : (TRANS == 'T') ? rocblas_operation_transpose : rocblas_operation_conjugate_transpose,
                (DIAG == 'U') ? rocblas_diagonal_unit : rocblas_diagonal_non_unit,
                N, N, &ONE, triangle_factor, N, RESULT, N);




            return 0;
        }
        
        template<typename MYCOMPLEX>
        int ZTRSM(char SIDE, char UPLO, char TRANS, char DIAG, int N, const MYCOMPLEX* triangle_factor, const MYCOMPLEX* A, MYCOMPLEX* RESULT) {


            gyblasHandle_t handle = SingletonHandle::Get();


            if (A != RESULT) {
                gyMemcpy(RESULT, A, sizeof(MYCOMPLEX) * N * N, gyMemcpyDeviceToDevice);
            }

            MYCOMPLEX ONE{ 1.0,0.0 };

            rocblas_ztrsm(handle, 
                (SIDE == 'L') ? rocblas_side_left : rocblas_side_right,
                (UPLO == 'U') ? rocblas_fill_upper : rocblas_fill_lower,
                (TRANS == 'N') ? rocblas_operation_none : (TRANS == 'T') ? rocblas_operation_transpose : rocblas_operation_conjugate_transpose,
                (DIAG == 'U') ? rocblas_diagonal_unit : rocblas_diagonal_non_unit,
                N, N, (const rocblas_double_complex*)&ONE, (const rocblas_double_complex*)triangle_factor, N, (rocblas_double_complex*)RESULT, N);



            return 0;
        }

    }
}


inline
int blas_DTRSM(char SIDE, char UPLO, char TRANS, char is_unit, int N, const double* triangle_factor, const  double* A, double* RESULT)
{
    return gy::blas::DTRSM(SIDE, UPLO, TRANS, is_unit, N, triangle_factor, A, RESULT);
}

template<typename MYCOMPLEX>
int blas_ZTRSM(char SIDE, char UPLO, char TRANS, char is_unit, int N, const MYCOMPLEX* triangle_factor, const MYCOMPLEX* A, MYCOMPLEX* RESULT)
{
    return gy::blas::ZTRSM(SIDE, UPLO, TRANS, is_unit, N, triangle_factor, A, RESULT);
}

inline
int lapack_DTRTRS(char UPLO, char TRANS, char is_unit, int N, const double* triangle_factor, const double* A, double* RESULT)
{
    return gy::blas::DTRSM('L', UPLO, TRANS, is_unit, N, triangle_factor, A, RESULT);
}

template<typename MYCOMPLEX>
int lapack_ZTRTRS(char UPLO, char TRANS, char is_unit, int N, const MYCOMPLEX* triangle_factor, const MYCOMPLEX* A, MYCOMPLEX* RESULT)
{
    return gy::blas::ZTRSM('L', UPLO, TRANS, is_unit, N, triangle_factor, A, RESULT);
}

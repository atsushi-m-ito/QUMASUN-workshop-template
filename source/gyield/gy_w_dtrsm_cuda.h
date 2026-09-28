#pragma once

#include "gyield.h"
#include "gy_blas_init.h"

namespace gy {

    namespace blas {


        int DTRSM(char SIDE, char UPLO, char TRANS, char DIAG, int N, const double* triangle_factor, const double* A, double* RESULT) {


            gyblasHandle_t handle = SingletonHandle::Get();


            if (A != RESULT) {
                gyMemcpy(RESULT, A, sizeof(double) * N * N, gyMemcpyDeviceToDevice);
            }


            double ONE = 1.0;
            cublasDtrsm(handle, 
                (SIDE == 'L') ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT,
                (UPLO == 'U') ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER,
                (TRANS == 'N') ? CUBLAS_OP_N : (TRANS == 'T') ? CUBLAS_OP_T : CUBLAS_OP_C,
                (DIAG == 'U') ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
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

            cublasZtrsm(handle, 
                (SIDE == 'L') ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT,
                (UPLO == 'U') ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER,
                (TRANS == 'N') ? CUBLAS_OP_N : (TRANS == 'T') ? CUBLAS_OP_T : CUBLAS_OP_C,
                (DIAG == 'U') ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
                N, N, (const cuDoubleComplex*)&ONE, (const cuDoubleComplex*)triangle_factor, N, (cuDoubleComplex*)RESULT, N);



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

#pragma once

#ifdef GY_WITH_CUDA
#include "gyield/gy_w_dtrsm_cuda.h"
#elif defined (GY_WITH_HIP)
#include "gyield/gy_w_dtrsm_hip.h"
#else  //CPU

#include <stdio.h>
#include <stdlib.h>
#include "wrap_lapack.h"

//Solve
//A * X = B or A^t * X = B,
//where A is upper/lower triangle matrix//


#ifndef NO_LAPACKE

inline
int blas_DTRSM(char SIDE, char UPLO, char TRANS, char is_unit, int N, const double* L_or_U, const double* A, double* RESULT)
{

    int n = N;
    int LDA = n;

    if (A != RESULT) {
        memcpy(RESULT, A, sizeof(double) * N * N);
    }

    cblas_dtrsm(CblasColMajor, 
        (SIDE == 'R') ? CblasRight : CblasLeft, 
        (UPLO == 'U') ? CblasUpper : CblasLower,
        (TRANS== 'T') ? CblasTrans : (TRANS == 'C') ? CblasConjTrans : CblasNoTrans,
        (is_unit=='U') ? CblasUnit : CblasNonUnit,
         n, n, 1.0, L_or_U, n, RESULT, n);

    return 0;
}


template<typename MYCOMPLEX>
int blas_ZTRSM(char SIDE, char UPLO, char TRANS, char is_unit, int N, const MYCOMPLEX* L_or_U, const MYCOMPLEX* A, MYCOMPLEX* RESULT)
{

    int n = N;
    int LDA = n;
    const double ONE[2]{ 1.0,0.0 };

    if (A != RESULT) {
        memcpy(RESULT, A, sizeof(MYCOMPLEX) * N * N);
    }

    cblas_ztrsm(CblasColMajor,
        (SIDE == 'R') ? CblasRight : CblasLeft,
        (UPLO == 'U') ? CblasUpper : CblasLower,
        (TRANS == 'T') ? CblasTrans : (TRANS == 'C') ? CblasConjTrans : CblasNoTrans,
        (is_unit == 'U') ? CblasUnit : CblasNonUnit,
        n, n, ONE, L_or_U, n, RESULT, n);

    return 0;
}

inline
int lapack_DTRTRS(char UPLO, char TRANS, char is_unit, int N, const double* L_or_U, const double* A, double* RESULT)
{

  int n=N;
  int LDA=n;
  
  if (A != RESULT) {
      memcpy(RESULT, A, sizeof(double) * N * N);
  }

  int INFO = LAPACKE_dtrtrs(LAPACK_COL_MAJOR, UPLO, TRANS, is_unit, n, n , L_or_U, n, RESULT, n);


  
  if (INFO>0) {
    /* printf("\n%s: error in dsyevd_, info=%d\n\n",name,INFO); */
  }
  else if (INFO<0) {
    printf(" info=%d\n",INFO);

  }

   
  return INFO;
}


template<typename MYCOMPLEX>
int lapack_ZTRTRS(char UPLO, char TRANS, char is_unit, int N, const MYCOMPLEX* L_or_U, const MYCOMPLEX* A, MYCOMPLEX* RESULT)
{

    int n = N;
    int LDA = n;
    

    if (A != RESULT) {
        memcpy(RESULT, A, sizeof(lapack_complex_double) * N * N);
    }

    int INFO = LAPACKE_ztrtrs(LAPACK_COL_MAJOR, UPLO, TRANS, is_unit, n, n, (lapack_complex_double*)L_or_U, n, (lapack_complex_double*)RESULT, n);



    if (INFO > 0) {
        /* printf("\n%s: error in dsyevd_, info=%d\n\n",name,INFO); */
    } else if (INFO < 0) {
        printf(" info=%d\n", INFO);

    }

    return INFO;
}
#else


#ifdef LAPACK_NO_HEADER
extern "C" {
#ifdef _NEC
#define DTRTRS dtrtrs_
#define ZTRTRS ztrtrs_
#endif

    void DTRTRS(const char* uplo, const char* trans, const char* diag,
        const int* n, const int* nrhs, const double* a,
        const int* lda, double* b, const int* ldb,
        int* info);
    void ZTRTRS(const char* uplo, const char* trans, const char* diag,
        const int* n, const int* nrhs, const MKL_Complex16* a,
        const int* lda, MKL_Complex16* b, const int* ldb,
        int* info);
}
#endif

inline
int lapack_DTRTRS(char UPLO, char TRANS, char is_unit, int N, double* L_or_U, double* A, double* RESULT)
{

    INTEGER n = N;
    INTEGER LDA = n;
    INTEGER INFO;

    int i, j;

    if (A != RESULT) {
        memcpy(RESULT, A, sizeof(double) * N * N);
    }

    DTRTRS(&UPLO, &TRANS, &is_unit, &n, &n, L_or_U, &n, RESULT, &n, &INFO);



    if (INFO > 0) {
        /* printf("\n%s: error in dsyevd_, info=%d\n\n",name,INFO); */
    } else if (INFO < 0) {
        printf(" info=%d\n", INFO);

    } else { /* (INFO==0) */
        /* store eigenvectors */
          /*
        for (i=0;i<EVmax;i++) {
          for (j=0;j<n;j++) {
            a[i*n + j]= A[i*n+j];
          }
        }
        */
    }

    return INFO;
}



template<typename MYCOMPLEX>
int lapack_ZTRTRS(char UPLO, char TRANS, char is_unit, int N, MYCOMPLEX* L_or_U, MYCOMPLEX* A, MYCOMPLEX* RESULT)
{

    INTEGER n = N;
    INTEGER LDA = n;
    INTEGER INFO;

    int i, j;

    if (A != RESULT) {
        memcpy(RESULT, A, sizeof(lapack_complex_double) * N * N);
    }

    ZTRTRS(&UPLO, &TRANS, &is_unit, &n, &n, (MKL_Complex16*)L_or_U, &n, (MKL_Complex16*)RESULT, &n, &INFO);



    if (INFO > 0) {
        /* printf("\n%s: error in dsyevd_, info=%d\n\n",name,INFO); */
    } else if (INFO < 0) {
        printf(" info=%d\n", INFO);

    } else { /* (INFO==0) */
        /* store eigenvectors */
          /*
        for (i=0;i<EVmax;i++) {
          for (j=0;j<n;j++) {
            a[i*n + j]= A[i*n+j];
          }
        }
        */
    }

    return INFO;
}

#endif
#endif

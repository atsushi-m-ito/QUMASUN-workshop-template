#pragma once

#ifdef GY_WITH_CUDA
#include "gyield/gy_w_dpotrf_cuda.h"
#elif defined (GY_WITH_HIP)
#include "gyield/gy_w_dpotrf_hip.h"
#else  //CPU

#include <stdio.h>
#include <stdlib.h>
#include "wrap_lapack.h"

#ifndef NO_LAPACKE

inline
int lapack_DPOTRF(char UPLO, int N, const double* A, double* L_or_U)
{

  int n=N;
  int LDA=n;
  
  
  if (A != L_or_U) {
      memcpy(L_or_U, A, sizeof(double) * N * N);
  }

  int INFO = LAPACKE_dpotrf(LAPACK_COL_MAJOR, UPLO, n, L_or_U, n);


  
  if (INFO>0) {
    /* printf("\n%s: error in dsyevd_, info=%d\n\n",name,INFO); */
  }
  else if (INFO<0) {
    printf(" info=%d\n",INFO);

  }
   
  return INFO;
}



template<typename MYCOMPLEX>
int lapack_ZPOTRF(char UPLO, int N, const MYCOMPLEX* A, MYCOMPLEX* L_or_U)
{

    int n = N;
    int LDA = n;
    

    if (A != L_or_U) {
        memcpy(L_or_U, A, sizeof(lapack_complex_double) * N * N);
    }

    int INFO = LAPACKE_zpotrf(LAPACK_COL_MAJOR, UPLO, n, (lapack_complex_double*)L_or_U, n);



    if (INFO > 0) {
        /* printf("\n%s: error in dsyevd_, info=%d\n\n",name,INFO); */
    } else if (INFO < 0) {
        printf(" info=%d\n", INFO);
    } 

    return INFO;
}

#else


inline
int lapack_DPOTRF(char UPLO, int N, double* A, double* L_or_U)
{

    INTEGER n = N;
    INTEGER LDA = n;
    INTEGER INFO;

    int i, j;

    if (A != L_or_U) {
        memcpy(L_or_U, A, sizeof(double) * N * N);
    }

    DPOTRF(&UPLO, &n, L_or_U, &n, &INFO);



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
int lapack_ZPOTRF(char UPLO, int N, MYCOMPLEX* A, MYCOMPLEX* L_or_U)
{

    INTEGER n = N;
    INTEGER LDA = n;
    INTEGER INFO;

    int i, j;

    if (A != L_or_U) {
        memcpy(L_or_U, A, sizeof(lapack_complex_double) * N * N);
    }

    ZPOTRF(&UPLO, &n, (MKL_Complex16*)L_or_U, &n, &INFO);



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
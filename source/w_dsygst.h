#pragma once

#include <stdio.h>
#include <stdlib.h>
#include "wrap_lapack.h"

#ifndef NO_LAPACKE

/*
* 
*RESULTに生成されたL^{-1} A (L^t)^{-1} もしくは (U^t)^{-1} A U^{-1}も、
* UPLOの指定に合わせて上もしくは下三角領域しか有効な数値が入っていない
*/
inline
int lapack_DSYGST(char UPLO, int N, double* A, const double* L_or_U_from_B, double* RESULT)
{
  int itype = 1;//1 is AX=lambdaBX, 2 is ABX = lambdaX
  int n=N;
  int LDA=n;
  
  int i,j;

  if (A != RESULT) {
      memcpy(RESULT, A, sizeof(double) * N * N);
  }

  int INFO = LAPACKE_dsygst(LAPACK_COL_MAJOR, itype, UPLO, n, RESULT, n, L_or_U_from_B, n);


  
  if (INFO>0) {
    /* printf("\n%s: error in dsyevd_, info=%d\n\n",name,INFO); */
  }
  else if (INFO<0) {
    printf(" info=%d\n",INFO);

  }
   
  return INFO;
}


template<typename MYCOMPLEX>
int lapack_ZHEGST(char UPLO, int N, MYCOMPLEX* A, const MYCOMPLEX* L_or_U_from_B, MYCOMPLEX* RESULT)
{

    int itype = 1;//1 is AX=lambdaBX, 2 is ABX = lambdaX
    int n = N;
    int LDA = n;
    
    int i, j;

    if (A != RESULT) {
        memcpy(RESULT, A, sizeof(lapack_complex_double) * N * N);
    }

    int INFO = LAPACKE_zhegst(LAPACK_COL_MAJOR, itype, UPLO, n, (lapack_complex_double*)RESULT, n, (lapack_complex_double*)L_or_U_from_B, n);



    if (INFO > 0) {
        /* printf("\n%s: error in dsyevd_, info=%d\n\n",name,INFO); */
    } else if (INFO < 0) {
        printf(" info=%d\n", INFO);

    }

    return INFO;
}

#else

/*
*
*RESULTに生成されたL^{-1} A (L^t)^{-1} もしくは (U^t)^{-1} A U^{-1}も、
* UPLOの指定に合わせて上もしくは下三角領域しか有効な数値が入っていない
*/
inline
int lapack_DSYGST(char UPLO, int N, double* A, const double* L_or_U_from_B, double* RESULT)
{
    INTEGER itype = 1;//1 is AX=lambdaBX, 2 is ABX = lambdaX
    INTEGER n = N;
    INTEGER LDA = n;
    INTEGER INFO;

    int i, j;

    if (A != RESULT) {
        memcpy(RESULT, A, sizeof(double) * N * N);
    }

    DSYGST(&itype, &UPLO, &n, RESULT, &n, L_or_U_from_B, &n, &INFO);



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
int lapack_ZHEGST(char UPLO, int N, MYCOMPLEX* A, const MYCOMPLEX* L_or_U_from_B, MYCOMPLEX* RESULT)
{

    INTEGER itype = 1;//1 is AX=lambdaBX, 2 is ABX = lambdaX
    INTEGER n = N;
    INTEGER LDA = n;
    INTEGER INFO;

    int i, j;

    if (A != RESULT) {
        memcpy(RESULT, A, sizeof(lapack_complex_double) * N * N);
    }

    ZHEGST(&itype, &UPLO, &n, (MKL_Complex16*)RESULT, &n, (MKL_Complex16*)L_or_U_from_B, &n, &INFO);



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


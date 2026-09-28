#pragma once

#ifdef GY_WITH_CUDA
#include "gyield/gy_w_zheevd_cuda.h"
#elif defined (GY_WITH_HIP)
#include "gyield/gy_w_zheevd_hip.h"
#else  //CPU


#include <stdio.h>
#include <stdlib.h>
#include "wrap_lapack.h"

#ifndef NO_LAPACKE

template<class MYCOMPLEX>
int lapack_ZHEEVD(char UPLO, MYCOMPLEX*A, double *eigen_values, MYCOMPLEX* eigen_vectors, int N)
{

  /* 
     DSYEVD
  
     input:  N;
     input:  A[n][n];  matrix A
     output: eigen_vectors[n][n];  eigevectors
     output: eigen_values[n];    eigenvalues 
  */
    
  char  JOBZ = ((eigen_vectors) ? 'V' : 'N');
  
  int n=N;
  int LDA=n;
  int LDZ=n;
  

  //Because lapack input matrix and output eigen vector is same pointor, A should be copied to eiven_vectors buffer//
  MYCOMPLEX* a = eigen_vectors;
  if (A != a) {
      memcpy(a, A, sizeof(MYCOMPLEX) * N * N);
  }

  int INFO = LAPACKE_zheevd(LAPACK_COL_MAJOR, JOBZ, UPLO, n, (lapack_complex_double*)a, LDA, eigen_values);


  
  if (INFO>0) {
    /* printf("\n%s: error in dsyevd_, info=%d\n\n",name,INFO); */
  }
  else if (INFO<0) {
    printf(" info=%d\n",INFO);

  }



  return INFO;
}

#else


template<class MYCOMPLEX>
int lapack_ZHEEVD(char UPLO, MYCOMPLEX* A, double* eigen_values, MYCOMPLEX* eigen_vectors, int N)
{

    /*
       DSYEVD

       input:  N;
       input:  A[n][n];  matrix A
       output: eigen_vectors[n][n];  eigevectors
       output: eigen_values[n];    eigenvalues
    */


    char  JOBZ = ((eigen_vectors) ? 'V' : 'N');

    INTEGER n = N;
    INTEGER LDA = n;
    double VL, VU; /* dummy */
    INTEGER IL, IU;
    double ABSTOL = LAPACK_ABSTOL;
    INTEGER M;

    INTEGER LDZ = n;
    INTEGER LWORK, LIWORK;
    lapack_complex_double* WORK;
    INTEGER* IWORK;
    INTEGER INFO;

    int i, j;

    //A=(double*)malloc(sizeof(double)*n*n);
    /*
    LWORK=  1 + 6*n + 2*n*n;
    WORK=(double*)malloc(sizeof(double)*LWORK);

    LIWORK = 3 + 5*n;
    IWORK=(INTEGER*)malloc(sizeof(INTEGER)*LIWORK);
    */

    LWORK = 2 * N + N * N;
    WORK = (lapack_complex_double*)malloc(sizeof(lapack_complex_double) * LWORK);
    //memset(WORK, 0, sizeof(lapack_complex_double)*LWORK);		/* AITUNE */

    INTEGER LRWORK = 1 + 5 * N + 2 * N * N;
    double* RWORK = (double*)malloc(sizeof(double) * LRWORK);
    //memset(RWORK, 0, sizeof(double)*LRWORK);		/* AITUNE */

    LIWORK = 3 + 5 * N;
    IWORK = (INTEGER*)malloc(sizeof(INTEGER) * LIWORK);
    //memset(IWORK, 0, sizeof(INTEGER)*LIWORK);		/* AITUNE */

    IL = 1;
    IU = N;



    //Because lapack input matrix and output eigen vector is same pointor, A should be copied to eiven_vectors buffer//
    lapack_complex_double* a = ((eigen_vectors) ? (lapack_complex_double*)eigen_vectors : new lapack_complex_double[N * N]);

    for (i = 0; i < n * n; i++) {
        eigen_vectors[i] = A[i];
    }


    ZHEEVD(&JOBZ, &UPLO, &n, a, &LDA, eigen_values, WORK, &LWORK, RWORK, &LRWORK, IWORK, &LIWORK, &INFO);



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

    free(IWORK); free(WORK);

    if (eigen_vectors == NULL) {
        delete[] a;
    }

    return INFO;
}


#endif
#endif

template<class MYCOMPLEX>
int lapack_ZHEEVD(MYCOMPLEX* A, double* eigen_values, MYCOMPLEX* eigen_vectors, int N)
{
    return lapack_ZHEEVD('U', A, eigen_values, eigen_vectors, N);
}

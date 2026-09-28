#pragma once

#ifdef GY_WITH_CUDA
#include "gyield/gy_w_dsygvd_cuda.h"
#elif defined (GY_WITH_HIP)
#include "gyield/gy_w_dsygvd_hip.h"
#else  //CPU
#include <stdio.h>
#include <stdlib.h>
#include "wrap_lapack.h"

#ifndef NO_LAPACKE

inline
int lapack_DSYGVD(char UPLO, const double *A, const double *B, double *eigen_values, double *eigen_vectors, double *triangle_factor, int N)
{

  /* 
     DSYGVD
  
     input:  N;
     input:  A[n][n];  matrix A
     input:  B[n][n];  matrix B
     output: eigen_vectors[n][n];  eigevectors
     output: eigen_values[n];    eigenvalues 
	 output: triangle_factor[n][n];  triangle_factor U or L from the Cholesky factorization B = U^t * U or B = L * L^t

  */


	int ITYPE = 1;	// 1 indicate "A v = e B v" type problems//
	char  JOBZ = ((eigen_vectors) ? 'V' : 'N');  

  int n=N;
  int LDA=N;
  int LDB=N;
  int LDZ=n;
  

   //Because lapack input matrix and output eigen vector is same pointor, A should be copied to eiven_vectors buffer//
	//double* a = ((eigen_vectors) ? eigen_vectors : new double[N*N]);

  memcpy(eigen_vectors, A, sizeof(double) * N * N);
  double* b = ((triangle_factor) ? triangle_factor : new double[N*N]);
  memcpy(b, B, sizeof(double) * N * N);
  

  //DSYEVD( JOBZ, UPLO, &n, eigen_vectors, &LDA, eigen_values, WORK, &LWORK, IWORK, &LIWORK, &INFO ); 
    int INFO = LAPACKE_dsygvd(LAPACK_COL_MAJOR, ITYPE, JOBZ, UPLO, n, eigen_vectors, LDA, b, LDB, eigen_values);


  
  if (INFO>0) {
    /* printf("\n%s: error in dsyevd_, info=%d\n\n",name,INFO); */
  }
  else if (INFO<0) {
    printf(" info=%d\n",INFO);

  }

   
  
  if(triangle_factor == nullptr){
	  delete [] b;
  }

  return INFO;

}


#else

inline
int lapack_DSYGVD(char UPLO, const double* A, const double* B, double* eigen_values, double* eigen_vectors, double* triangle_factor, int N)
{

    /*
       DSYGVD

       input:  N;
       input:  A[n][n];  matrix A
       input:  B[n][n];  matrix B
       output: eigen_vectors[n][n];  eigevectors
       output: eigen_values[n];    eigenvalues
       output: triangle_factor[n][n];  triangle_factor U or L from the Cholesky factorization B = U^t * U or B = L * L^t

    */


    INTEGER ITYPE = 1;	// 1 indicate "A v = e B v" type problems//
    char  JOBZ = ((eigen_vectors) ? 'V' : 'N');




    INTEGER n = N;
    INTEGER LDA = N;
    INTEGER LDB = N;

    double VL, VU; /* dummy */
    INTEGER IL, IU;
    double ABSTOL = LAPACK_ABSTOL;
    INTEGER M;

    INTEGER LDZ = n;
    INTEGER LWORK, LIWORK;
    double* WORK;
    INTEGER* IWORK;
    INTEGER INFO;

    int i, j;

    //A=(double*)malloc(sizeof(double)*n*n);

    LWORK = 1 + 6 * n + 2 * n * n;
    WORK = (double*)malloc(sizeof(double) * LWORK);

    LIWORK = 3 + 5 * n;
    IWORK = (INTEGER*)malloc(sizeof(INTEGER) * LIWORK);


    IL = 1;
    IU = N;

    //Because lapack input matrix and output eigen vector is same pointor, A should be copied to eiven_vectors buffer//
    double* a = ((eigen_vectors) ? eigen_vectors : new double[N * N]);

    for (i = 0; i < n * n; i++) {
        eigen_vectors[i] = A[i];
    }

    double* b = ((triangle_factor) ? triangle_factor : new double[N * N]);

    for (i = 0; i < n * n; i++) {
        b[i] = B[i];
    }



    //DSYEVD( JOBZ, UPLO, &n, eigen_vectors, &LDA, eigen_values, WORK, &LWORK, IWORK, &LIWORK, &INFO ); 
    DSYGVD(&ITYPE, &JOBZ, &UPLO, &n, a, &LDA, b, &LDB, eigen_values, WORK, &LWORK, IWORK, &LIWORK, &INFO);



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
    if (triangle_factor == NULL) {
        delete[] b;
    }

    return INFO;

}
#endif
#endif

inline
int lapack_DSYGVD(const double* A, const double* B, double* eigen_values, double* eigen_vectors, double* triangle_factor, int N){
    return lapack_DSYGVD('U', A, B, eigen_values, eigen_vectors, triangle_factor, N);
}

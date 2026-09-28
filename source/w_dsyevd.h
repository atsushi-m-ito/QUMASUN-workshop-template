#pragma once

#ifdef GY_WITH_CUDA
#include "gyield/gy_w_dsyevd_cuda.h"
#elif defined (GY_WITH_HIP)
#include "gyield/gy_w_dsyevd_hip.h"
#else  //CPU
#include <stdio.h>
#include <stdlib.h>
#include "wrap_lapack.h"


#ifndef NO_LAPACKE

inline
int lapack_DSYEVD(char UPLO, double* A, double* eigen_values, double* eigen_vectors, int N)
{

    /*
       DSYEVD

       input:  N;
       input:  A[n][n];  matrix A
       output: eigen_vectors[n][n];  eigevectors
       output: eigen_values[n];    eigenvalues
    */

    char JOBZ = ((eigen_vectors) ? 'V' : 'N');

    int n = N;
    int LDA = n;
    
    //Because lapack input matrix and output eigen vector is same pointor, A should be copied to eiven_vectors buffer//
    double* a = ((eigen_vectors) ? eigen_vectors : new double[N * N]);
    memcpy(a, A, N * N * sizeof(double));

    int info = LAPACKE_dsyevd(LAPACK_COL_MAJOR, JOBZ, UPLO, n, a, LDA, eigen_values);

    if (info > 0) {
        /* printf("\n%s: error in dsyevd_, info=%d\n\n",name,INFO); */
    } else if (info < 0) {
        printf(" info=%d\n", info);

    }

    if (eigen_vectors == NULL) {
        delete[] a;
    }

    return info;
}

#else
inline
int lapack_DSYEVD(char UPLO, double *A, double *eigen_values, double *eigen_vectors, int N)
{

  /* 
     DSYEVD
  
     input:  N;
     input:  A[n][n];  matrix A
     output: eigen_vectors[n][n];  eigevectors
     output: eigen_values[n];    eigenvalues 
  */
    




  char JOBZ = ((eigen_vectors) ? 'V' : 'N');
  
  

  INTEGER n=N;
  INTEGER LDA=n;
  double VL,VU; /* dummy */
  INTEGER IL,IU; 
  double ABSTOL=LAPACK_ABSTOL;
  INTEGER M;

  INTEGER LDZ=n;
  INTEGER LWORK,LIWORK;
  double *WORK;
  INTEGER *IWORK;
  INTEGER INFO;

  int i,j;

  //A=(double*)malloc(sizeof(double)*n*n);

  LWORK=  1 + 6*n + 2*n*n;
  WORK=(double*)malloc(sizeof(double)*LWORK);

  LIWORK = 3 + 5*n;
  IWORK=(INTEGER*)malloc(sizeof(INTEGER)*LIWORK);


  IL = 1;
  IU = N; 
 


  //Because lapack input matrix and output eigen vector is same pointor, A should be copied to eiven_vectors buffer//
	double* a = ((eigen_vectors) ? eigen_vectors : new double[N*N]);
    
	for (i=0;i<n*n;i++) {
		a[i] = A[i];
	}
	  

	DSYEVD( &JOBZ, &UPLO, &n, a, &LDA, eigen_values, WORK, &LWORK, IWORK, &LIWORK, &INFO ); 


  
  if (INFO>0) {
    /* printf("\n%s: error in dsyevd_, info=%d\n\n",name,INFO); */
  }
  else if (INFO<0) {
    printf(" info=%d\n",INFO);

  }
  else{ /* (INFO==0) */
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

  if(eigen_vectors == NULL){
	  delete [] a;
  }
  
  return INFO;
}
#endif
#endif

inline
int lapack_DSYEVD(double* A, double* eigen_values, double* eigen_vectors, int N){
    return lapack_DSYEVD('U', A, eigen_values, eigen_vectors, N);
}
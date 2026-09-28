#pragma once

#include <stdio.h>
#include <stdlib.h>
#include "wrap_lapack.h"

#ifndef NO_LAPACKE

inline
int lapack_DSYTRD_EDC(double *A, double *eigen_values, double *eigen_vectors, int N)
{

  /* 
     DSYEVD
  
     input:  N;
     input:  A[n][n];  matrix A
     output: eigen_vectors[n][n];  eigevectors
     output: eigen_values[n];    eigenvalues 
  */

  char  JOBZ = ((eigen_vectors) ? 'V' : 'N');
  
  char  UPLO='U';

  int n=N;
  int LDA=n;
  double VL,VU; /* dummy */
  double ABSTOL=LAPACK_ABSTOL;
  int M;

  int LDZ=n;


  //Because lapack input matrix and output eigen vector is same pointer, A should be copied to eiven_vectors buffer//
	double* a = new double[N*N];
	memcpy(a, A, sizeof(double) * N * N);
		
	
	double* D = new double[N];
	double* E = new double[N];
	double* tau = new double[N];
	
	//三重対角化
	int INFO = LAPACKE_dsytrd(LAPACK_COL_MAJOR, UPLO, n, a, LDA, D, E, tau); 
	
	if (INFO!=0) {
		printf(" info=%d\n",INFO);

	}
  
	double* Z = ((eigen_vectors) ? eigen_vectors : new double[N*N]);

	//三重対角行列の固有値を求める
	//DSTEVD( &JOBZ, &n, D, E, Z, &LDZ, WORK, &LWORK, IWORK, &LIWORK, &INFO ); 
	

	//三重対角行列の固有値を求める2
	char  COMPZ='I';	
	LAPACKE_dstedc(LAPACK_COL_MAJOR, COMPZ, n, D, E, Z, LDZ);

	//変換行列を掛けて元の行列に対する固有ベクトル変換
	char  LEFT='L';
	char  TRANS='N';
	LAPACKE_dormtr(LAPACK_COL_MAJOR, LEFT, UPLO, TRANS, n, n, a, LDA, tau, Z, LDZ);
	
  
  if (INFO>0) {
    /* printf("\n%s: error in dsyevd_, info=%d\n\n",name,INFO); */
  }
  else if (INFO<0) {
    printf(" info=%d\n",INFO);

  }

  if(eigen_vectors == NULL){
	  delete [] Z;
  }

  for(int i = 0; i < N; i++){
	  eigen_values[i] = D[i];
  }

  delete [] a;
  delete [] D;
  delete [] E;
  delete [] tau;
  
  return INFO;
}

#else


inline
int lapack_DSYTRD_EDC(double* A, double* eigen_values, double* eigen_vectors, int N)
{

	/*
	   DSYEVD

	   input:  N;
	   input:  A[n][n];  matrix A
	   output: eigen_vectors[n][n];  eigevectors
	   output: eigen_values[n];    eigenvalues
	*/





	char  JOBZ = ((eigen_vectors) ? 'V' : 'N');

	char* UPLO = "U";

	INTEGER n = N;
	INTEGER LDA = n;
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
	double* a = new double[N * N];
	for (i = 0; i < n * n; i++) {
		a[i] = A[i];
	}

	double* D = new double[N];
	double* E = new double[N];
	double* tau = new double[N];

	//三重対角化
	DSYTRD(UPLO, &n, a, &LDA, D, E, tau, WORK, &LWORK, &INFO);

	if (INFO != 0) {
		printf(" info=%d\n", INFO);

	}

	double* Z = ((eigen_vectors) ? eigen_vectors : new double[N * N]);

	//三重対角行列の固有値を求める
	//DSTEVD( &JOBZ, &n, D, E, Z, &LDZ, WORK, &LWORK, IWORK, &LIWORK, &INFO ); 


	//三重対角行列の固有値を求める2
	char* COMPZ = "I";
	DSTEDC(COMPZ, &n, D, E, Z, &LDZ, WORK, &LWORK, IWORK, &LIWORK, &INFO);

	//変換行列を掛けて元の行列に対する固有ベクトル変換
	char* LEFT = "L";
	char* TRANS = "N";
	DORMTR(LEFT, UPLO, TRANS, &n, &n, a, &LDA, tau, Z, &LDZ, WORK, &LWORK, &INFO);


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
		delete[] Z;
	}

	for (int i = 0; i < N; i++) {
		eigen_values[i] = D[i];
	}

	delete[] a;
	delete[] D;
	delete[] E;
	delete[] tau;

	return INFO;
}

#endif

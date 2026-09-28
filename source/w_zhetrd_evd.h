#pragma once
#include <stdio.h>
#include <stdlib.h>
#include "wrap_lapack.h"

#ifndef NO_LAPACKE

template<typename MYCOMPLEX>
int lapack_ZHETRD_EDC(MYCOMPLEX*A, double *eigen_values, MYCOMPLEX*eigen_vectors, int N)
{

  /* 
     DSYEVD
  
     input:  N;
     input:  A[n][n];  matrix A
     output: eigen_vectors[n][n];  eigevectors
     output: eigen_values[n];    eigenvalues 
  */
    




  char  JOBZ = ((eigen_vectors) ? 'V' : 'N');
  
  char UPLO='U';

  int n=N;
  int LDA=n;
  double VL,VU; /* dummy */
  int IL,IU;
  double ABSTOL=LAPACK_ABSTOL;
  int M;

  int LDZ=n;
  
  IL = 1;
  IU = N; 
 


  //Because lapack input matrix and output eigen vector is same pointor, A should be copied to eiven_vectors buffer//
	lapack_complex_double* a = new lapack_complex_double[N*N];
	memcpy(a, A, sizeof(lapack_complex_double) * N * N);
	

	double* D = new double[N];
	double* E = new double[N];
	lapack_complex_double* tau = new lapack_complex_double[N];
	
	//三重対角化
	int INFO = LAPACKE_zhetrd(LAPACK_COL_MAJOR, UPLO, n, (lapack_complex_double*)a, LDA, D, E, (lapack_complex_double*)tau );

	
	if (INFO!=0) {
		printf(" info=%d\n",INFO);

	}
  
	lapack_complex_double* z = ((eigen_vectors) ? (lapack_complex_double*)eigen_vectors : new lapack_complex_double[N*N]);
    
	//三重対角行列の固有値を求める2
	char  COMPZ='I';	
	INFO = LAPACKE_zstedc(LAPACK_COL_MAJOR, COMPZ, n, D, E, (lapack_complex_double*)z, LDZ);

	//変換行列を掛けて元の行列に対する固有ベクトル変換
	char  LEFT='L';
	char  TRANS='N';
	INFO = LAPACKE_zunmtr(LAPACK_COL_MAJOR, LEFT, UPLO, TRANS, n, n, (lapack_complex_double*)a, LDA, (lapack_complex_double*)tau, (lapack_complex_double*)z, LDZ);
	
  
  if (INFO>0) {
    /* printf("\n%s: error in dsyevd_, info=%d\n\n",name,INFO); */
  }
  else if (INFO<0) {
    printf(" info=%d\n",INFO);

  }

  if(eigen_vectors == nullptr){
	  delete [] z;
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



template<typename MYCOMPLEX>
int lapack_ZHETRD_EDC(dcomplex* A, double* eigen_values, dcomplex* eigen_vectors, int N)
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
	dcomplex* WORK;
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
	WORK = (dcomplex*)malloc(sizeof(dcomplex) * LWORK);
	//memset(WORK, 0, sizeof(dcomplex)*LWORK);		/* AITUNE */

	INTEGER LRWORK = 1 + 5 * N + 2 * N * N;
	double* RWORK = (double*)malloc(sizeof(double) * LRWORK);
	//memset(RWORK, 0, sizeof(double)*LRWORK);		/* AITUNE */

	LIWORK = 3 + 5 * N;
	IWORK = (INTEGER*)malloc(sizeof(INTEGER) * LIWORK);
	//memset(IWORK, 0, sizeof(INTEGER)*LIWORK);		/* AITUNE */

	IL = 1;
	IU = N;



	//Because lapack input matrix and output eigen vector is same pointor, A should be copied to eiven_vectors buffer//
	dcomplex* a = new dcomplex[N * N];

	for (i = 0; i < n * n; i++) {
		a[i] = A[i];
	}

	double* D = new double[N];
	double* E = new double[N];
	dcomplex* tau = new dcomplex[N];

	//三重対角化
	ZHETRD(UPLO, &n, (MKL_Complex16*)a, &LDA, D, E, (MKL_Complex16*)tau, (MKL_Complex16*)WORK, &LWORK, &INFO);


	if (INFO != 0) {
		printf(" info=%d\n", INFO);

	}

	dcomplex* z = ((eigen_vectors) ? eigen_vectors : new dcomplex[N * N]);

	//三重対角行列の固有値を求める2
	char* COMPZ = "I";
	ZSTEDC(COMPZ, &n, D, E, (MKL_Complex16*)z, &LDZ, (MKL_Complex16*)WORK, &LWORK, RWORK, &LRWORK, IWORK, &LIWORK, &INFO);

	//変換行列を掛けて元の行列に対する固有ベクトル変換
	char* LEFT = "L";
	char* TRANS = "N";
	ZUNMTR(LEFT, UPLO, TRANS, &n, &n, (MKL_Complex16*)a, &LDA, (MKL_Complex16*)tau, (MKL_Complex16*)z, &LDZ, (MKL_Complex16*)WORK, &LWORK, &INFO);


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
		delete[] z;
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

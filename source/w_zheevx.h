#pragma once
#include <stdio.h>
#include <stdlib.h>
#include "wrap_lapack.h"

#ifndef NO_LAPACKE


template<class MYCOMPLEX>
int lapack_ZHEEVX(MYCOMPLEX* A, double* eigen_values, MYCOMPLEX* eigen_vectors, MYCOMPLEX* triangle_factor, int N) {

	/*
	ZHEEVX

	input:  N;
	input:  A[n][n];  matrix A
	input:  B[n][n];  matrix B: S matrix
	output: eigen_vectors[n][n];  eigevectors
	output: eigen_values[n];    eigenvalues
	*/

	//INTEGER ITYPE = 1;	// 1 indicate "A v = e B v" type problems//
	char  JOBZ = ((eigen_vectors) ? 'V' : 'N');

	char  RANGE = 'A';
	char  UPLO = 'U';

	int n = N;
	int LDA = n;
	double VL, VU; /* dummy */
	double ABSTOL = LAPACK_ABSTOL;
	int M;

	int LDZ = n;
	int* IFAIL = (int*)malloc(sizeof(int) * N);
	
	int IL = 1;
	int IU = N;




    lapack_complex_double* Acopy = (lapack_complex_double*)malloc(sizeof(lapack_complex_double) * n * n);
    memcpy(Acopy, A, sizeof(lapack_complex_double) * n * n);

    int INFO = LAPACKE_zheevx(LAPACK_COL_MAJOR, JOBZ, RANGE, UPLO, n, Acopy, LDA,
        VL, VU, IL, IU, ABSTOL, &M,
        eigen_values, (lapack_complex_double*)eigen_vectors, LDZ, IFAIL);
    free(Acopy);


	if (INFO > 0) {
		/* printf("\n%s: error in dsyevd_, info=%d\n\n",name,INFO); */
	} else if (INFO < 0) {
		printf(" info=%d\n", INFO);

	} 
	free(IFAIL);


	return INFO;
}

#else

template<class MYCOMPLEX>
int lapack_ZHEEVX(MYCOMPLEX*A, double *eigen_values, MYCOMPLEX*eigen_vectors, MYCOMPLEX* triangle_factor, int N) {

	/*
	ZHEEVX

	input:  N;
	input:  A[n][n];  matrix A
	input:  B[n][n];  matrix B: S matrix
	output: eigen_vectors[n][n];  eigevectors
	output: eigen_values[n];    eigenvalues
	*/





	//INTEGER ITYPE = 1;	// 1 indicate "A v = e B v" type problems//
	char  JOBZ = ((eigen_vectors) ? 'V' : 'N');

	char  RANGE = 'A';
	char  UPLO = 'U';

	INTEGER n = N;
	INTEGER LDA = n;
	double VL, VU; /* dummy */
	INTEGER IL, IU;
	double ABSTOL = LAPACK_ABSTOL;
	INTEGER M;

	INTEGER LDZ = n;
	INTEGER LIWORK;
	INTEGER *IWORK;
	INTEGER *IFAIL = (INTEGER*)malloc(sizeof(INTEGER)*N);
	INTEGER INFO;

	//int i, j;

	INTEGER LWORK = 4 * N; // > 2*N //
	lapack_complex_double* WORK = (lapack_complex_double*)malloc(sizeof(lapack_complex_double)*LWORK);


	INTEGER LRWORK = 7 * N;
	double* RWORK = (double*)malloc(sizeof(double)*LRWORK);
	//memset(RWORK, 0, sizeof(double)*LRWORK);		/* AITUNE */

	LIWORK = 5 * N;
	IWORK = (INTEGER*)malloc(sizeof(INTEGER)*LIWORK);
	//memset(IWORK, 0, sizeof(INTEGER)*LIWORK);		/* AITUNE */

	IL = 1;
	IU = N;




	ZHEEVX(&JOBZ, &RANGE, &UPLO, &n, (MKL_Complex16*)A, &LDA,  
		&VL, &VU, &IL, &IU, &ABSTOL, &M, 
		eigen_values, (MKL_Complex16*)eigen_vectors, &LDZ, (MKL_Complex16*)WORK, &LWORK, RWORK, IWORK, IFAIL, &INFO);



	if (INFO>0) {
		/* printf("\n%s: error in dsyevd_, info=%d\n\n",name,INFO); */
	} else if (INFO<0) {
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
	free(IFAIL);
	free(IWORK); free(RWORK); free(WORK);


	return INFO;
}
#endif

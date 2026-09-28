#pragma once
#include <stdio.h>
#include <stdlib.h>
#include "wrap_lapack.h"

#ifndef NO_LAPACKE

inline
int lapack_DSYEVX(double *A, double *eigen_values, double*eigen_vectors, int N) {

	/*
	DSYEVX

	input:  N;
	input:  A[n][n];  matrix A
	output: eigen_vectors[n][n];  eigevectors
	output: eigen_values[n];    eigenvalues
	*/

	int ITYPE = 1;	// 1 indicate "A v = e B v" type problems//
	char  JOBZ = ((eigen_vectors) ? 'V' : 'N');

	char  RANGE = 'A';
	char  UPLO = 'U';

	int n = N;
	int LDA = n;
	int LDB = n;
	double VL=0.0, VU = 0.0; /* dummy */
	
	double ABSTOL = LAPACK_ABSTOL;
	int M;

	int LDZ = n;
	int *IFAIL = (int*)malloc(sizeof(int)*N);
	
	int IL = 1;
	int IU = N;

    double* Acopy = (double*)malloc(sizeof(double) * n * n);
    memcpy(Acopy, A, sizeof(double) * n * n);
	int INFO = LAPACKE_dsyevx(LAPACK_COL_MAJOR, JOBZ, RANGE, UPLO, n, Acopy, LDA, VL, VU, IL, IU, ABSTOL, &M,
		eigen_values, eigen_vectors, LDZ, IFAIL);
    free(Acopy);


	if (INFO>0) {
		/* printf("\n%s: error in dsyevd_, info=%d\n\n",name,INFO); */
	} else if (INFO<0) {
		printf(" info=%d\n", INFO);

	}
	free(IFAIL);

	return INFO;
}

#else
#ifdef LAPACK_NO_HEADER
extern "C" {
#ifdef _NEC
#define DSYEVX dsyevx_
#endif
	void DSYEVX(const char* jobz, const char* range, const char* uplo,
		const int* n, double a[], const int* lda,
		const double* vl, const double* vu, const int* il, const int* iu,
		const double* abstol, int* m, double w[], double z[], const int* ldz,
		double work[], const int* lwork, int iwork[], int jfail[], int* info);
	/*
	DSYEVX(&JOBZ, &RANGE, &UPLO, &n, Acopy, &LDA, &VL, &VU, &IL, &IU, &ABSTOL, &M,
		eigen_values, eigen_vectors, &LDZ, WORK, &LWORK, IWORK, IFAIL, &INFO);
		*/
}
#endif

inline
int lapack_DSYEVX(double* A, double* eigen_values, double* eigen_vectors, int N) {

	/*
	DSYEVX

	input:  N;
	input:  A[n][n];  matrix A
	output: eigen_vectors[n][n];  eigevectors
	output: eigen_values[n];    eigenvalues
	*/





	INTEGER ITYPE = 1;	// 1 indicate "A v = e B v" type problems//
	char  JOBZ = ((eigen_vectors) ? 'V' : 'N');

	char  RANGE = 'A';
	char  UPLO = 'U';

	INTEGER n = N;
	INTEGER LDA = n;
	INTEGER LDB = n;
	double VL, VU; /* dummy */
	INTEGER IL, IU;
	double ABSTOL = LAPACK_ABSTOL;
	INTEGER M;

	INTEGER LDZ = n;
	INTEGER LIWORK;
	INTEGER* IWORK;
	INTEGER* IFAIL = (INTEGER*)malloc(sizeof(INTEGER) * N);
	INTEGER INFO;

	//int i, j;

	INTEGER LWORK = 8 * N; // > 8*N //
	double* WORK = (double*)malloc(sizeof(double) * LWORK);


	//INTEGER LRWORK = 7 * N;
	//double* RWORK = (double*)malloc(sizeof(double)*LRWORK);
	//memset(RWORK, 0, sizeof(double)*LRWORK);		/* AITUNE */

	LIWORK = 5 * N;
	IWORK = (INTEGER*)malloc(sizeof(INTEGER) * LIWORK);
	//memset(IWORK, 0, sizeof(INTEGER)*LIWORK);		/* AITUNE */

	IL = 1;
	IU = N;

	//copy A because LAPACK destroyies input matrix//
	double* Acopy = (double*)malloc(sizeof(double) * n * n);
	memcpy(Acopy, A, sizeof(double) * n * n);

	DSYEVX(&JOBZ, &RANGE, &UPLO, &n, Acopy, &LDA, &VL, &VU, &IL, &IU, &ABSTOL, &M,
		eigen_values, eigen_vectors, &LDZ, WORK, &LWORK, IWORK, IFAIL, &INFO);

	free(Acopy);

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
	free(IFAIL);
	free(IWORK);; free(WORK);

	return INFO;
}

#endif

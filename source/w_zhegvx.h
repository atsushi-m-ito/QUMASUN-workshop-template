#pragma once

#ifdef GY_WITH_CUDA
#include "gyield/gy_w_zhegvd_cuda.h"
#elif defined (GY_WITH_HIP)
#include "gyield/gy_w_zhegvd_hip.h"
#else  //CPU

#include <stdio.h>
#include <stdlib.h>
#include "wrap_lapack.h"

#ifndef NO_LAPACKE

template<class MYCOMPLEX>
int lapack_ZHEGVX(char UPLO, MYCOMPLEX*A, MYCOMPLEX*B, double *eigen_values, MYCOMPLEX*eigen_vectors, int N, int ilower, int iupper) {

	/*
	ZHEGVX

	input:  N;
	input:  A[n][n];  matrix A
	input:  B[n][n];  matrix B: S matrix
	output: eigen_vectors[n][n];  eigevectors
	output: eigen_values[n];    eigenvalues
	*/



	int ITYPE = 1;	// 1 indicate "A v = e B v" type problems//
	char  JOBZ = ((eigen_vectors) ? 'V' : 'N');

	int IL = ilower;
	int IU = iupper;
    char  RANGE = ((IL == 1) && (IU == N)) ? 'A' : 'I'; //all eigen values and vectors//
	

	int n = N;
	int LDA = n;
	int LDB = n;
	double VL=0.0, VU=0.0; /* dummy */
	double ABSTOL = LAPACK_ABSTOL;
	int M;

	int LDZ = n;
	int *IFAIL = (int*)malloc(sizeof(int)*N);
	

	

    lapack_complex_double* Acopy = (lapack_complex_double*)malloc(sizeof(lapack_complex_double) * n * n);
    memcpy(Acopy, A, sizeof(lapack_complex_double) * n * n);
	int INFO = LAPACKE_zhegvx(LAPACK_COL_MAJOR, ITYPE, JOBZ, RANGE, UPLO, n, Acopy, LDA, (lapack_complex_double*)B, LDB,
		VL, VU, IL, IU, ABSTOL, &M, eigen_values, (lapack_complex_double*)eigen_vectors, LDZ, IFAIL);
    free(Acopy);



	if (INFO>0) {
		
	} else if (INFO<0) {
		printf(" info=%d\n", INFO);

	} 
	free(IFAIL);
	
	return INFO;
}



#else


#ifdef LAPACK_NO_HEADER
#ifndef lapack_complex_double
#define lapack_complex_double  std::complex<double>
#endif
extern "C" {
#ifdef _NEC
#define ZHEGVX  zhegvx_
#endif
	void ZHEGVX(const int* itype, const char* jobz, const char* range,
		const char* uplo, const int* n, lapack_complex_double* a,
		const int* lda, lapack_complex_double* b, const int* ldb,
		const double* vl, const double* vu, const int* il,
		const int* iu, const double* abstol, int* m, double* w,
		lapack_complex_double* z, const int* ldz, lapack_complex_double* work,
		const int* lwork, double* rwork, int* iwork,
		int* ifail, int* info);
}
#endif

template<class MYCOMPLEX>
inline
int lapack_ZHEGVX(MYCOMPLEX* A, MYCOMPLEX* B, double* eigen_values, MYCOMPLEX* eigen_vectors, int N, int ilower, int iupper, int ld_all) {

	/*
	ZHEGVX

	input:  N;
	input:  A[n][n];  matrix A
	input:  B[n][n];  matrix B: S matrix
	output: eigen_vectors[n][n];  eigevectors
	output: eigen_values[n];    eigenvalues
	*/





	INTEGER ITYPE = 1;	// 1 indicate "A v = e B v" type problems//
	char  JOBZ = ((eigen_vectors) ? 'V' : 'N');

	INTEGER IL = ilower;
	INTEGER IU = iupper;
	char  RANGE = ((IL == 1) && (IU == N)) ? 'A' : 'I'; //all eigen values and vectors//
	char  UPLO = 'U';

	INTEGER n = N;
	INTEGER LDA = ld_all;
	INTEGER LDB = ld_all;
	double VL, VU; /* dummy */
	double ABSTOL = LAPACK_ABSTOL;
	INTEGER M;

	INTEGER LDZ = ld_all;
	INTEGER LIWORK;
	INTEGER* IWORK;
	INTEGER* IFAIL = (INTEGER*)malloc(sizeof(INTEGER) * N);
	INTEGER INFO;

	//int i, j;

	INTEGER LWORK = 4 * N; // > 2*N //
	MYCOMPLEX* WORK = (MYCOMPLEX*)malloc(sizeof(MYCOMPLEX) * LWORK);


	INTEGER LRWORK = 7 * N;
	double* RWORK = (double*)malloc(sizeof(double) * LRWORK);
	//memset(RWORK, 0, sizeof(double)*LRWORK);		/* AITUNE */

	LIWORK = 5 * N;
	IWORK = (INTEGER*)malloc(sizeof(INTEGER) * LIWORK);
	//memset(IWORK, 0, sizeof(INTEGER)*LIWORK);		/* AITUNE */

	IL = 1;
	IU = N;



	ZHEGVX(&ITYPE, &JOBZ, &RANGE, &UPLO, &n, (lapack_complex_double*)A, &LDA, (lapack_complex_double*)B, &LDB,
		&VL, &VU, &IL, &IU, &ABSTOL, &M,
		eigen_values, (lapack_complex_double*)eigen_vectors, &LDZ, (lapack_complex_double*)WORK, &LWORK, RWORK, IWORK, IFAIL, &INFO);



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
	free(IWORK); free(RWORK); free(WORK);

	return INFO;
}




#endif
#endif

template<class MYCOMPLEX>
int lapack_ZHEGVX(MYCOMPLEX* A, MYCOMPLEX* B, double* eigen_values, MYCOMPLEX* eigen_vectors, int N, int ilower, int iupper) {
	return lapack_ZHEGVX('U', A, B, eigen_values, eigen_vectors, N, ilower, iupper);
}
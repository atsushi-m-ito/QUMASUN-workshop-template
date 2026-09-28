#pragma once

#include <stdio.h>
#include <stdlib.h>
#include "transpose.h"
#include "soacomplex.h"

#if defined( GY_WITH_CUDA)|| defined (GY_WITH_HIP)
#include "gyield/gy_w_dgemm.h"

#else  //CPU

#include "wrap_lapack.h"

#ifdef LAPACK_NO_HEADER
#ifndef lapack_complex_double
#define lapack_complex_double  std::complex<double>
#endif
extern "C" {
#ifdef _NEC
//#define dgemm
#endif
void dgemm_(const char* transa, const char* transb, const int* m, const int* n, const int* k,
		const double* alpha, const double* a, const int* lda, const double* b, const int* ldb,
		const double* beta, double* c, const int* ldc);
}
#endif




/*
* DGEMM wrapper
* C = alpha*(A^t*B) + beta*C
* matrix A is KxM grids (A[iy + K * ix] where ix in [0,M), iy in [0,K)
* matrix B is KxN grids (A[iy + K * ix] where ix in [0,N), iy in [0,K)
*/
inline
void blas_DGEMM_t(int M, int N, int K, const double* A, const double* B, double* C, int strideC, double alpha, double beta) {
	cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, M, N, K, alpha, A, K, B, K, beta, C, strideC);
	//cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, M, B, K, beta, C, strideC);
	//cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, alpha, A, K, B, K, beta, C, strideC);
	
}
#if 0
inline
void blas_DGEMM_t_ld(int M, int N, int K, double* A, int lda, double* B, int ldb, double* C, int ldc, double alpha, double beta) {
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    //cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, M, B, K, beta, C, strideC);
    //cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, alpha, A, K, B, K, beta, C, strideC);

}
#endif

/*
* DSYRK wrapper
* C = alpha*(A^t*A) + beta*C
* matrix A is KxM grids (A[iy + K * ix] where ix in [0,M), iy in [0,K)
*/
inline
void blas_DSYRK_t(int N, int K, const double* A, double* C, int strideC, double alpha, double beta) {
    cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans, N, K, alpha, A, K, beta, C, strideC);

}

#ifdef USE_DGEMMT
inline
void blas_DGEMMT_t_up(int N, int K, const double* A, const double* B, double* C, int strideC, double alpha, double beta) {
	cblas_dgemmt(CblasColMajor, CblasUpper, CblasTrans, CblasNoTrans, N, K, alpha, A, K, B, K, beta, C, strideC);
	//cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, M, B, K, beta, C, strideC);
	//cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, alpha, A, K, B, K, beta, C, strideC);

}
#endif
#if 0
inline
void blas_DGEMM_t2(int M, int N, int K, double* A, double* B, double* C, int strideC, double alpha, double beta) {
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, alpha, A, K, B, K, beta, C, strideC);
	
}
#endif
inline
void blas_DGEMM_bt(int M, int N, int K, const double* A, const double* B, double* C, int strideC, double alpha, double beta) {
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, M, N, K, alpha, A, K, B, K, beta, C, strideC);
    

}



/*
* DGEMM wrapper
* C = alpha*(A*B) + beta*C
* matrix A is KxM grids (A[iy + K * ix] where ix in [0,M), iy in [0,K)
* matrix B is KxN grids (A[iy + K * ix] where ix in [0,N), iy in [0,K)
*/
inline
void blas_DGEMM_n(int M, int N, int K, const double* A, const double* B, double* C, int strideC, double alpha, double beta) {
	//cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N, beta, C, strideC);
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, M, B, K, beta, C, strideC);
}


inline
void blas_DGEMM(char TransA, char TransB, int M, int N, int K, double alpha, const double* A, int lda, const double* B, int ldb, double beta, double* C, int ldc) {
    
    cblas_dgemm(CblasColMajor, 
        TransA == 'T' ? CblasTrans : CblasNoTrans,
        TransB == 'T' ? CblasTrans : CblasNoTrans, 
        M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}


inline
void blas_ZGEMM(char TransA, char TransB, int M, int N, int K, const void* alpha, const void* A, int lda, 
    const void* B, int ldb, const void* beta, void* C, int ldc) {
    cblas_zgemm(CblasColMajor, 
        TransA == 'C' ? CblasConjTrans : CblasNoTrans,
        TransB == 'C' ? CblasConjTrans : CblasNoTrans,
        M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}


#ifdef _NEC
inline
void nec_DMM_add(int N, int M, int K, double* A, double* B, double* C, int strideC, double alpha) {
#pragma _NEC ivdep
	for (int m = 0; m < M; ++m) {
#pragma _NEC ivdep
		for (int k = 0; k < K; ++k) {
#pragma _NEC ivdep
			for (int n = 0; n < N; ++n) {
				C[n + strideC * m] += alpha * A[k + K * n] * B[k + K * m];
			}
		}
	}
}
#endif

/*
* DGEMV wrapper
* C = alpha*(A^t*V) + beta*C
* matrix A is KxM grids (A[iy + K * ix] where ix in [0,M), iy in [0,K)
* matrix V is K grids (A[iy] where iy in [0,K)
*/
inline
void blas_DGEMV_t(int M, int K, const double* A, const double* V, double* C, double alpha, double beta) {

	cblas_dgemv(CblasRowMajor, CblasNoTrans, M, K, alpha, A, K, V, 1, beta, C, 1);

}
#if 0
inline
void blas_ZGEMV_t(int M, int K, void* A, void* V, void* C, OneComplex alpha, OneComplex beta) {

    cblas_zgemv(CblasRowMajor, CblasNoTrans, M, K, &alpha, A, K, V, 1, &beta, C, 1);

}
#endif
#endif

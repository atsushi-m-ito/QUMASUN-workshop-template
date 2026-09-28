#pragma once

#include "w_dpotrf.h"
#include "w_dtrtrs.h"
#include "w_zheevd.h"


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
int lapack_ZHEGVD(char UPLO, const MYCOMPLEX* A, const MYCOMPLEX* B, double* eigen_values, MYCOMPLEX* eigen_vectors, int N) {

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



    int n = N;
    int LDA = n;
    int LDB = n;
    

    memcpy(eigen_vectors, A, sizeof(MYCOMPLEX) * N * N);
    MYCOMPLEX* mirrorB = new MYCOMPLEX[N * N];
    memcpy(mirrorB, B, sizeof(MYCOMPLEX) * N * N);

    int INFO =LAPACKE_zhegvd(LAPACK_COL_MAJOR, ITYPE, JOBZ, UPLO, n, (lapack_complex_double*)eigen_vectors, LDA, (lapack_complex_double*)mirrorB, LDB, eigen_values);

    delete[]mirrorB;

    if (INFO > 0) {
        /* printf("\n%s: error in dsyevd_, info=%d\n\n",name,INFO); */
    } else if (INFO < 0) {
        printf(" info=%d\n", INFO);

    } 
    

    return INFO;
}

#else


#ifdef LAPACK_NO_HEADER
#ifndef lapack_complex_double
#define lapack_complex_double  std::complex<double>
#endif
extern "C" {
#ifdef _NEC
#define ZHEGVD  zhegvd_
#endif
	void ZHEGVD(const int* itype, const char* jobz,
		const char* uplo, const int* n, lapack_complex_double* a,
		const int* lda, lapack_complex_double* b, const int* ldb,
		double* w,
		//lapack_complex_double* z, const int* ldz, 
		lapack_complex_double* work,
		const int* lwork, double* rwork, const int* lrwork, int* iwork, const int* liwork,
		int* info);
}
#endif

template<class MYCOMPLEX>
inline
int lapack_ZHEGVD(char UPLO, MYCOMPLEX* A, MYCOMPLEX* B, double* eigen_values, MYCOMPLEX* eigen_vectors, int N) {

	/*
	ZHEGVX

	input:  N;
	input:  A[n][n];  matrix A
	input:  B[n][n];  matrix B: S matrix
	output: eigen_vectors[n][n];  eigevectors
	output: eigen_values[n];    eigenvalues
	*/




	printf("test----------------------------------------------------------------------\n");
	INTEGER ITYPE = 1;	// 1 indicate "A v = e B v" type problems//
	char  JOBZ = ((eigen_vectors) ? 'V' : 'N');


	INTEGER n = N;
	INTEGER LDA = n;
	INTEGER LDB = n;
	double ABSTOL = LAPACK_ABSTOL;

	INTEGER LDZ = n;


	INTEGER INFO;

	//int i, j;

	INTEGER LWORK = 2 * N + N * N;
	MYCOMPLEX* WORK = (MYCOMPLEX*)malloc(sizeof(MYCOMPLEX) * LWORK);


	INTEGER LRWORK = 1 + 5 * N + 2 * N * N;
	double* RWORK = (double*)malloc(sizeof(double) * LRWORK);


	INTEGER LIWORK = 3 + 5 * N;
	INTEGER* IWORK = (INTEGER*)malloc(sizeof(INTEGER) * LIWORK);
	//memset(IWORK, 0, sizeof(INTEGER)*LIWORK);		/* AITUNE */


	memcpy(eigen_vectors, A, sizeof(MYCOMPLEX) * N * N);
	MYCOMPLEX* mirrorB = new MYCOMPLEX[N * N];
	memcpy(mirrorB, B, sizeof(MYCOMPLEX) * N * N);

	ZHEGVD(&ITYPE, &JOBZ, &UPLO, &n, (lapack_complex_double*)eigen_vectors, &LDA, (lapack_complex_double*)mirrorB, &LDB,
		eigen_values, (lapack_complex_double*)WORK, &LWORK, RWORK, &LRWORK, IWORK, &LIWORK, &INFO);
	//(lapack_complex_double*)eigen_vectors, &LDZ, 


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
	free(IWORK); free(RWORK); free(WORK);

	return INFO;
}
#endif
#endif


template<class MYCOMPLEX>
int lapack_ZHEGVD(const MYCOMPLEX* A, const MYCOMPLEX* B, double* eigen_values, MYCOMPLEX* eigen_vectors, int N) {
	return lapack_ZHEGVD('U', A, B, eigen_values, eigen_vectors, N);
}

/*
    * 一般化固有値問題を解くZHEGVDの代わりにCholesky分解とEVDを使う
	* 
	* 複素の場合はエルミートかつ転置記号tはエルミート共役の意味
	*
	* Bが正定値行列(波動関数のS-matrixはグラム行列であり正定値行列)を前提に、
	* BをCholesky分解して、(tはエルミート共役)
	* B = L L^t or U^t U
	* となるL or Uを求める。Lは下三角行列, Uは上三角行列。
	* L, Uを求めるのはDPOTRFをBに対して使う。
	*
	* これを利用して、目的の一般化固有値問題を書き換えていく
	* A z = lambda B z
	* A (L^t)^{-1} L^t z = lambda L L^t z
	* 両辺に左からL^{-1}を作用して
	* L^{-1} A (L^t)^{-1} L^t z = lambda L^t z
	* よって、
	* C = L^{-1} A (L^t)^{-1}
	* z' = L^t z
	* とすれば
	* C z' = lambda z'
	* という対称行列Xの固有値問題に帰着する.
	*
	* これを解くためにDSYEVDをCに対して用いる。
	* またCを求めるために、
	* L X = A
	* となる解をDTRTRS(DTRSM)によって求めると、(BLASのDTRSMのLAPACKラッパーがDTRTRS)
	* X = L^{-1} A
	* を得る。
	* さらにBLASのDTRSMを使って、エルミート共役作業も任せてしまって
	* C L^t = X
	* の解Cを求めれば、
	* C = X (L^t)^{-1} = L^{-1} A (L^t)^{-1}
	* を得られる。
	*
	*
	* B = U^t U を用いる場合は次のようにする
	* A z = lambda B z
	* => A U^{-1} U z = lambda U^t U z
	* 両辺に左から (U^t)^{-1} を作用して
	* (U^t)^{-1} A U^{-1} U z = lambda U z
	* => C z' = lambda z'
	* where
	* C = (U^t)^{-1} A U^{-1}
	* z' = U z
	* これを解くには
	* U^t X = A
	* => X = (U^t)^{-1} A
	* C U = X
	* => C = X U^{-1} = (U^t)^{-1} A U^{-1}
	*
	*
	* 最後に得られたz'に(L^t)^{-1}を作用して固有ベクトルzを得る
    * 
    * CUDAではZHEGVD_emuは遅いことをA30で確認
	*/
template<class MYCOMPLEX>
int lapack_ZHEGVD_emu(char UPLO, const MYCOMPLEX* A, const MYCOMPLEX* B, double* eigen_values, MYCOMPLEX* eigen_vectors, MYCOMPLEX* work_nn2, int N) {

	int info = 0;

	if (UPLO == 'L') {
		auto* L = work_nn2;
		auto* C = eigen_vectors;
		auto* Z = C;

		// B = L L^t
		info = lapack_ZPOTRF('L', N, B, L);
		
		// C = L^ { -1 } A(L ^ t)^ { -1 }
		blas_ZTRSM('L', 'L', 'N', 'N', N, L, A, C);
		blas_ZTRSM('R', 'L', 'C', 'N', N, L, C, C);
		
		// C z' = ei z'
		info = lapack_ZHEEVD('L', C, eigen_values, Z, N);
		
		// Z = (L^t)^{-1} Z' の計算
		blas_ZTRSM('L', 'L', 'C', 'N', N, L, Z, Z);
		
	} else { //UPLO=='U'
		auto* U = work_nn2;
        auto* C = eigen_vectors;
        auto* Z = C;

		
		// B = U^t U
		info = lapack_ZPOTRF('U', N, B, U);
		
		// C = (U^t)^{ -1 } A U^{ -1 }
		blas_ZTRSM('L', 'U', 'C', 'N', N, U, A, C);
		blas_ZTRSM('R', 'U', 'N', 'N', N, U, C, C);
		
		// C z' = ei z'
		info = lapack_ZHEEVD('U', C, eigen_values, Z, N);
		
		// Z = (U)^{-1} Z' の計算
		blas_ZTRSM('L', 'U', 'N', 'N', N, U, Z, Z);
		
	}

	return info;
}

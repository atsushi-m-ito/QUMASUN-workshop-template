#ifdef USE_MPI
#pragma once
#include <mpi.h>
#include "mpi_helper.h"
#include "w_dgemm.h"

#include "w_zhegvd.h"
#include "GridRange.h"
#include "GridScatterGather.h"

#include "vecmath.h"
#include "inverse_m.h"
#include "print_matrix.h"
#include "soacomplex.h"
#include "zgemm_mpi1d.h"
#include "innerprod_hermite.h"
#include "conjugate.h"
#include "matrix_2x2real_to_complex.h"
#include <memory>
#include "work_arena.h"
#include "StopWatch.h"

#include "gyield/gy_for.h"
#include "gyield/gy_blas_mp.h"


#ifdef USE_SCALAPACK
#include "w_pzhegvx.h"
#define USE_PZHEGVD_IMPL2
#ifdef USE_PZHEGVD_IMPL2
#include "w_pzhegvd_impl2.h"
#else
#include "w_pzhegvd_impl.h"
#endif
#endif

//LOBPCG法で複数の固有ベクトルを求める//
//複素数version//

//対角化を手動でせずに、小行列の一般化固有値解法に任せた方が良い//
//なぜなら、(1)matrix Aの演算が1回/iterationで済む。//

//dVolを乗じずに内積を取って1になるように関数の入り口でxをscaleする//
//出口でxを逆scaleして大きさを戻す//
//pはスケール後のものが関数から出力される//



//#ifdef _NEC
//#define SELF_TRANSPOSE
//#endif

//#define DEBUG_PRINT_MATRIX
//#define LOBPCG_DEBUG_PRINT

//XR, RP of S-matrix is directly calculated from Cx,Cr,Cp. But accuracy is low., and then convergence has problem.
//ただし、低エネルギー側のベクトルを固定するLOBPCG_FIX_LOWERとは現状で併用負荷(Rの直交化ができないため)
#define DIRECT_S_MATRIX_XR_RP

//X^t X, P^t P, R^t Rの計算にDSYRKを使う//
#if !defined(GY_WITH_CUDA) && !defined(GY_WITH_HIP)
#define USE_DSYRK 
#endif

//S行列を次のステップでも使いまわす場合//
//安定しないのでしばらくpending//
// 20250505時点でバグがあり動作しない//
//#define S_MATRIX_RELOAD


//#define EVERY_NORMALIZE_P
// 
//数値誤差が発生時にrollbackとescapeをする実装(must)
//20241105:default化
//#define CHECK_NEXT_VECTOR


//#define LOBPCG_Z_PRINT_EIGEN
//#define LOBPCG_Z_PRINT_NORM

//ZGEMM3(HS2HS)の行列行列積をmpiの並列法. どちらかを選ぶ//
//#define ZGEMM_3_MPI1D
#define ZGEMM_BLASMP    2

//各SCFにおいてLOBPCGの最初のステップではベクトルpを引き継がない場合//
//実際はSCF収束が全く進まない//
//#define VECTOR_P_NO_CONTINUE

//S-matrix, H-matrixの計算において同じもの同士の行列積の計算量を3/4に
// 安定だが高速化の効果は殆ど5パーセント程度//
#define XtX_3_4

//inline constexpr int lobpcg_fix_residual_threshold_id = 2;

//#define DEBUGPRINT3
//#define SUB_WATCH_ON   //for debug//

namespace LOBPCG{


    inline constexpr double LIMIT_Z_NORM_X_FOR_RESET_MATRIX = 1.0e-3;
    inline constexpr double LIMIT_Z_NORM_R_FOR_ESCAPE = 1.0e-10;

    /*
    * H-matrix and S-matrix are calculated
    * 
    * H-matrix is defined by
    * (x,r,p)^t \hat{H} (x,r,p)
    * 
    * S-matrix is defined by
    * (x,r,p)^t (x,r,p)
    */
inline
int MakeMatrix_z(int local_size, int num_solution, OneComplex* Sa, OneComplex* Sb,
	const SoAComplex* x, const SoAComplex* r, const SoAComplex* p,
	const SoAComplex* Ax, const SoAComplex* Ar, const SoAComplex* Ap) {


	const int n3 = num_solution * 3;

	for (int j = 0; j < num_solution; ++j) {
		for (int k = 0; k < num_solution; ++k) {

			Sa[j * n3 + k] = SoAC::InnerProd(x[k], Ax[j], local_size);
			Sa[j * n3 + k + num_solution] = SoAC::InnerProd(r[k], Ax[j], local_size);
			Sa[j * n3 + k + 2 * num_solution] = SoAC::InnerProd(p[k], Ax[j], local_size);

			Sa[(j + num_solution) * n3 + k] = SoAC::InnerProd(x[k], Ar[j], local_size);
			Sa[(j + num_solution) * n3 + k + num_solution] = SoAC::InnerProd(r[k], Ar[j], local_size);
			Sa[(j + num_solution) * n3 + k + 2 * num_solution] = SoAC::InnerProd(p[k], Ar[j], local_size);

			Sa[(j + 2 * num_solution) * n3 + k] = SoAC::InnerProd(x[k], Ap[j], local_size);
			Sa[(j + 2 * num_solution) * n3 + k + num_solution] = SoAC::InnerProd(r[k], Ap[j], local_size);
			Sa[(j + 2 * num_solution) * n3 + k + 2 * num_solution] = SoAC::InnerProd(p[k], Ap[j], local_size);


			Sb[j * n3 + k] = SoAC::InnerProd(x[k], x[j], local_size);
			Sb[j * n3 + k + num_solution] = SoAC::InnerProd(r[k], x[j], local_size);
			Sb[j * n3 + k + 2 * num_solution] = SoAC::InnerProd(p[k], x[j], local_size);

			Sb[(j + num_solution) * n3 + k] = SoAC::InnerProd(x[k], r[j], local_size);
			Sb[(j + num_solution) * n3 + k + num_solution] = SoAC::InnerProd(r[k], r[j], local_size);
			Sb[(j + num_solution) * n3 + k + 2 * num_solution] = SoAC::InnerProd(p[k], r[j], local_size);

			Sb[(j + 2 * num_solution) * n3 + k] = SoAC::InnerProd(x[k], p[j], local_size);
			Sb[(j + 2 * num_solution) * n3 + k + num_solution] = SoAC::InnerProd(r[k], p[j], local_size);
			Sb[(j + 2 * num_solution) * n3 + k + 2 * num_solution] = SoAC::InnerProd(p[k], p[j], local_size);


		}
	}

	int num_gemm_call = 18;
	return num_gemm_call*2;
}


#ifdef SELF_TRANSPOSE
    /*
    * H-matrix and S-matrix are calculated
    *
    * H-matrix is defined by
    * (x,r,p)^t \hat{H} (x,r,p)
    *
    * S-matrix is defined by
    * (x,r,p)^t (x,r,p)
    */
inline
int MakeMatrix_z_blas(int local_size, int num_solution, OneComplex* Sa, OneComplex* Sb, double* temp_mat_d_2n2n,
	const SoAComplex* x, const SoAComplex* r, const SoAComplex* p,
	const SoAComplex* Ax, const SoAComplex* Ar, const SoAComplex* Ap,
	SoAComplex* tmpM,
	bool is_skip_x_p, bool is_skip_Ax_Ap, bool is_skip_xr_rp) {

	const int N = num_solution;
	const int n2 = num_solution * 2;
	const int n3 = num_solution * 3;
	//const int n6 = n3 * 2;
    const int n6 = n2;

	auto AoSComplexFrom2by2Real = [](int N, OneComplex* Sa, int lda, double* temp_mat_d_2n2n, int ldt) {
#pragma ivdep
		for (int m = 0; m < N; ++m) {
			for (int n = 0; n < N; ++n) {
				Sa[n + lda * m].r = temp_mat_d_2n2n[(n * 2) + ldt * (m * 2)] + temp_mat_d_2n2n[(n * 2 + 1) + ldt * (m * 2 + 1)];
				Sa[n + lda * m].i = -temp_mat_d_2n2n[(n * 2 + 1) + ldt * (m * 2)] + temp_mat_d_2n2n[(n * 2) + ldt * (m * 2 + 1)];
			}
		}
		};


	double* X_t = tmpM[0].re;
	transpose(local_size, n2, x[0].re, local_size, X_t, n2);


	if (!is_skip_Ax_Ap) {
		blas_DGEMM_n(n2, n2, local_size, X_t, Ax[0].re, temp_mat_d_2n2n, n6, 1.0, 0.0);
        gy::Synchronize();
		AoSComplexFrom2by2Real(num_solution, Sa, n3, temp_mat_d_2n2n, n6);
	}
	blas_DGEMM_n(n2, n2, local_size, X_t, Ar[0].re, temp_mat_d_2n2n, n6, 1.0, 0.0);
    gy::Synchronize();
	AoSComplexFrom2by2Real(num_solution, Sa + N * n3, n3, temp_mat_d_2n2n, n6);
	if (!is_skip_Ax_Ap) {
		blas_DGEMM_n(n2, n2, local_size, X_t, Ap[0].re, temp_mat_d_2n2n, n6, 1.0, 0.0);
        gy::Synchronize();
		AoSComplexFrom2by2Real(num_solution, Sa + N * (2 * n3), n3, temp_mat_d_2n2n, n6);
	}

	if (!is_skip_x_p) {
		blas_DGEMM_n(n2, n2, local_size, X_t, x[0].re, temp_mat_d_2n2n, n6, 1.0, 0.0);
        gy::Synchronize();
		AoSComplexFrom2by2Real(num_solution, Sb, n3, temp_mat_d_2n2n, n6);
	}
#ifdef DIRECT_S_MATRIX_XR_RP
	if (!is_skip_xr_rp)
#endif
	{
		blas_DGEMM_n(n2, n2, local_size, X_t, r[0].re, temp_mat_d_2n2n, n6, 1.0, 0.0);
        gy::Synchronize();
		AoSComplexFrom2by2Real(num_solution, Sb + N * n3, n3, temp_mat_d_2n2n, n6);
	}
	if (!is_skip_x_p) {
		blas_DGEMM_n(n2, n2, local_size, X_t, p[0].re, temp_mat_d_2n2n, n6, 1.0, 0.0);
        gy::Synchronize();
		AoSComplexFrom2by2Real(num_solution, Sb + N * (2 * n3), n3, temp_mat_d_2n2n, n6);
	}


	double* R_t = tmpM[0].re;
	transpose(local_size, n2, r[0].re, local_size, R_t, n2);

	blas_DGEMM_n(n2, n2, local_size, R_t, Ar[0].re, temp_mat_d_2n2n, n6, 1.0, 0.0);
    gy::Synchronize();
	AoSComplexFrom2by2Real(num_solution, Sa + N * (n3 + 1), n3, temp_mat_d_2n2n, n6);

	blas_DGEMM_n(n2, n2, local_size, R_t, Ap[0].re, temp_mat_d_2n2n, n6, 1.0, 0.0);
    gy::Synchronize();
	AoSComplexFrom2by2Real(num_solution, Sa + N * (2 * n3 + 1), n3, temp_mat_d_2n2n, n6);

	blas_DGEMM_n(n2, n2, local_size, R_t, r[0].re, temp_mat_d_2n2n, n6, 1.0, 0.0);
    gy::Synchronize();
	AoSComplexFrom2by2Real(num_solution, Sb + N * (n3 + 1), n3, temp_mat_d_2n2n, n6);

#ifdef DIRECT_S_MATRIX_XR_RP
	if (!is_skip_xr_rp)
#endif
	{
		blas_DGEMM_n(n2, n2, local_size, R_t, p[0].re, temp_mat_d_2n2n, n6, 1.0, 0.0);
        gy::Synchronize();
		AoSComplexFrom2by2Real(num_solution, Sb + N * (2 * n3 + 1), n3, temp_mat_d_2n2n, n6);
	}


	double* P_t = tmpM[0].re;
	transpose(local_size, n2, p[0].re, local_size, P_t, n2);

	if (!is_skip_Ax_Ap) {
		blas_DGEMM_n(n2, n2, local_size, P_t, Ap[0].re, temp_mat_d_2n2n, n6, 1.0, 0.0);
        gy::Synchronize();
		AoSComplexFrom2by2Real(num_solution, Sa + N * (2 * n3 + 2), n3, temp_mat_d_2n2n, n6);
	}
	if (!is_skip_x_p) {
		blas_DGEMM_n(n2, n2, local_size, P_t, p[0].re, temp_mat_d_2n2n, n6, 1.0, 0.0);
        gy::Synchronize();
		AoSComplexFrom2by2Real(num_solution, Sb + N * (2 * n3 + 2), n3, temp_mat_d_2n2n, n6);
	}


	//number of call of dgemm//
	int num_gemm_call = 18 - 6; //6 means lower triangle
	if (is_skip_x_p) num_gemm_call -= 3;
	if (is_skip_Ax_Ap) num_gemm_call -= 3;
#ifdef DIRECT_S_MATRIX_XR_RP
	if (is_skip_xr_rp) num_gemm_call -= 2;
#endif
	return num_gemm_call*2;
}

#else
    /*
    * H-matrix and S-matrix are calculated
    *
    * H-matrix is defined by
    * (x,r,p)^t \hat{H} (x,r,p)
    *
    * S-matrix is defined by
    * (x,r,p)^t (x,r,p)
    * 
    * temporary work memory "temp_mat_d_2n2n" needs the size of double of n2 * n2.
    */

template <typename DURATION>
static
double DoubleSec(DURATION d) {
    return 1.0e-9 * (double)std::chrono::duration_cast<std::chrono::nanoseconds>(d).count();
}


inline
int MakeMatrix_z_blas(int local_size, int num_solution, OneComplex* __restrict Sa, OneComplex* __restrict Sb,
    double* __restrict temp_mat_d_2n2n,
	const SoAComplex* __restrict x, const SoAComplex* __restrict r, const SoAComplex* __restrict p,
	const SoAComplex* __restrict Ax, const SoAComplex* __restrict Ar, const SoAComplex* __restrict Ap,
    
	bool is_skip_x_p, bool is_skip_Ax_Ap, bool is_skip_xr_rp) {

	const int N = num_solution;
	const int n2 = num_solution * 2;
	const int n3 = num_solution * 3;

    gy::Synchronize();
#ifdef SUB_WATCH_ON
    StopWatch<3, true> subwatch;

#define SUB_WATCH(x)  {gy::Synchronize(); subwatch.Record(x);}

    auto GemmGFLOPS = [](int N, int M, int K, int64_t count, double time) {
        return (double)count * (double)N * (double)M * (double)K * 2.0e-9 / time;//factor 2 means fma//
        };


    std::chrono::high_resolution_clock::time_point stm, etm;
#define BEGIN_GEMM()   {stm = std::chrono::high_resolution_clock::now();}
#define END_GEMM()     {etm = std::chrono::high_resolution_clock::now(); printf("sum-DGEMM-GFLOPS: %f\n", GemmGFLOPS(n2, n2, local_size, 1, DoubleSec(etm -stm))); }

#else
#define SUB_WATCH(x)     
#define BEGIN_GEMM()   
#define END_GEMM()   
#endif

	if (!is_skip_Ax_Ap) {
#if 0//def XtX_3_4
        blas_DGEMM_t_ld(N, N, local_size, x[0].re, local_size * 2, Ax[0].re, local_size * 2, temp_mat_d_2n2n, N, 1.0, 0.0);
        blas_DGEMM_t_ld(N, N, local_size, x[0].im, local_size * 2, Ax[0].im, local_size * 2, temp_mat_d_2n2n, N, 1.0, 1.0);
        blas_DGEMM_t_ld(N, N, local_size, x[0].re, local_size * 2, Ax[0].im, local_size * 2, temp_mat_d_2n2n + N * N, N, 1.0, 0.0);
        AoSComplexFromReIm(N, Sa, n3, temp_mat_d_2n2n, N);
#else
        BEGIN_GEMM();
        blas_DGEMM_t(n2, n2, local_size, x[0].re, Ax[0].re, temp_mat_d_2n2n, n2, 1.0, 0.0);
        SUB_WATCH(0);
        END_GEMM();
        AoSComplexFrom2by2Real(num_solution, Sa, n3, temp_mat_d_2n2n, n2);
        SUB_WATCH(1);
#endif
    }
    BEGIN_GEMM();
	blas_DGEMM_t(n2, n2, local_size, x[0].re, Ar[0].re, temp_mat_d_2n2n, n2, 1.0, 0.0);
    SUB_WATCH(0);
    END_GEMM();
    AoSComplexFrom2by2Real(num_solution, Sa + N * n3, n3, temp_mat_d_2n2n, n2);
    SUB_WATCH(1);

#if 0//def XtX_3_4
    blas_DGEMM_t_ld(N, N, local_size, r[0].re, local_size * 2, Ar[0].re, local_size * 2, temp_mat_d_2n2n, N, 1.0, 0.0);
    blas_DGEMM_t_ld(N, N, local_size, r[0].im, local_size * 2, Ar[0].im, local_size * 2, temp_mat_d_2n2n, N, 1.0, 1.0);
    blas_DGEMM_t_ld(N, N, local_size, r[0].re, local_size * 2, Ar[0].im, local_size * 2, temp_mat_d_2n2n + N * N, N, 1.0, 0.0);
    SUB_WATCH(0);
    AoSComplexFromReIm(N, Sa + N * (n3 + 1), n3, temp_mat_d_2n2n, N);
    SUB_WATCH(1);
#else
    BEGIN_GEMM();    
    blas_DGEMM_t(n2, n2, local_size, r[0].re, Ar[0].re, temp_mat_d_2n2n, n2, 1.0, 0.0);    
    SUB_WATCH(0);
    END_GEMM();
    /*
    printf("GEMM-size: %d , %d, %d\n", n2, n2, local_size);
    BEGIN_GEMM();
    for (int q = 0; q < 10; ++q) {
        blas_DGEMM_t(n2, n2, local_size, r[0].re, Ar[0].re, temp_mat_d_2n2n, n2, 1.0, 0.0);
    }
    gy::Synchronize();
    END_GEMM();
    */
	AoSComplexFrom2by2Real(num_solution, Sa + N * (n3 + 1), n3, temp_mat_d_2n2n, n2);
    SUB_WATCH(1);
#endif

	if (!is_skip_Ax_Ap) {
        BEGIN_GEMM();
		blas_DGEMM_t(n2, n2, local_size, x[0].re, Ap[0].re, temp_mat_d_2n2n, n2, 1.0, 0.0);
        SUB_WATCH(0);
        END_GEMM();
		AoSComplexFrom2by2Real(num_solution, Sa + N * (2 * n3), n3, temp_mat_d_2n2n, n2);
        SUB_WATCH(1);
	}
    BEGIN_GEMM();
	blas_DGEMM_t(n2, n2, local_size, r[0].re, Ap[0].re, temp_mat_d_2n2n, n2, 1.0, 0.0);
    SUB_WATCH(0);
    END_GEMM();
	AoSComplexFrom2by2Real(num_solution, Sa + N * (2 * n3 + 1), n3, temp_mat_d_2n2n, n2);
    SUB_WATCH(1);
	if (!is_skip_Ax_Ap) {
#if 0//def XtX_3_4
        blas_DGEMM_t_ld(N, N, local_size, p[0].re, local_size * 2, Ap[0].re, local_size * 2, temp_mat_d_2n2n, N, 1.0, 0.0);
        blas_DGEMM_t_ld(N, N, local_size, p[0].im, local_size * 2, Ap[0].im, local_size * 2, temp_mat_d_2n2n, N, 1.0, 1.0);
        blas_DGEMM_t_ld(N, N, local_size, p[0].re, local_size * 2, Ap[0].im, local_size * 2, temp_mat_d_2n2n + N * N, N, 1.0, 0.0);
        SUB_WATCH(0);
        AoSComplexFromReIm(N, Sa + N * (2 * n3 + 2), n3, temp_mat_d_2n2n, N);
        SUB_WATCH(1);
#else
        BEGIN_GEMM();
		blas_DGEMM_t(n2, n2, local_size, p[0].re, Ap[0].re, temp_mat_d_2n2n, n2, 1.0, 0.0);
        SUB_WATCH(0);
        END_GEMM();
		AoSComplexFrom2by2Real(num_solution, Sa + N * (2 * n3 + 2), n3, temp_mat_d_2n2n, n2);
        SUB_WATCH(1);
#endif
    }




	if (!is_skip_x_p) {
#if 0//def XtX_3_4
        blas_DGEMM_t_ld(N, N, local_size, x[0].re, local_size * 2, x[0].re, local_size * 2, temp_mat_d_2n2n, N, 1.0, 0.0);
        blas_DGEMM_t_ld(N, N, local_size, x[0].im, local_size * 2, x[0].im, local_size * 2, temp_mat_d_2n2n, N, 1.0, 1.0);
        blas_DGEMM_t_ld(N, N, local_size, x[0].re, local_size * 2, x[0].im, local_size * 2, temp_mat_d_2n2n + N*N, N, 1.0, 0.0);
        SUB_WATCH(0);
        AoSComplexFromReIm(N, Sb, n3, temp_mat_d_2n2n, N);
        SUB_WATCH(1);
#elif defined(USE_DSYRK)
        blas_DSYRK_t(n2, local_size, x[0].re, temp_mat_d_2n2n, n2, 1.0, 0.0);
        //blas_DGEMM_t(n2, n2, local_size, x[0].re, x[0].re, temp_mat_d_2n2n, n2, 1.0, 0.0);
        SUB_WATCH(2);
        AoSComplexTrUFrom2by2Real(num_solution, Sb, n3, temp_mat_d_2n2n, n2);
        //AoSComplexFrom2by2Real(num_solution, Sb, n3, temp_mat_d_2n2n, n2);
        SUB_WATCH(1);        
#else
        BEGIN_GEMM();
		blas_DGEMM_t(n2, n2, local_size, x[0].re, x[0].re, temp_mat_d_2n2n, n2, 1.0, 0.0);
        SUB_WATCH(0);
        END_GEMM();
		AoSComplexFrom2by2Real(num_solution, Sb, n3, temp_mat_d_2n2n, n2);
        SUB_WATCH(1);
#endif
    }
#ifdef DIRECT_S_MATRIX_XR_RP
	if (!is_skip_xr_rp) 
#endif
	{
        BEGIN_GEMM();
		blas_DGEMM_t(n2, n2, local_size, x[0].re, r[0].re, temp_mat_d_2n2n, n2, 1.0, 0.0);
        SUB_WATCH(0);
        END_GEMM();
        AoSComplexFrom2by2Real(num_solution, Sb + N * n3, n3, temp_mat_d_2n2n, n2);
        SUB_WATCH(1);
	}
#if 0//def XtX_3_4
    blas_DGEMM_t_ld(N, N, local_size, r[0].re, local_size * 2, r[0].re, local_size * 2, temp_mat_d_2n2n, N, 1.0, 0.0);
    blas_DGEMM_t_ld(N, N, local_size, r[0].im, local_size * 2, r[0].im, local_size * 2, temp_mat_d_2n2n, N, 1.0, 1.0);
    blas_DGEMM_t_ld(N, N, local_size, r[0].re, local_size * 2, r[0].im, local_size * 2, temp_mat_d_2n2n + N * N, N, 1.0, 0.0);
    SUB_WATCH(0);
    AoSComplexFromReIm(N, Sb + N * (n3 + 1), n3, temp_mat_d_2n2n, N);
    SUB_WATCH(1);

#elif defined(USE_DSYRK)
    blas_DSYRK_t(n2, local_size, r[0].re, temp_mat_d_2n2n, n2, 1.0, 0.0);
    //blas_DGEMM_t(n2, n2, local_size, r[0].re, r[0].re, temp_mat_d_2n2n, n2, 1.0, 0.0);
    SUB_WATCH(2);
    AoSComplexTrUFrom2by2Real(num_solution, Sb + N * (n3 + 1), n3, temp_mat_d_2n2n, n2);
    //AoSComplexFrom2by2Real(num_solution, Sb + N * (n3 + 1), n3, temp_mat_d_2n2n, n2);
    SUB_WATCH(1);

#else
    BEGIN_GEMM();
	blas_DGEMM_t(n2, n2, local_size, r[0].re, r[0].re, temp_mat_d_2n2n, n2, 1.0, 0.0);
    SUB_WATCH(0);
    END_GEMM();
	AoSComplexFrom2by2Real(num_solution, Sb + N * (n3 + 1), n3, temp_mat_d_2n2n, n2);
    SUB_WATCH(1);
#endif

	if (!is_skip_x_p) {
        BEGIN_GEMM();
		blas_DGEMM_t(n2, n2, local_size, x[0].re, p[0].re, temp_mat_d_2n2n, n2, 1.0, 0.0);
        SUB_WATCH(0);
        END_GEMM();
		AoSComplexFrom2by2Real(num_solution, Sb + N * (2 * n3), n3, temp_mat_d_2n2n, n2);
        SUB_WATCH(1);
	}
#ifdef DIRECT_S_MATRIX_XR_RP
	if (!is_skip_xr_rp) 
#endif
	{
        BEGIN_GEMM();
		blas_DGEMM_t(n2, n2, local_size, r[0].re, p[0].re, temp_mat_d_2n2n, n2, 1.0, 0.0);
        SUB_WATCH(0);
        END_GEMM();
		AoSComplexFrom2by2Real(num_solution, Sb + N * (2 * n3 + 1), n3, temp_mat_d_2n2n, n2);
        SUB_WATCH(1);
	}
	if (!is_skip_x_p) {
#if 0//def XtX_3_4
        blas_DGEMM_t_ld(N, N, local_size, p[0].re, local_size * 2, p[0].re, local_size * 2, temp_mat_d_2n2n, N, 1.0, 0.0);
        blas_DGEMM_t_ld(N, N, local_size, p[0].im, local_size * 2, p[0].im, local_size * 2, temp_mat_d_2n2n, N, 1.0, 1.0);
        blas_DGEMM_t_ld(N, N, local_size, p[0].re, local_size * 2, p[0].im, local_size * 2, temp_mat_d_2n2n + N * N, N, 1.0, 0.0);
        SUB_WATCH(0);
        AoSComplexFromReIm(N, Sb + N * (2 * n3 + 2), n3, temp_mat_d_2n2n, N);
        SUB_WATCH(1);

#elif defined(USE_DSYRK)
        blas_DSYRK_t(n2, local_size, p[0].re, temp_mat_d_2n2n, n2, 1.0, 0.0);
        //blas_DGEMM_t(n2, n2, local_size, p[0].re, p[0].re, temp_mat_d_2n2n, n2, 1.0, 0.0);
        SUB_WATCH(2);
        AoSComplexTrUFrom2by2Real(num_solution, Sb + N * (2 * n3 + 2), n3, temp_mat_d_2n2n, n2);
        //AoSComplexFrom2by2Real(num_solution, Sb + N * (2 * n3 + 2), n3, temp_mat_d_2n2n, n2);
        SUB_WATCH(1);

#else
        BEGIN_GEMM();
		blas_DGEMM_t(n2, n2, local_size, p[0].re, p[0].re, temp_mat_d_2n2n, n2, 1.0, 0.0);
        SUB_WATCH(0);
        END_GEMM();
		AoSComplexFrom2by2Real(num_solution, Sb + N * (2 * n3 + 2), n3, temp_mat_d_2n2n, n2);
        SUB_WATCH(1);
#endif
    }

#ifdef USE_DSYRK
    //number of call of dgemm//
    //マトリックスの2倍の数値を返すように//
    //ただしDSYRKでは1倍とする//
    int num_gemm_call = 6*2 + (3+3*2); //6 means lower triangle
    if (is_skip_x_p) num_gemm_call -= 4;
    if (is_skip_Ax_Ap) num_gemm_call -= 3 * 2;
#ifdef DIRECT_S_MATRIX_XR_RP
    if (is_skip_xr_rp) num_gemm_call -= 2 * 2;
#endif
#else
	//number of call of dgemm//
    //マトリックスの2倍の数値を返すように//
	int num_gemm_call = 18 - 6; //6 means lower triangle
	if (is_skip_x_p) num_gemm_call -= 3;
	if (is_skip_Ax_Ap) num_gemm_call -= 3;
#ifdef DIRECT_S_MATRIX_XR_RP
	if (is_skip_xr_rp) num_gemm_call -= 2;
    num_gemm_call *= 2;
#endif
#endif
    gy::Synchronize();

#ifdef SUB_WATCH_ON
    if (GetProcessID(MPI_COMM_WORLD)) {

        printf("sum-DGEMM-GFLOPS-0: %f\n", GemmGFLOPS(n2, n2, local_size, subwatch.GetCount(0), subwatch.GetTime(0)));
        subwatch.Print("sub-DGEMM1-0", 0);
        subwatch.Print("sub-DGEMM1-1", 1);
#ifdef USE_DSYRK
        subwatch.Print("sub-DGEMM1-2", 2);
        printf("sum-DGEMM-GFLOPS-2: %f\n", GemmGFLOPS(n2, n2, local_size, subwatch.GetCount(2), subwatch.GetTime(2)) / 2.0);
#endif
    }
#endif

	return num_gemm_call;
}
#endif

//column major//
inline
void CopySubMat(int Nx, int Ny, OneComplex* dest, int ldd, const OneComplex* src, int lds) {
#if 1
    gy::For_1d_bundle(Ny, Nx, GY_LAMBDA(int iy, int ix){
        dest[iy + ldd * ix] = src[iy + lds * ix];
    });
#else
	#pragma ivdep
	for (int ix = 0; ix < Nx; ++ix) {
		for (int iy = 0; iy < Ny; ++iy) {
			dest[iy + ldd * ix] = src[iy + lds * ix];
		}
	}
#endif
}

inline
int NextMatrix_z(int num_solution, OneComplex* Sa, OneComplex* Sb,
	const OneComplex* CxCrCp, const double* eigen_values, bool is_use_p, bool need_H_matrix, OneComplex* temp_z_33) {

	DEBUG_PRINTF("Use BLAS in MakeMatrix_z_mat\n");


	const int N = num_solution;
	const int n3 = N * 3;
	const int n2 = N * 2;
	const int nn = N * N;
	const int next = n3 * n3 * 2;
	const OneComplex* Cx = CxCrCp;
	const OneComplex* Cr = CxCrCp + N;
	const OneComplex* Cp = CxCrCp + N * 2;
	const int stride_C = (is_use_p) ? num_solution * 3 : num_solution * 2;
	const int stride_S = num_solution * 3;

	const OneComplex ONE{ 1.0,0.0 };
	const OneComplex ZERO{ 0.0,0.0 };


    OneComplex* tmp1 = temp_z_33;
    OneComplex* tmpPtP = temp_z_33 + nn;
    OneComplex* tmpXtP = temp_z_33 + nn * 2;
    OneComplex* tmpXtX = temp_z_33 + nn * 3;

	{
		///for direct update (P^t P), (X^t P), (X^t X) /////////////////////////////////
		OneComplex* XtX = Sb;
		OneComplex* XtR = Sb + N * n3;
		OneComplex* XtP = Sb + N * 2 * n3;
		OneComplex* RtR = Sb + N * (n3 + 1);
		OneComplex* RtP = Sb + N * (2 * n3 + 1);
		OneComplex* PtP = Sb + N * (2 * n3 + 2);


		/*
		P'= R * Cr + P * Cp
		(P'^t P') = Cr^t [(R^t R) Cr + (R^t P) Cp] + Cp^t [(P^t R) Cr + (P^t P) Cp]
		*/
		//tmp1 = (R ^ t R) Cr
		blas_ZGEMM( 'N', 'N', N, N, N, &ONE, RtR, stride_S, Cr, stride_C, &ZERO, tmp1, N);
		if (is_use_p) {
			//tmp1 += (R ^ t P) Cp
			blas_ZGEMM( 'N', 'N', N, N, N, &ONE, RtP, stride_S, Cp, stride_C, &ONE, tmp1, N);
		}
		//tmpPtP = Cr^t tmp1
		blas_ZGEMM( 'C', 'N', N, N, N, &ONE, Cr, stride_C, tmp1, N, &ZERO, tmpPtP, N);

		if (is_use_p) {
			//tmp1 = (P ^ t R) Cr = (R ^ t P)^t Cr
			//tmp1 += (P ^ t P) Cp
			blas_ZGEMM( 'C', 'N', N, N, N, &ONE, RtP, stride_S, Cr, stride_C, &ZERO, tmp1, N);
			blas_ZGEMM( 'N', 'N', N, N, N, &ONE, PtP, stride_S, Cp, stride_C, &ONE, tmp1, N);
			//tmpPtP += Cp^t tmp1
			blas_ZGEMM( 'C', 'N', N, N, N, &ONE, Cp, stride_C, tmp1, N, &ONE, tmpPtP, N);
		}


		//(X'^t P') = (P'^t P') + Cx^t [(X^t R) Cr + (X^t P) Cp]
		//tmp1 = (X^t R) Cr
		blas_ZGEMM( 'N', 'N', N, N, N, &ONE, XtR, stride_S, Cr, stride_C, &ZERO, tmp1, N);
		if (is_use_p) {
			//tmp1 += (X^t P) Cp
			blas_ZGEMM( 'N', 'N', N, N, N, &ONE, XtP, stride_S, Cp, stride_C, &ONE, tmp1, N);
		}
        gy::Synchronize();

		//tmpXtP = tmpPtP
		CopySubMat(N, N, tmpXtP, N, tmpPtP, N);
		//tmpXtP += Cx^t tmp1
		blas_ZGEMM( 'C', 'N', N, N, N, &ONE, Cx, stride_C, tmp1, N, &ONE, tmpXtP, N);

		//(X'^t X') = (X'^t P') + [(X^t X) Cx + (X^t R) Cr + (X^t P)Cp]^t Cx
		//here, tmp1 == (X^t R) Cr + (X^t P)Cp
		//tmp1 += (X^t X) Cx
		blas_ZGEMM( 'N', 'N', N, N, N, &ONE, XtX, stride_S, Cx, stride_C, &ONE, tmp1, N);
        gy::Synchronize();

		//tmpXtX = tmpXtP
		CopySubMat(N, N, tmpXtX, N, tmpXtP, N);
		//tmpXtX += tmp1^t Cx
		blas_ZGEMM( 'C', 'N', N, N, N, &ONE, tmp1, N, Cx, stride_C, &ONE, tmpXtX, N);
        gy::Synchronize();

		//P'^t P' = tmpPtP;
		CopySubMat(N, N, PtP, n3, tmpPtP, N);

		//X'^t P' = tmpXtP;
		CopySubMat(N, N, XtP, n3, tmpXtP, N);

		//X'^t X' = tmpXtX;
		CopySubMat(N, N, XtX, n3, tmpXtX, N);

	}


	int num_gemm_call = is_use_p ? 11 : 6;
	if (!need_H_matrix) return num_gemm_call*2;
	
    OneComplex* tmpPtHP = temp_z_33 + nn * 4;
    OneComplex* tmpXtHP = temp_z_33 + nn * 5;
    OneComplex* tmpXtHX = temp_z_33 + nn * 6;

	{
		///for direct update (P^t HP), (X^t HP), (X^t HX) /////////////////////////////////
		OneComplex* XtHX = Sa;
		OneComplex* XtHR = Sa + N * n3;
		OneComplex* XtHP = Sa + N * 2 * n3;
		OneComplex* RtHR = Sa + N * (n3 + 1);
		OneComplex* RtHP = Sa + N * (2 * n3 + 1);
		OneComplex* PtHP = Sa + N * (2 * n3 + 2);

		
		
		/*
		P'= R * Cr + P * Cp
		(P'^t HP') = Cr^t [(R^t HR) Cr + (R^t HP) Cp] + Cp^t [(P^t HR) Cr + (P^t HP) Cp]
		*/
		//tmp1 = (R ^ t HR) Cr
		blas_ZGEMM( 'N', 'N', N, N, N, &ONE, RtHR, stride_S, Cr, stride_C, &ZERO, tmp1, N);
		if (is_use_p) {
			//tmp1 += (R ^ t HP) Cp
			blas_ZGEMM( 'N', 'N', N, N, N, &ONE, RtHP, stride_S, Cp, stride_C, &ONE, tmp1, N);
		}
		//tmpPtHP = Cr^t tmp1
		blas_ZGEMM( 'C', 'N', N, N, N, &ONE, Cr, stride_C, tmp1, N, &ZERO, tmpPtHP, N);

		if (is_use_p) {
			//tmp1 = (P ^ t HR) Cr = (R ^ t HP)^t Cr
			//tmp1 += (P ^ t HP) Cp
			blas_ZGEMM( 'C', 'N', N, N, N, &ONE, RtHP, stride_S, Cr, stride_C, &ZERO, tmp1, N);
			blas_ZGEMM( 'N', 'N', N, N, N, &ONE, PtHP, stride_S, Cp, stride_C, &ONE, tmp1, N);
			//tmpPtHP += Cp^t tmp1
			blas_ZGEMM( 'C', 'N', N, N, N, &ONE, Cp, stride_C, tmp1, N, &ONE, tmpPtHP, N);
		}


		//(X'^t HP') = (P'^t HP') + Cx^t [(X^t HR) Cr + (X^t HP) Cp]
		//tmp1 = (X^t HR) Cr
		blas_ZGEMM( 'N', 'N', N, N, N, &ONE, XtHR, stride_S, Cr, stride_C, &ZERO, tmp1, N);
		if (is_use_p) {
			//tmp1 += (X^t HP) Cp
			blas_ZGEMM( 'N', 'N', N, N, N, &ONE, XtHP, stride_S, Cp, stride_C, &ONE, tmp1, N);
		}
        gy::Synchronize();

		//tmpXtHP = tmpPtHP
		CopySubMat(N, N, tmpXtHP, N, tmpPtHP, N);
		//tmpXtHP += Cx^t tmp1
		blas_ZGEMM( 'C', 'N', N, N, N, &ONE, Cx, stride_C, tmp1, N, &ONE, tmpXtHP, N);

		//(X'^t HX') = (X'^t HP') + [(X^t HX) Cx + (X^t HR) Cr + (X^t HP)Cp]^t Cx
		//here, tmp1 == (X^t HR) Cr + (X^t HP)Cp
		//tmp1 += (X^t HX) Cx
		blas_ZGEMM( 'N', 'N', N, N, N, &ONE, XtHX, stride_S, Cx, stride_C, &ONE, tmp1, N);
        gy::Synchronize();

		//tmpXtHX = tmpXtP
		CopySubMat(N, N, tmpXtHX, N, tmpXtHP, N);
		//tmpXtHX += tmp1^t Cx
		blas_ZGEMM( 'C', 'N', N, N, N, &ONE, tmp1, N, Cx, stride_C, &ONE, tmpXtHX, N);
        gy::Synchronize();

		//P'^t HP' = tmpPtHP;
		CopySubMat(N, N, PtHP, n3, tmpPtHP, N);

		//X'^t HP' = tmpXtHP;
		CopySubMat(N, N, XtHP, n3, tmpXtHP, N);

		//X'^t HX' = tmpXtHX;
		CopySubMat(N, N, XtHX, n3, tmpXtHX, N);
	}


#ifdef DIRECT_S_MATRIX_XR_RP
	{//direct update  (X^t R) and (R^t P)
		{
			//(X'^t R') = (X'^t H X') - (X'^t X')E'//
			OneComplex* XtR = Sb + N * n3;
			int ldm = n3;

			for (int j = 0; j < N; ++j) {
				for (int i = 0; i < N; ++i) {
					XtR[i + ldm * j].r = tmpXtHX[i + N * j].r - tmpXtX[i + N * j].r * eigen_values[j];
					XtR[i + ldm * j].i = tmpXtHX[i + N * j].i - tmpXtX[i + N * j].i * eigen_values[j];
				}
			}
		}

		{
			//(R'^t P') = (X'^t H P') - E'(X'^t P')//
			OneComplex* RtP = Sb + N * (2 * n3 + 1);
			int ldm = n3;

			for (int j = 0; j < N; ++j) {
				for (int i = 0; i < N; ++i) {
					RtP[i + ldm * j].r = tmpXtHP[i + N * j].r - eigen_values[i] * tmpXtP[i + N * j].r;
					RtP[i + ldm * j].i = tmpXtHP[i + N * j].i - eigen_values[i] * tmpXtP[i + N * j].i;
				}
			}
		}
	}
#endif

	num_gemm_call += is_use_p ? 11 : 6;
	return num_gemm_call * 2;
}

/*
* 行列行列積をMPIで1次元並列する
* C += A*Bにおいて、B側を縦に分割(N列をN/P列毎に持たせる)
* 結果のCはN/P列だけが信頼できるので、あとからAllgatherで結合する.
* 
* temporally working memory "tmp_z_33" needs the size of 3N*3N of complex.
*/
inline
int NextMatrix_z_mpi1d(const MPI_Comm& comm,
    int num_solution, OneComplex* Sa, OneComplex* Sb, 
    const OneComplex* CxCrCp, const double* eigen_values, bool is_use_p, bool need_H_matrix, OneComplex* __restrict tmp_z_33) {

    DEBUG_PRINTF("Use BLAS in MakeMatrix_z_mat\n");

    StopWatch<3, true> mywatch;

    const int N = num_solution;
    const int n3 = N * 3;
    const int n2 = N * 2;
    const int nn = N * N;

    const OneComplex* Cx = CxCrCp;
    const OneComplex* Cr = CxCrCp + N;
    const OneComplex* Cp = CxCrCp + N * 2;
    const int stride_C = (is_use_p) ? num_solution * 3 : num_solution * 2;
    const int stride_S = num_solution * 3;

    const OneComplex ONE{ 1.0,0.0 };
    const OneComplex ZERO{ 0.0,0.0 };

    const int proc_id = GetProcessID(comm);
    const int num_procs = GetNumProcess(comm);

    gy::Synchronize();

    OneComplex* tmp1 = tmp_z_33;
    OneComplex* tmpPtP = tmp_z_33 + nn;
    OneComplex* tmpXtP = tmp_z_33 + nn * 2;
    OneComplex* tmpXtX = tmp_z_33 + nn * 3;

    {
        ///for direct update (P^t P), (X^t P), (X^t X) /////////////////////////////////
        OneComplex* XtX = Sb;
        OneComplex* XtR = Sb + N * n3;
        OneComplex* XtP = Sb + N * 2 * n3;
        OneComplex* RtR = Sb + N * (n3 + 1);
        OneComplex* RtP = Sb + N * (2 * n3 + 1);
        OneComplex* PtP = Sb + N * (2 * n3 + 2);

        mywatch.Restart();
        /*
        P'= R * Cr + P * Cp
        (P'^t P') = Cr^t [(R^t R) Cr + (R^t P) Cp] + Cp^t [(P^t R) Cr + (P^t P) Cp]
        */
        //tmp1 = (R ^ t R) Cr
        cblas_zgemm_mpi1d(proc_id, num_procs, 'N', 'N', N, N, N, &ONE, RtR, stride_S, Cr, stride_C, &ZERO, tmp1, N);
        if (is_use_p) {
            //tmp1 += (R ^ t P) Cp
            cblas_zgemm_mpi1d(proc_id, num_procs, 'N', 'N', N, N, N, &ONE, RtP, stride_S, Cp, stride_C, &ONE, tmp1, N);
        }
        //tmpPtP = Cr^t tmp1
        cblas_zgemm_mpi1d(proc_id, num_procs, 'C', 'N', N, N, N, &ONE, Cr, stride_C, tmp1, N, &ZERO, tmpPtP, N);

        if (is_use_p) {
            //tmp1 = (P ^ t R) Cr = (R ^ t P)^t Cr
            //tmp1 += (P ^ t P) Cp
            cblas_zgemm_mpi1d(proc_id, num_procs, 'C', 'N', N, N, N, &ONE, RtP, stride_S, Cr, stride_C, &ZERO, tmp1, N);
            cblas_zgemm_mpi1d(proc_id, num_procs, 'N', 'N', N, N, N, &ONE, PtP, stride_S, Cp, stride_C, &ONE, tmp1, N);
            //tmpPtP += Cp^t tmp1
            cblas_zgemm_mpi1d(proc_id, num_procs, 'C', 'N', N, N, N, &ONE, Cp, stride_C, tmp1, N, &ONE, tmpPtP, N);
        }
        gy::Synchronize();
        mywatch.Record(0);

        zmatrix_gather_mpi1d(proc_id, num_procs, comm, N, N, tmpPtP);
        mywatch.Record(1);

        //(X'^t P') = (P'^t P') + Cx^t [(X^t R) Cr + (X^t P) Cp]
        //tmp1 = (X^t R) Cr
        cblas_zgemm_mpi1d(proc_id, num_procs, 'N', 'N', N, N, N, &ONE, XtR, stride_S, Cr, stride_C, &ZERO, tmp1, N);
        if (is_use_p) {
            //tmp1 += (X^t P) Cp
            cblas_zgemm_mpi1d(proc_id, num_procs, 'N', 'N', N, N, N, &ONE, XtP, stride_S, Cp, stride_C, &ONE, tmp1, N);
        }
        gy::Synchronize();
        mywatch.Record(0);
        //tmpXtP = tmpPtP
        CopySubMat(N, N, tmpXtP, N, tmpPtP, N);
        mywatch.Record(2);
        //tmpXtP += Cx^t tmp1
        cblas_zgemm_mpi1d(proc_id, num_procs, 'C', 'N', N, N, N, &ONE, Cx, stride_C, tmp1, N, &ONE, tmpXtP, N);
        gy::Synchronize();
        mywatch.Record(0);
        zmatrix_gather_mpi1d(proc_id, num_procs, comm, N, N, tmpXtP);
        mywatch.Record(1);
        //(X'^t X') = (X'^t P') + [(X^t X) Cx + (X^t R) Cr + (X^t P)Cp]^t Cx
        //here, tmp1 == (X^t R) Cr + (X^t P)Cp
        //tmp1 += (X^t X) Cx
        cblas_zgemm_mpi1d(proc_id, num_procs, 'N', 'N', N, N, N, &ONE, XtX, stride_S, Cx, stride_C, &ONE, tmp1, N);
        gy::Synchronize();
        mywatch.Record(0);
        zmatrix_gather_mpi1d(proc_id, num_procs, comm, N, N, tmp1);
        mywatch.Record(1);
#if 1
        //tmpXtX = tmpXtP^t
        Conjugate(N, N, tmpXtP, N, tmpXtX, N);
        //CopySubMat(N, N, tmpXtX, N, tmpXtP, N);
        mywatch.Record(2);
        //tmpXtX += Cx^t tmp1
        cblas_zgemm_mpi1d(proc_id, num_procs, 'C', 'N', N, N, N, &ONE, Cx, stride_C, tmp1, N, &ONE, tmpXtX, N);
        gy::Synchronize();
        mywatch.Record(0);
#else
        //tmpXtX = tmpXtP
        CopySubMat(N, N, tmpXtX, N, tmpXtP, N);
        mywatch.Record(2);
        //tmpXtX += tmp1^t Cx
        cblas_zgemm_mpi1d(proc_id, num_procs, 'C', 'N', N, N, N, &ONE, tmp1, N, Cx, stride_C, &ONE, tmpXtX, N);
        gy::Synchronize();
        mywatch.Record(0);
#endif
        zmatrix_gather_mpi1d(proc_id, num_procs, comm, N, N, tmpXtX);
        mywatch.Record(1);
        //P'^t P' = tmpPtP;
        CopySubMat(N, N, PtP, n3, tmpPtP, N);

        //X'^t P' = tmpXtP;
        CopySubMat(N, N, XtP, n3, tmpXtP, N);

        //X'^t X' = tmpXtX;
        CopySubMat(N, N, XtX, n3, tmpXtX, N);
        mywatch.Record(2);

        if (proc_id==0) {
            mywatch.Print("NextMatrix_z_mpi1d(0):GEMM    ", 0);
            mywatch.Print("NextMatrix_z_mpi1d(1):AlltoAll", 1);
            mywatch.Print("NextMatrix_z_mpi1d(2):copy    ", 2);
        }
    }


    int num_gemm_call = is_use_p ? 11 : 6;
    if (!need_H_matrix) return num_gemm_call*2;

    OneComplex* tmpPtHP = tmp_z_33 + nn * 4;
    OneComplex* tmpXtHP = tmp_z_33 + nn * 5;
    OneComplex* tmpXtHX = tmp_z_33 + nn * 6;
    {
        ///for direct update (P^t HP), (X^t HP), (X^t HX) /////////////////////////////////
        OneComplex* XtHX = Sa;
        OneComplex* XtHR = Sa + N * n3;
        OneComplex* XtHP = Sa + N * 2 * n3;
        OneComplex* RtHR = Sa + N * (n3 + 1);
        OneComplex* RtHP = Sa + N * (2 * n3 + 1);
        OneComplex* PtHP = Sa + N * (2 * n3 + 2);


        /*
        P'= R * Cr + P * Cp
        (P'^t HP') = Cr^t [(R^t HR) Cr + (R^t HP) Cp] + Cp^t [(P^t HR) Cr + (P^t HP) Cp]
        */
        //tmp1 = (R ^ t HR) Cr
        cblas_zgemm_mpi1d(proc_id, num_procs,  'N', 'N', N, N, N, &ONE, RtHR, stride_S, Cr, stride_C, &ZERO, tmp1, N);
        if (is_use_p) {
            //tmp1 += (R ^ t HP) Cp
            cblas_zgemm_mpi1d(proc_id, num_procs,  'N', 'N', N, N, N, &ONE, RtHP, stride_S, Cp, stride_C, &ONE, tmp1, N);
        }
        //tmpPtHP = Cr^t tmp1
        cblas_zgemm_mpi1d(proc_id, num_procs,  'C', 'N', N, N, N, &ONE, Cr, stride_C, tmp1, N, &ZERO, tmpPtHP, N);

        if (is_use_p) {
            //tmp1 = (P ^ t HR) Cr = (R ^ t HP)^t Cr
            //tmp1 += (P ^ t HP) Cp
            cblas_zgemm_mpi1d(proc_id, num_procs,  'C', 'N', N, N, N, &ONE, RtHP, stride_S, Cr, stride_C, &ZERO, tmp1, N);
            cblas_zgemm_mpi1d(proc_id, num_procs,  'N', 'N', N, N, N, &ONE, PtHP, stride_S, Cp, stride_C, &ONE, tmp1, N);
            //tmpPtHP += Cp^t tmp1
            cblas_zgemm_mpi1d(proc_id, num_procs,  'C', 'N', N, N, N, &ONE, Cp, stride_C, tmp1, N, &ONE, tmpPtHP, N);
        }
        gy::Synchronize();

        zmatrix_gather_mpi1d(proc_id, num_procs, comm, N, N, tmpPtHP);


        //(X'^t HP') = (P'^t HP') + Cx^t [(X^t HR) Cr + (X^t HP) Cp]
        //tmp1 = (X^t HR) Cr
        cblas_zgemm_mpi1d(proc_id, num_procs,  'N', 'N', N, N, N, &ONE, XtHR, stride_S, Cr, stride_C, &ZERO, tmp1, N);
        if (is_use_p) {
            //tmp1 += (X^t HP) Cp
            cblas_zgemm_mpi1d(proc_id, num_procs,  'N', 'N', N, N, N, &ONE, XtHP, stride_S, Cp, stride_C, &ONE, tmp1, N);
        }
        gy::Synchronize();

        //tmpXtHP = tmpPtHP
        CopySubMat(N, N, tmpXtHP, N, tmpPtHP, N);
        //tmpXtHP += Cx^t tmp1
        cblas_zgemm_mpi1d(proc_id, num_procs,  'C', 'N', N, N, N, &ONE, Cx, stride_C, tmp1, N, &ONE, tmpXtHP, N);
        gy::Synchronize();

        zmatrix_gather_mpi1d(proc_id, num_procs, comm, N, N, tmpXtHP);

        //(X'^t HX') = (X'^t HP') + [(X^t HX) Cx + (X^t HR) Cr + (X^t HP)Cp]^t Cx
        //here, tmp1 == (X^t HR) Cr + (X^t HP)Cp
        //tmp1 += (X^t HX) Cx
        cblas_zgemm_mpi1d(proc_id, num_procs,  'N', 'N', N, N, N, &ONE, XtHX, stride_S, Cx, stride_C, &ONE, tmp1, N);
        gy::Synchronize();
        zmatrix_gather_mpi1d(proc_id, num_procs, comm, N, N, tmp1);

#if 1
        //tmpXtHX = tmpXtP^t
        Conjugate(N, N, tmpXtHP, N, tmpXtHX, N);
        //tmpXtHX += Cx^t tmp1
        cblas_zgemm_mpi1d(proc_id, num_procs, 'C', 'N', N, N, N, &ONE, Cx, stride_C, tmp1, N, &ONE, tmpXtHX, N);
        gy::Synchronize();

#else
        //tmpXtHX = tmpXtP
        CopySubMat(N, N, tmpXtHX, N, tmpXtHP, N);
        //tmpXtHX += tmp1^t Cx
        cblas_zgemm_mpi1d(proc_id, num_procs,  'C', 'N', N, N, N, &ONE, tmp1, N, Cx, stride_C, &ONE, tmpXtHX, N);
        gy::Synchronize();
#endif

        zmatrix_gather_mpi1d(proc_id, num_procs, comm, N, N, tmpXtHX);

        //P'^t HP' = tmpPtHP;
        CopySubMat(N, N, PtHP, n3, tmpPtHP, N);

        //X'^t HP' = tmpXtHP;
        CopySubMat(N, N, XtHP, n3, tmpXtHP, N);

        //X'^t HX' = tmpXtHX;
        CopySubMat(N, N, XtHX, n3, tmpXtHX, N);
    }


#ifdef DIRECT_S_MATRIX_XR_RP
    {//direct update  (X^t R) and (R^t P)   
        OneComplex* tmpXtR = tmp_z_33 + nn * 7;
        const int begin_n = (N * proc_id) / num_procs;
        const int end_n = (N * (proc_id + 1)) / num_procs;
        {
            //(X'^t R') = (X'^t H X') - (X'^t X')E'//
            OneComplex* XtR = Sb + N * n3;
            int ldm = n3;

#if 1
#ifndef GY_WITH_CUDA_OR_HIP
            struct alignas(16) double2 { double x; double y; };
#endif
            const double2* devXtHX = (double2*)tmpXtHX;
            const double2* devXtX = (double2*)tmpXtX;
            double2* devXtR = (double2*)tmpXtR;
            gy::For_1d_bundle<int>(N, end_n - begin_n, GY_LAMBDA(int i, int n) {
                const int j = n + begin_n;
                devXtR[i + N * j].x = devXtHX[i + N * j].x - devXtX[i + N * j].x * eigen_values[j];
                devXtR[i + N * j].y = devXtHX[i + N * j].y - devXtX[i + N * j].y * eigen_values[j];
            });

#else
            for (int j = begin_n; j < end_n; ++j) {
                for (int i = 0; i < N; ++i) {
                    tmpXtR[i + N * j].r = tmpXtHX[i + N * j].r - tmpXtX[i + N * j].r * eigen_values[j];
                    tmpXtR[i + N * j].i = tmpXtHX[i + N * j].i - tmpXtX[i + N * j].i * eigen_values[j];
                }
            }
#endif
            zmatrix_gather_mpi1d(proc_id, num_procs, comm, N, N, &tmpXtR[0]);
            CopySubMat(N, N, XtR, n3, &tmpXtR[0], N);

        }

        {
            //(R'^t P') = (X'^t H P') - E'(X'^t P')//
            OneComplex* RtP = Sb + N * (2 * n3 + 1);
            int ldm = n3;


            auto& tmpRtP = tmpXtR;

#if 1
#ifndef GY_WITH_CUDA_OR_HIP
            struct alignas(16) double2 { double x; double y; };
#endif
            const double2* devXtHP = (double2*)tmpXtHP;
            const double2* devXtP = (double2*)tmpXtP;
            double2* devRtP = (double2*)tmpRtP;
            gy::For_1d_bundle<int>(N, end_n - begin_n, GY_LAMBDA(int i, int n) {
                const int j = n + begin_n;
                devRtP[i + N * j].x = devXtHP[i + N * j].x - eigen_values[i] * devXtP[i + N * j].x;
                devRtP[i + N * j].y = devXtHP[i + N * j].y - eigen_values[i] * devXtP[i + N * j].y;
            });

#else
            for (int j = begin_n; j < end_n; ++j) {
                for (int i = 0; i < N; ++i) {
                    tmpRtP[i + N * j].r = tmpXtHP[i + N * j].r - eigen_values[i] * tmpXtP[i + N * j].r;
                    tmpRtP[i + N * j].i = tmpXtHP[i + N * j].i - eigen_values[i] * tmpXtP[i + N * j].i;
                }
            }
#endif
            zmatrix_gather_mpi1d(proc_id, num_procs, comm, N, N, &tmpRtP[0]);
            CopySubMat(N, N, RtP, n3, &tmpRtP[0], N);

        }
    }
#endif

    num_gemm_call += is_use_p ? 11 : 6;
    return num_gemm_call * 2;
}

/*
* 行列行列積をBlasMPで2次元並列する
*
* temporally working memory "tmp_z_33" needs the size of 3N*3N of complex.
*/
inline
int NextMatrix_z_blasmp(const MPI_Comm& comm,
    int num_solution, OneComplex* Sa, OneComplex* Sb,
    const OneComplex* CxCrCp, const double* eigen_values, bool is_use_p, bool need_H_matrix, OneComplex* __restrict tmp_z_33) {

    DEBUG_PRINTF("Use BLAS in MakeMatrix_z_mat\n");

#ifndef GY_WITH_CUDA_OR_HIP
    struct alignas(16) double2 { double x; double y; };
#endif


    constexpr bool is_detail_time = false;
    StopWatch<4, is_detail_time> mywatch;

    const int N = num_solution;
    const int n3 = N * 3;
    const int n2 = N * 2;
    const int nn = N * N;

    const OneComplex* Cx = CxCrCp;
    const OneComplex* Cr = CxCrCp + N;
    const OneComplex* Cp = CxCrCp + N * 2;
    const int stride_C = (is_use_p) ? num_solution * 3 : num_solution * 2;
    const int stride_S = num_solution * 3;

    const OneComplex ONE{ 1.0,0.0 };
    const OneComplex ZERO{ 0.0,0.0 };

    const int proc_id = GetProcessID(comm);
    const int num_procs = GetNumProcess(comm);

    gy::Synchronize();

    OneComplex* tmp1 = tmp_z_33;
    OneComplex* tmpPtP = tmp_z_33 + nn;
    OneComplex* tmpXtP = tmp_z_33 + nn * 2;
    OneComplex* tmpXtX = tmp_z_33 + nn * 3;
    OneComplex* tmpPtHP = tmpPtP;
    OneComplex* tmpXtHP = tmp_z_33 + nn * 4;
    OneComplex* tmpXtHX = tmp_z_33 + nn * 5;
    OneComplex* work = tmp_z_33 + nn * 6;  //necessary work size <is greater than >= 3 * nn  //

    gy::BlasMpInfo blasmp = gy::CreateBlasMP(comm);
    gy::BlasMp_Matrix_t typeC = gy::BlasMp_Matrix_t::MakeOriginalMatrix(N, N, stride_C);
    gy::BlasMp_Matrix_t typeS = gy::BlasMp_Matrix_t::MakeOriginalMatrix(N, N, stride_S);

    ///for direct update (P^t P), (X^t P), (X^t X) /////////////////////////////////
    OneComplex* XtX = Sb;
    OneComplex* XtR = Sb + N * n3;
    OneComplex* XtP = Sb + N * 2 * n3;
    OneComplex* RtR = Sb + N * (n3 + 1);
    OneComplex* RtP = Sb + N * (2 * n3 + 1);
    OneComplex* PtP = Sb + N * (2 * n3 + 2);

    OneComplex* XtHX = Sa;
    OneComplex* XtHR = Sa + N * n3;
    OneComplex* XtHP = Sa + N * 2 * n3;
    OneComplex* RtHR = Sa + N * (n3 + 1);
    OneComplex* RtHP = Sa + N * (2 * n3 + 1);
    OneComplex* PtHP = Sa + N * (2 * n3 + 2);


    mywatch.Restart();
    gy::BlasMp_Matrix_t typeN = gy::BlasMp_Matrix_t::MakeOriginalMatrix(N, N, N);
    gy::BlasMp_Matrix_t typeT = gy::BlasMp_Matrix_t::MakeOriginalMatrix(N, N, N);
    gy::BlasMp_Matrix_t typeJ = gy::BlasMp_Matrix_t::MakeOriginalMatrix(N, N, N);

    //step 1, calculate PtP and PtHP//
    //step 1-1, PtP//
    /*
    P'= R * Cr + P * Cp
    (P'^t P') = Cr^t [(R^t R) Cr + (R^t P) Cp] + Cp^t [(P^t R) Cr + (P^t P) Cp]
    */
    //tmp1 = (R ^ t R) Cr
    gy::blasmp_ZGEMM('N', 'N', ONE, typeS, RtR, typeC, Cr, ZERO, &typeN, tmp1, work, blasmp);
    if (is_use_p) {
        //tmp1 += (R ^ t P) Cp
        gy::blasmp_ZGEMM('N', 'N', ONE, typeS, RtP, typeC, Cp, ONE, &typeN, tmp1, work, blasmp);
    }
    //tmpPtP = Cr^t tmp1
    gy::blasmp_ZGEMM('C', 'N', ONE, typeC, Cr, typeN, tmp1, ZERO, &typeT, tmpPtP, work, blasmp);

    if (is_use_p) {
        //tmp1 = (P ^ t R) Cr = (R ^ t P)^t Cr
        //tmp1 += (P ^ t P) Cp
        gy::blasmp_ZGEMM('C', 'N', ONE, typeS, RtP, typeC, Cr, ZERO, &typeN, tmp1, work, blasmp);
        gy::blasmp_ZGEMM('N', 'N', ONE, typeS, PtP, typeC, Cp, ONE, &typeN, tmp1, work, blasmp);
        //tmpPtP += Cp^t tmp1
        gy::blasmp_ZGEMM('C', 'N', ONE, typeC, Cp, typeN, tmp1, ONE, &typeT, tmpPtP, work, blasmp);
    }
    gy::Synchronize();
    mywatch.Record(0);
    //tmpXtP = tmpPtP
    gy::CopyMemory(tmpXtP, tmpPtP, sizeof(OneComplex) * typeT.ld * typeT.ld_w);
    mywatch.Record(2);

    //repair full tmpPtP
    gy::blasmp_join_to_entire(typeT, tmpPtP, &typeJ, tmpPtP, work, blasmp);
    mywatch.Record(1);
    //P'^t P' = tmpPtP;
    CopySubMat(N, N, PtP, n3, tmpPtP, N);
    mywatch.Record(2);

    //step 1-2, PtHP//

    if (need_H_matrix) {

        /*
        P'= R * Cr + P * Cp
        (P'^t HP') = Cr^t [(R^t HR) Cr + (R^t HP) Cp] + Cp^t [(P^t HR) Cr + (P^t HP) Cp]
        */
        //tmp1 = (R ^ t HR) Cr
        gy::blasmp_ZGEMM('N', 'N', ONE, typeS, RtHR, typeC, Cr, ZERO, &typeN, tmp1, work, blasmp);
        if (is_use_p) {
            //tmp1 += (R ^ t HP) Cp
            gy::blasmp_ZGEMM('N', 'N', ONE, typeS, RtHP, typeC, Cp, ONE, &typeN, tmp1, work, blasmp);
        }
        //tmpPtHP = Cr^t tmp1
        gy::blasmp_ZGEMM('C', 'N', ONE, typeC, Cr, typeN, tmp1, ZERO, &typeT, tmpPtHP, work, blasmp);

        if (is_use_p) {
            //tmp1 = (P ^ t HR) Cr = (R ^ t HP)^t Cr
            //tmp1 += (P ^ t HP) Cp
            gy::blasmp_ZGEMM('C', 'N', ONE, typeS, RtHP, typeC, Cr, ZERO, &typeN, tmp1, work, blasmp);
            gy::blasmp_ZGEMM('N', 'N', ONE, typeS, PtHP, typeC, Cp, ONE, &typeN, tmp1, work, blasmp);
            //tmpPtHP += Cp^t tmp1
            gy::blasmp_ZGEMM('C', 'N', ONE, typeC, Cp, typeN, tmp1, ONE, &typeT, tmpPtHP, work, blasmp);
        }
        gy::Synchronize();
        mywatch.Record(0);

        //tmpXtHP = tmpPtHP
        gy::CopyMemory(tmpXtHP, tmpPtHP, sizeof(OneComplex) * typeT.ld * typeT.ld_w);
        mywatch.Record(2);

        //repair tmpPtHP
        gy::blasmp_join_to_entire(typeT, tmpPtHP, &typeJ, tmpPtHP, work, blasmp);
        mywatch.Record(1);

        //P'^t HP' = tmpPtHP;
        CopySubMat(N, N, PtHP, n3, tmpPtHP, N);

    }



    //step 2, XtP and XtHP//
    //step 2-1, XtP//

    //(X'^t P') = (P'^t P') + Cx^t [(X^t R) Cr + (X^t P) Cp]
    //tmp1 = (X^t R) Cr
    gy::blasmp_ZGEMM('N', 'N', ONE, typeS, XtR, typeC, Cr, ZERO, &typeN, tmp1, work, blasmp);
    if (is_use_p) {
        //tmp1 += (X^t P) Cp
        gy::blasmp_ZGEMM('N', 'N', ONE, typeS, XtP, typeC, Cp, ONE, &typeN, tmp1, work, blasmp);
    }
    gy::Synchronize();
    mywatch.Record(0);

    //tmpXtP += Cx^t tmp1
    gy::blasmp_ZGEMM('C', 'N', ONE, typeC, Cx, typeN, tmp1, ONE, &typeT, tmpXtP, work, blasmp);
    gy::Synchronize();
    mywatch.Record(0);

    //step 3-1-1, XtX//
    //(X'^t X') = (X'^t P') + [(X^t X) Cx + (X^t R) Cr + (X^t P)Cp]^t Cx
    //here, tmp1 == (X^t R) Cr + (X^t P)Cp
    //tmp1 += (X^t X) Cx
    gy::blasmp_ZGEMM('N', 'N', ONE, typeS, XtX, typeC, Cx, ONE, &typeN, tmp1, work, blasmp);
    gy::Synchronize();
    mywatch.Record(0);
    //zmatrix_gather_mpi1d(proc_id, num_procs, comm, N, N, tmp1);
    //mywatch.Record(1);

    //step 3-1-2: XtX //
    //repair full tmpXtP (temporary use the buffer of tmpPtP//
    auto* tmpXtP_entire = tmpPtP;
    gy::blasmp_join_to_entire(typeT, tmpXtP, &typeJ, tmpXtP_entire, work, blasmp);
    mywatch.Record(1);
    //X'^t P' = tmpXtP; (temporary use the buffer of tmpPtP//
    CopySubMat(N, N, XtP, n3, tmpXtP_entire, N);

    //tmpXtX = tmpXtP^t (temporary use the buffer of tmpPtP//
    Conjugate(N, N, tmpXtP_entire, N, tmpXtX, N);
    mywatch.Record(2);
    typeT = gy::BlasMp_Matrix_t::MakeOriginalMatrix(N, N, N);
    //tmpXtX += Cx^t tmp1
    gy::blasmp_ZGEMM('C', 'N', ONE, typeC, Cx, typeN, tmp1, ONE, &typeT, tmpXtX, work, blasmp);
    gy::Synchronize();
    mywatch.Record(0);


    if (need_H_matrix) {

        //step 2-2, XtHP//
        //(X'^t HP') = (P'^t HP') + Cx^t [(X^t HR) Cr + (X^t HP) Cp]
        //tmp1 = (X^t HR) Cr
        gy::blasmp_ZGEMM('N', 'N', ONE, typeS, XtHR, typeC, Cr, ZERO, &typeN, tmp1, work, blasmp);
        if (is_use_p) {
            //tmp1 += (X^t HP) Cp
            gy::blasmp_ZGEMM('N', 'N', ONE, typeS, XtHP, typeC, Cp, ONE, &typeN, tmp1, work, blasmp);
        }
        gy::Synchronize();


        //tmpXtHP += Cx^t tmp1
        gy::blasmp_ZGEMM('C', 'N', ONE, typeC, Cx, typeN, tmp1, ONE, &typeT, tmpXtHP, work, blasmp);
        gy::Synchronize();
        mywatch.Record(0);

        //step 3-2-1, XtHX//
        //(X'^t HX') = (X'^t HP') + [(X^t HX) Cx + (X^t HR) Cr + (X^t HP)Cp]^t Cx
        //here, tmp1 == (X^t HR) Cr + (X^t HP)Cp
        //tmp1 += (X^t HX) Cx
        gy::blasmp_ZGEMM('N', 'N', ONE, typeS, XtHX, typeC, Cx, ONE, &typeN, tmp1, work, blasmp);
        gy::Synchronize();
        //zmatrix_gather_mpi1d(proc_id, num_procs, comm, N, N, tmp1);
        mywatch.Record(0);

        //step 3-2-2: XtHX //
        //repair tmpXtHP (temporary use the buffer of tmpPtP//
        auto* tmpXtHP_entire = tmpPtP;
        gy::blasmp_join_to_entire(typeT, tmpXtHP, &typeJ, tmpXtHP_entire, work, blasmp);
        mywatch.Record(1);
        //X'^t HP' = tmpXtHP;
        CopySubMat(N, N, XtHP, n3, tmpXtHP_entire, N);

        //tmpXtHX = tmpXtP^t
        Conjugate(N, N, tmpXtHP_entire, N, tmpXtHX, N);
        mywatch.Record(2);
        //reset dividing
        typeT = gy::BlasMp_Matrix_t::MakeOriginalMatrix(N, N, N);
        //tmpXtHX += Cx^t tmp1
        gy::blasmp_ZGEMM('C', 'N', ONE, typeC, Cx, typeN, tmp1, ONE, &typeT, tmpXtHX, work, blasmp);
        gy::Synchronize();
        mywatch.Record(0);



#ifdef DIRECT_S_MATRIX_XR_RP
        //step 4-1: 
        //(R'^t P') = (X'^t H P') - E'(X'^t P')//
        OneComplex* RtP = Sb + N * (2 * n3 + 1);
        auto& tmpRtP = tmpPtP;
        {
            const double2* devXtHP = (double2*)tmpXtHP;
            const double2* devXtP = (double2*)tmpXtP;
            double2* devRtP = (double2*)tmpRtP;
            const int ld = typeT.ld;
            const int offset_i = ld * GetProcessID(blasmp.mpi_comm_col);
            gy::For_1d_bundle<int>(typeT.height, typeT.width, GY_LAMBDA(int i, int j) {
                const int ii = i + offset_i;
                devRtP[i + ld * j].x = devXtHP[i + ld * j].x - eigen_values[ii] * devXtP[i + ld * j].x;
                devRtP[i + ld * j].y = devXtHP[i + ld * j].y - eigen_values[ii] * devXtP[i + ld * j].y;
            });
        }
        mywatch.Record(3);

        //repair tmpRtP
        gy::blasmp_join_to_entire(typeT, tmpRtP, &typeJ, tmpRtP, work, blasmp);
        mywatch.Record(1);
        //R'^t P' = tmpRtP;
        CopySubMat(N, N, RtP, n3, &tmpRtP[0], N);
        mywatch.Record(2);

        //step 4-2: 
        //(X'^t R') = (X'^t H X') - (X'^t X')E'//
        OneComplex* XtR = Sb + N * n3;
        OneComplex* tmpXtR = tmpPtP;
        {
            const double2* devXtHX = (double2*)tmpXtHX;
            const double2* devXtX = (double2*)tmpXtX;
            double2* devXtR = (double2*)tmpXtR;
            const int ld = typeT.ld;
            const int offset_j = typeT.ld_w * GetProcessID(blasmp.mpi_comm_row);
            gy::For_1d_bundle<int>(typeT.height, typeT.width, GY_LAMBDA(int i, int j) {
                const int jj = j + offset_j;
                devXtR[i + ld * j].x = devXtHX[i + ld * j].x - devXtX[i + ld * j].x * eigen_values[jj];
                devXtR[i + ld * j].y = devXtHX[i + ld * j].y - devXtX[i + ld * j].y * eigen_values[jj];
            });
        }
        mywatch.Record(3);

        //repair tmpXtR
        gy::blasmp_join_to_entire(typeT, tmpXtR, &typeJ, tmpXtR, work, blasmp);        
        mywatch.Record(1);
        CopySubMat(N, N, XtR, n3, &tmpXtR[0], N);
        mywatch.Record(2);
#endif
    }

    //steo 3-3: store XtX and XtHX to matrix//
    //repair full tmpXtX
    gy::blasmp_join_to_entire(typeT, tmpXtX, &typeJ, tmpXtX, work, blasmp);
    mywatch.Record(1);

    //X'^t X' = tmpXtX;
    CopySubMat(N, N, XtX, n3, tmpXtX, N);
    mywatch.Record(2);

    if (need_H_matrix) {
        //repair tmpXtHX
        gy::blasmp_join_to_entire(typeT, tmpXtHX, &typeJ, tmpXtHX, work, blasmp);
        mywatch.Record(1);

        //X'^t HX' = tmpXtHX;
        CopySubMat(N, N, XtHX, n3, tmpXtHX, N);
        mywatch.Record(2);
    }




    int num_gemm_call = is_use_p ? 11 : 6;
    if (need_H_matrix) {
        num_gemm_call *= 2;
    }

    if constexpr (is_detail_time) {
        if (proc_id == 0) {
            mywatch.Print("NextMatrix_z_mpi1d(0):GEMM    ", 0);
            mywatch.Print("NextMatrix_z_mpi1d(1):AlltoAll", 1);
            mywatch.Print("NextMatrix_z_mpi1d(2):copy    ", 2);
            mywatch.Print("NextMatrix_z_mpi1d(3):diagonal", 3);
        }
    }

    gy::ReleaseBlasMP(blasmp);


    return num_gemm_call * 2;
}


void ScaleHSMatrix_z(int N, OneComplex* Sa, OneComplex* Sb, const double* sqrt_norms_x, const double* sqrt_norms_p, const double* sqrt_norms_r, bool need_H_matrix) {
	const int n3 = N * 3;
	const int ldd = n3;

	auto ScaleMatrix = [](int N, OneComplex* m, int ldm, const double* scale_i, const double* scale_j) {
		if (scale_j == nullptr) {
			//nothing to do//
		} else if (scale_i == nullptr) {
			for (int j = 0; j < N; ++j) {
				for (int i = 0; i < N; ++i) {
					m[i + ldm * j].r *= scale_j[j];
					m[i + ldm * j].i *= scale_j[j];
				}
			}
		} else {
			for (int j = 0; j < N; ++j) {
				for (int i = 0; i < N; ++i) {
					m[i + ldm * j].r *= scale_i[i] * scale_j[j];
					m[i + ldm * j].i *= scale_i[i] * scale_j[j];
				}
			}
		}
		};


	//for
	OneComplex* XtX = Sb;
	OneComplex* XtP = Sb + N * 2 * n3;
	OneComplex* PtP = Sb + N * (2 * n3 + 2);
	ScaleMatrix(N, PtP, ldd, sqrt_norms_p, sqrt_norms_p);
	ScaleMatrix(N, XtP, ldd, sqrt_norms_x, sqrt_norms_p);
	ScaleMatrix(N, XtX, ldd, sqrt_norms_x, sqrt_norms_x);

	if (!need_H_matrix) return;

	OneComplex* XtHX = Sa;
	OneComplex* XtHP = Sa + N * 2 * n3;
	OneComplex* PtHP = Sa + N * (2 * n3 + 2);
	ScaleMatrix(N, PtHP, ldd, sqrt_norms_p, sqrt_norms_p);
	ScaleMatrix(N, XtHP, ldd, sqrt_norms_x, sqrt_norms_p);
	ScaleMatrix(N, XtHX, ldd, sqrt_norms_x, sqrt_norms_x);

#ifdef DIRECT_S_MATRIX_XR_RP
	OneComplex* XtR = Sb + N * n3;
	OneComplex* RtP = Sb + N * (2 * n3 + 1);
	ScaleMatrix(N, XtR, ldd, sqrt_norms_x, sqrt_norms_r);
	ScaleMatrix(N, RtP, ldd, sqrt_norms_r, sqrt_norms_p);
#endif
}




/*
X = {x[0], x[1], ..., x[num_solution-1]}
X' = X * Cx + R * Cr + P * Cp
P' = R * Cr + P * Cp
* where when the first update (is_use_p==false),
X' = X * Cx + R * Cr
P' = R * Cr
because CG vector P == 0.
Pay attention. P' is written in buffer of P, while X' is witten in tmp. 
Then user should be copy tmp to buffer of X after this function call.

And, this function can be used for update of AX and AP when argument is 
AX = AX * Cx + AR * Cr + AP * Cp
AP = AR * Cr + AP * Cp
AX = AX * Cx + AR * Cr
AP = AR * Cr
*
* When need_Ax is true, Ap and Ap are calculated
* 
* temporary work memory:
* tmp : required size is (n * local_size) of complex 
* rotC: required size is (n * n * 3 * 2) of complex
*/
inline
int NextVector_z2(int local_size, int num_solution, const OneComplex* CxCrCp,
    bool is_use_p, 
    SoAComplex* x, SoAComplex* r, SoAComplex* p,
    SoAComplex* tmp, OneComplex* __restrict tmp_mat_z_6nn) {

    const int64_t n3 = num_solution * 3;
    // rotCおよびcnjCにはCx,Cr,Cpの回転したものが連続的に収められている.
    // これをx,r,pに乗じるが、x,r,pはSoAの複素形式になっているので,
    // totC, cnjCのオフセットおよびストライド(leading dimension)ずらしで対応する
    OneComplex* rotC = tmp_mat_z_6nn;
    OneComplex* cnjC = rotC + num_solution * n3;
    double* rotCx = (double*)rotC;
    double* rotCr = (double*)(rotC + num_solution);
    double* rotCp = (double*)(rotC + num_solution * 2);
    double* cnjCx = (double*)(cnjC);
    double* cnjCr = (double*)(cnjC + num_solution);
    double* cnjCp = (double*)(cnjC + num_solution * 2);

    const int stride_C = (is_use_p) ? num_solution * 6 : num_solution * 4;


    if (is_use_p) {

        gy::For_1d_bundle<int64_t>(n3, num_solution, GY_LAMBDA(int64_t n, int m){
            rotC[n + n3 * m].r = CxCrCp[n + n3 * m].i;
            rotC[n + n3 * m].i = CxCrCp[n + n3 * m].r;

            cnjC[n + n3 * m].r = CxCrCp[n + n3 * m].r;
            cnjC[n + n3 * m].i = -CxCrCp[n + n3 * m].i;

        });

    } else {
        const int64_t n2 = num_solution * 2;
        gy::For_1d_bundle<int64_t>(n2, num_solution, GY_LAMBDA(int64_t n, int m){
            rotC[n + n2 * m].r = CxCrCp[n + n2 * m].i;
            rotC[n + n2 * m].i = CxCrCp[n + n2 * m].r;

            cnjC[n + n2 * m].r = CxCrCp[n + n2 * m].r;
            cnjC[n + n2 * m].i = -CxCrCp[n + n2 * m].i;
        });
    }


    //TMP = R * Cr + P * Cp;
    blas_DGEMM( 'N', 'N', local_size, num_solution, num_solution * 2, 1.0, r[0].re, local_size, cnjCr, stride_C, 0.0, tmp[0].re, local_size * 2);
    blas_DGEMM( 'N', 'N', local_size, num_solution, num_solution * 2, 1.0, r[0].re, local_size, rotCr, stride_C, 0.0, tmp[0].im, local_size * 2);

    if (is_use_p) {
        blas_DGEMM( 'N', 'N', local_size, num_solution, num_solution * 2, 1.0, p[0].re, local_size, cnjCp, stride_C, 1.0, tmp[0].re, local_size * 2);
        blas_DGEMM( 'N', 'N', local_size, num_solution, num_solution * 2, 1.0, p[0].re, local_size, rotCp, stride_C, 1.0, tmp[0].im, local_size * 2);
    }
    gy::Synchronize();

    //P_next = TMP;
#if 1
    gy::CopyMemory(p[0].re, tmp[0].re, sizeof(OneComplex) * local_size * num_solution);
#elif 0
#if defined(GY_WITH_CUDA) || defined(GY_WITH_HIP)
    gyMemcpy(p[0].re, tmp[0].re, sizeof(OneComplex) * local_size * num_solution, gyMemcpyDeviceToDevice);
#else
    memcpy(p[0].re, tmp[0].re, sizeof(OneComplex) * local_size * num_solution);
#endif
#else
    for (int k = 0; k < num_solution; ++k) {
        SoAC::Copy(p[k], tmp[k], local_size);
    }
#endif

    //TMP = TMP + X * Cx;
    blas_DGEMM( 'N', 'N', local_size, num_solution, num_solution * 2, 1.0, x[0].re, local_size, cnjCx, stride_C, 1.0, tmp[0].re, local_size * 2);
    blas_DGEMM( 'N', 'N', local_size, num_solution, num_solution * 2, 1.0, x[0].re, local_size, rotCx, stride_C, 1.0, tmp[0].im, local_size * 2);
    /*
    //X_next = TMP;
    for (int k = 0; k < num_solution; ++k) {
        SoAC::Copy(x[k], tmp[k], local_size);
    }
    */
    gy::Synchronize();

    int num_gemm_call = is_use_p ? 6 : 4;
    return num_gemm_call * 2;
}

inline
void LoadSMatrix_d(int N, OneComplex* Sb, OneComplex* src) {
	const int n3 = N * 3;
	const int ldd = n3;

	OneComplex* XtX = Sb;
	OneComplex* XtP = Sb + N * 2 * n3;
	OneComplex* PtP = Sb + N * (2 * n3 + 2);
	CopySubMat(N, N, XtX, n3, src, N);
	CopySubMat(N, N, XtP, n3, src + N * N, N);
	CopySubMat(N, N, PtP, n3, src + N * N * 2, N);

}

inline
void SaveSMatrix_d(int N, OneComplex* dest, const OneComplex* Sb) {
	const int n3 = N * 3;
	const int ldd = n3;

	const OneComplex* XtX = Sb;
	const OneComplex* XtP = Sb + N * 2 * n3;
	const OneComplex* PtP = Sb + N * (2 * n3 + 2);
	CopySubMat(N, N, dest, N, XtX, n3);
	CopySubMat(N, N, dest + N * N, N, XtP, n3);
	CopySubMat(N, N, dest + N * N * 2, N, PtP, n3);

}






/*
* fixしたベクトルのactiveなベクトルが直交するようにする
* X'_act = X_act - X_fix * { (X_fix)^t * X_act}
*/
void VectorOrthogonalization(const GridRangeMPI& l_grid, SoAComplex* x_active, int num_active, const SoAComplex* x_fix, int num_fix, double* work, int is_print=0, const char* fname=nullptr) {
    if (num_fix <= 0)return;

    const int local_size = l_grid.Size3D();
    double* XftXa = work;
    OneComplex* locXftXa = (OneComplex*)(work + num_fix * 2 * num_active * 2);
    OneComplex* sumXftXa = locXftXa + num_fix * num_active;
    blas_DGEMM_t(num_fix*2, num_active * 2, local_size, x_fix[0].re, x_active[0].re, XftXa, num_fix*2, 1.0, 0.0);
    gy::Synchronize();

    for (int k = 0; k < num_active; ++k) {
        for (int j = 0; j < num_fix; ++j) {
            const double re = XftXa[(2 * j) + 2 * num_fix * (2 * k)] + XftXa[(2 * j + 1) + 2 * num_fix * (2 * k + 1)];
            const double im = -XftXa[(2 * j + 1) + 2 * num_fix * (2 * k)] + XftXa[(2 * j) + 2 * num_fix * (2 * k + 1)];
            locXftXa[j + num_fix * k].r = re;
            locXftXa[j + num_fix * k].i = im;
        }
    }
    MPI_Allreduce(locXftXa, sumXftXa, num_fix * 2 * num_active, MPI_DOUBLE, MPI_SUM, l_grid.mpi_comm);
    for (int k = 0; k < num_active; ++k) {
        for (int j = 0; j < num_fix; ++j) {
            const double re = sumXftXa[j + num_fix * k].r;
            const double im = sumXftXa[j + num_fix * k].i;
            XftXa[(2 * j) + 2 * num_fix * (2 * k)] = re;
            XftXa[(2 * j + 1) + 2 * num_fix * (2 * k + 1)] = re;
            XftXa[(2 * j) + 2 * num_fix * (2 * k + 1)] = im;
            XftXa[(2 * j + 1) + 2 * num_fix * (2 * k)] = -im;
        }
    }

    blas_DGEMM_n(local_size, num_active * 2, num_fix * 2, x_fix[0].re, XftXa, x_active[0].re, local_size, -1.0, 1.0);
    gy::Synchronize();

#ifdef DEBUG_PRINT_MATRIX
    if (is_print > 0) {

        //再度直交の確認//
        blas_DGEMM_t(num_fix * 2, num_active * 2, local_size, x_fix[0].re, x_active[0].re, XftXa, num_fix * 2, 1.0, 0.0);
        gy::Synchronize();

        for (int k = 0; k < num_active; ++k) {
            for (int j = 0; j < num_fix; ++j) {
                const double re = XftXa[(2 * j) + 2 * num_fix * (2 * k)] + XftXa[(2 * j + 1) + 2 * num_fix * (2 * k + 1)];
                const double im = -XftXa[(2 * j + 1) + 2 * num_fix * (2 * k)] + XftXa[(2 * j) + 2 * num_fix * (2 * k + 1)];
                locXftXa[j + num_fix * k].r = re;
                locXftXa[j + num_fix * k].i = im;
            }
        }
        MPI_Allreduce(locXftXa, sumXftXa, num_fix * 2 * num_active, MPI_DOUBLE, MPI_SUM, l_grid.mpi_comm);

        if (IsRoot(l_grid.mpi_comm)) {
            {
                std::string filepath("lobpcg_");
                filepath += fname + std::string("_matrix_re_");
                filepath += std::string("0", 3 - (std::to_string(is_print).length())) + std::to_string(is_print) + ".txt";
                FILE* fp = fopen(filepath.c_str(), "w");
                for (int k = 0; k < num_active; ++k) {
                    for (int j = 0; j < num_fix; ++j) {
                        fprintf(fp, "%.10f\t", sumXftXa[j + num_fix * k].r);
                    }
                    {
                        const int j = num_fix - 1;
                        fprintf(fp, "%.10f\n", sumXftXa[j + num_fix * k].r);
                    }
                }
                fclose(fp);
            }
            {
                std::string filepath("lobpcg_");
                filepath += fname + std::string("_matrix_im_");
                filepath += std::string("0", 3 - (std::to_string(is_print).length())) + std::to_string(is_print) + ".txt";
                FILE* fp = fopen(filepath.c_str(), "w");
                for (int k = 0; k < num_active; ++k) {
                    for (int j = 0; j < num_fix; ++j) {
                        fprintf(fp, "%.10f\t", sumXftXa[j + num_fix * k].i);
                    }
                    {
                        const int j = num_fix - 1;
                        fprintf(fp, "%.10f\n", sumXftXa[j + num_fix * k].i);
                    }
                }
                fclose(fp);
            }
        }
    }
#endif
}

/*
固有値問題を解くLOBPCG

問題点、iter_max==1でも前回の履歴からr,p,Ar,Apを継続させるべき

*/

template <class OperationA, class WATCH>
inline
void Eigen_z_multi_mpi(const GridRangeMPI& l_grid, const double dVol, const int num_solution, 
    const int iter_max,
    double* eigen_values, SoAComplex* eigen_vectors, double* keep, double* work, double* temporary,
    int* p_num_fix, const double residual_threshold,
    OneComplex* keep_S_matrix, int SCF_step, int sk,
#ifdef USE_SCALAPACK
    const BlacsGridInfo& blacs_grid,
#endif
    OperationA OpeA, WATCH& watch)
{
    gyCheckError(gyGetLastError(), (std::string("DEV-ERROR") + std::to_string(__LINE__)).c_str());
    //using namespace vecmath;

    int num_fix = *p_num_fix;
    int& n_head = num_fix;
    int num_active = num_solution - num_fix;
    const double residual_threshold_sq = residual_threshold * residual_threshold;

    const int proc_id = GetProcessID(l_grid.mpi_comm);
    const bool is_root = (proc_id == 0);
    const size_t local_size = l_grid.Size3D();

#ifdef SOA_ORBITAL_ORDER2
    auto IntervalPointers = [](double* a, size_t N, size_t num_solution) {
        std::vector<SoAComplex> pointers(num_solution);
        for (size_t i = 0; i < num_solution; ++i) {
            pointers[i].re = a + i * N;
            pointers[i].im = a + (i + num_solution) * N;
        }
        return pointers;
        };
#else
    auto IntervalPointers = [](double* a, size_t N, size_t num_solution) {
        std::vector<SoAComplex> pointers(num_solution);
        for (size_t i = 0; i < num_solution; ++i) {
            pointers[i].re = a + (i * 2) * N;
            pointers[i].im = a + (i * 2 + 1) * N;
        }
        return pointers;
        };
#endif

#if 1 //def USE_ARENA
    WorkArena<GY_ALIGNMENT> work_arena(work);

    auto x = eigen_vectors;
    auto r = IntervalPointers((double*)work_arena.Suballoc<OneComplex>(local_size * num_solution), local_size, num_solution);
    auto p = IntervalPointers(keep, local_size, num_solution);
    auto Ax = IntervalPointers((double*)work_arena.Suballoc<OneComplex>(local_size * num_solution), local_size, num_solution);
    auto Ar = IntervalPointers((double*)work_arena.Suballoc<OneComplex>(local_size * num_solution), local_size, num_solution);
    auto Ap = IntervalPointers((double*)work_arena.Suballoc<OneComplex>(local_size * num_solution), local_size, num_solution);
    //note: tmporary memory 'tmpM' should not be used when calling OpeA() because the temporary memory is also used in OpeA.
    auto tmpM = IntervalPointers(temporary, local_size, num_solution);
//    double* sub_buffer_head = work_arena.Suballoc<double>(1);
#else
    auto x = eigen_vectors;
    auto r = IntervalPointers(work, local_size, num_solution);
    auto p = IntervalPointers(keep, local_size, num_solution);
    auto Ax = IntervalPointers(work + local_size * num_solution * 2, local_size, num_solution);
    auto Ar = IntervalPointers(work + 2 * local_size * num_solution * 2, local_size, num_solution);
    auto Ap = IntervalPointers(work + 3 * local_size * num_solution * 2, local_size, num_solution);
    //note: tmporary memory 'tmpM' should not be used when calling OpeA() because the temporary memory is also used in OpeA.
    auto tmpM = IntervalPointers(temporary, local_size, num_solution);
    double* sub_buffer_head = work + 4 * local_size * num_solution * 2;
#endif
    
    

#ifdef VECTOR_P_NO_CONTINUE
    bool is_first_time = true;
#else
    bool is_first_time = (SCF_step == 0);
#endif
    bool is_loaded_S_matrix = false;
    constexpr int S_MAT_REST_LIMIT = 6;//S行列は数値誤差が積もるので定期的に直接計算//
#ifdef S_MATRIX_RELOAD
    //数値誤差が積もって発散する//
    if ((keep_S_matrix != nullptr) && (!is_first_time) && (SCF_step % S_MAT_REST_LIMIT != 0)) {
        LoadSMatrix_d(num_solution, Sb, keep_S_matrix);
        is_loaded_S_matrix = true;
    }
#endif




    const double limit = 1.0e-10;
#if 1 //def USE_ARENA

    double* l_norms = work_arena.Suballoc<double>(num_solution);
    double* l_norms_p = work_arena.Suballoc<double>(num_solution);
    double* l_norms_r = work_arena.Suballoc<double>(num_solution);
    double* sum_norms = work_arena.Suballoc<double>(num_solution);
    double* sum_norms_p = work_arena.Suballoc<double>(num_solution);
    double* sum_norms_r = work_arena.Suballoc<double>(num_solution);
    double* ei_mirror = work_arena.Suballoc<double>(num_solution);

#else
    //required size is (num_solution * 7);
    double* l_norms = sub_buffer_head;
    double* l_norms_p = l_norms + num_solution;
    double* l_norms_r = l_norms + num_solution * 2;
    double* sum_norms = l_norms + num_solution * 3;
    double* sum_norms_p = l_norms + num_solution * 4;
    double* sum_norms_r = l_norms + num_solution * 5;
    double* ei_mirror = l_norms + num_solution * 6;

    sub_buffer_head += num_solution * 7;
#endif

    if (is_root) {
        printf("Eigen solver with LOBPCG_z_multi_mpi\n");
    }

    //fixした成分も直交化で使うためにscalingは必要
#if 1
    
    MulC_bundle(num_solution, x[0].re, sqrt(dVol), local_size);
    gyCheckError(gyGetLastError(), (std::string("DEV-ERROR") + std::to_string(__LINE__)).c_str());
#else
    for (int k = 0; k < num_solution; ++k) {
        SoAC::MulC(x[k], sqrt(dVol), local_size);
    }
#endif
    watch.Record(10);

    //calculte initial redidual r//
    //Because Hamiltonian operator A was changed by changing electronic density rho, //
    // Ax and Ap should be re-calculated//
    OpeA((SoAComplex*)&Ax[n_head], (SoAComplex*)&x[n_head], num_active); // calculate Ax from x 	
    
    gyCheckError(gyGetLastError(), (std::string("DEV-ERROR") + std::to_string(__LINE__)).c_str());

    bool is_large_difference = false;
    double*& l_ei = l_norms;
    double*& ei = ei_mirror;
    double*& delta_ei = sum_norms_p;

#if 1
    InnerProd_Hermite_bundle(&l_ei[n_head], num_solution - n_head, x[n_head].re, Ax[n_head].re, local_size);
    gyCheckError(gyGetLastError(), (std::string("DEV-ERROR") + std::to_string(__LINE__)).c_str());
#else
    for (int k = n_head; k < num_solution; ++k) {
        l_ei[k] = SoAC::InnerProdReal(x[k], Ax[k], local_size);
    }

#endif

#if 0
    std::memcpy(&delta_ei[n_head], &eigen_values[n_head], sizeof(double) * (num_solution - n_head));
#else
    for (int k = n_head; k < num_solution; ++k) {
        delta_ei[k] = eigen_values[k];
    }
#endif

    watch.Record(10);
    MPI_Allreduce(l_ei + n_head, ei + n_head, num_active, MPI_DOUBLE, MPI_SUM, l_grid.mpi_comm);
    watch.Record(14);

#if 1
    ResidualNorm_bundle(&l_norms_r[n_head], num_solution - n_head, r[n_head].re, &ei[n_head], x[n_head].re, Ax[n_head].re, local_size);
    
    gyCheckError(gyGetLastError(), (std::string("DEV-ERROR") + std::to_string(__LINE__)).c_str());

    for (int k = n_head; k < num_solution; ++k) {
        delta_ei[k] = fabs(delta_ei[k] - ei[k]);        
    }
#else

    for (int k = n_head; k < num_solution; ++k) {
        double norm = 0.0;
        for (size_t i = 0; i < local_size; ++i) {
            r[k].re[i] = (Ax[k].re[i] - ei[k] * x[k].re[i]);
            r[k].im[i] = (Ax[k].im[i] - ei[k] * x[k].im[i]);
            norm += r[k].re[i] * r[k].re[i] + r[k].im[i] * r[k].im[i];
        }
        delta_ei[k] = fabs(delta_ei[k] - ei[k]);
        l_norms_r[k] = norm;
    }
#endif

    watch.Record(10);

    MPI_Allreduce(l_norms_r + n_head, sum_norms_r + n_head, num_active, MPI_DOUBLE, MPI_SUM, l_grid.mpi_comm);
    watch.Record(14);


#if 1


    if (proc_id == 0) {
        printf("delta Eigen value check=======\n");
        double threshold[6] = { 1.0e-6,1.0e-7,1.0e-8, 1.0e-9, 1.0e-10, 1.0e-11 };
        int fix_pos[6];
        for (int s = 0; s < 6; ++s) {
            fix_pos[s] = num_solution;
            for (int k = n_head; k < num_solution; ++k) {
                if (threshold[s] < delta_ei[k]) {
                    fix_pos[s] = k;
                    break;
                }
            }
            //printf(" delta Eigen[%d] = %.10f\n", k, delta_ei[k]);
            printf("  %.0g: %d, %.15f\n", threshold[s], fix_pos[s], delta_ei[std::min(fix_pos[s], num_solution-1)]);
        }
    }
    if (proc_id == 0) {
        printf("norm R check=======\n");
        double threshold[6] = { 1.0e-4,1.0e-5,1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9 };
        int fix_pos[6];
        for (int s = 0; s < 6; ++s) {
            fix_pos[s] = num_solution;
            const double threshold_sq = threshold[s] * threshold[s];
            for (int k = n_head; k < num_solution; ++k) {
                if (threshold_sq < sum_norms_r[k]) {
                    fix_pos[s] = k;
                    break;
                }
            }

            printf("  %.0g^2: %d, %g\n", threshold[s], fix_pos[s], sum_norms_r[std::min(fix_pos[s], num_solution - 1)]);
        }
    }

#endif


    if (proc_id == 0) {
        int fix_pos = num_solution;
        const double threshold_sq = residual_threshold* residual_threshold;
        for (int k = n_head; k < num_solution; ++k) {
            if (threshold_sq < sum_norms_r[k]) {
                fix_pos = k;
                break;
            }
        }        

        num_fix = std::min(fix_pos, num_solution / 2);
        printf("num of fix = %d / %d\n", num_fix, num_solution);

    }
    int org_active = num_active;
    MPI_Bcast(&num_fix, 1, MPI_INT, 0, l_grid.mpi_comm);
    
    num_active = num_solution - num_fix;
    if (num_active != org_active) {
        *p_num_fix = num_fix;
        
        if (proc_id == 0) {
            printf("Note: num_fix is changed from %d to %d\n", org_active, num_active); fflush(stdout);
        }
    }



    watch.Record(10);

    if (iter_max < 1) {
        //iterationしないのでnormalizeを元に戻して終了//   
#if 1

        MulC_bundle(num_solution, x[0].re, 1.0 / sqrt(dVol), local_size);
#else
        for (int k = 0; k < num_solution; ++k) {
            SoAC::MulC(x[k], 1.0 / sqrt(dVol), local_size);
        }
#endif
        watch.Record(10);
        return;
    }


    //buffer allocation///////////////

    const int N = num_active;
    const int n3 = 3 * N;
    const int nn9 = N * N * 9;

#if 1//def USE_ARENA
    
    //WorkArena<GY_ALIGNMENT> work_arena(sub_buffer_head);
    double* S_e_value = work_arena.Suballoc<double>(n3);
    OneComplex* S_e_vector = work_arena.Suballoc<OneComplex>(nn9);
    OneComplex* Sa = work_arena.Suballoc<OneComplex>(nn9);
    OneComplex* Sb = work_arena.Suballoc<OneComplex>(nn9);
    OneComplex* red_Sa = work_arena.Suballoc<OneComplex>(nn9);
    OneComplex* red_Sb = work_arena.Suballoc<OneComplex>(nn9);
    OneComplex* temp_mat33 = work_arena.Suballoc<OneComplex>(nn9);  //need 9nn of complex//
    //gy::ZeroClear<OneComplex>(Sa, (S_e_vector - Sa));
    //gy::ZeroClear<double>((double*)Sa, nn9 * 4 * 2);
#else

    //required size is (n9 * 6 * 2 + n3);
    //sub_buffer_head = (double*)(((((intptr_t)sub_buffer_head) + 16 - 1) / 16)* 16);
    
    auto AlignedSize = [](size_t size, size_t ALIGNMENT) {
        return (size_t)((size + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
    };
    size_t size_sa = AlignedSize(sizeof(OneComplex) * nn9, GY_ALIGNMENT) / sizeof(OneComplex);
    //size_t size_sa = nn9;
    printf("SIZE: nn9 = %zd, size_sa=%zd\n", nn9, size_sa);
    sub_buffer_head = (double*)work_arena.Suballoc<OneComplex>(size_sa * 6);


    OneComplex* Sa = (OneComplex*)sub_buffer_head;
    OneComplex* Sb = Sa + size_sa;
    OneComplex* red_Sa = Sa + size_sa * 2;
    OneComplex* red_Sb = Sa + size_sa * 3;
    OneComplex* S_e_vector = Sa + size_sa * 4;
    OneComplex* temp_mat33 = Sa + size_sa * 5;  //need 9nn of complex//
    double* S_e_value = (double*)(Sa + size_sa * 6);
    gy::ZeroClear<double>((double*)Sa, size_sa * 4 * 2);
#endif
    
    


    if (is_first_time) {

        gy::ZeroClear<double>(p[0].re, num_solution * local_size * 2);
        gy::ZeroClear<double>(Ap[0].re, num_solution* local_size * 2);

        gyCheckError(gyGetLastError(), (std::string("DEV-ERROR") + std::to_string(__LINE__)).c_str());
    } else {
        //低エネルギー側のベクトルをfixする場合にactiveなベクトル(の移動方向)を直交化
        if (num_active != org_active) //原理的にはnum_fixが変わったときだけでいいはずだが、数値的に安定させるために毎ステップ直交化が必要//
        {
            VectorOrthogonalization(l_grid, &p[n_head], num_active, &x[0], num_fix, tmpM[0].re, SCF_step, "PaXf");
        }
        OpeA((SoAComplex*)&Ap[n_head], (SoAComplex*)&p[n_head], num_active);

    }


    //main loop of LOBPCG


    int iter;
    for (iter = 0; iter < iter_max; ++iter) {


        if (is_root) {
            printf("  LOBPCG iteration: %d\n", iter + 1); fflush(stdout);
        }

        VectorOrthogonalization(l_grid, &r[n_head], num_active, &x[0], num_fix, tmpM[0].re, SCF_step, "RaXf");

        gyCheckError(gyGetLastError(), (std::string("DEV-ERROR") + std::to_string(__LINE__) ).c_str());

        OpeA((SoAComplex*)&Ar[n_head], (SoAComplex*)&r[n_head], num_active);

        bool required_reduce_H_S = true;

        gyCheckError(gyGetLastError(), (std::string("DEV-ERROR") + std::to_string(__LINE__)).c_str());

        int num_gemm1_call = 0;
        if (iter == 0) {
            num_gemm1_call = MakeMatrix_z_blas(local_size, num_active, Sa, Sb, (double*)temp_mat33, &x[n_head], &r[n_head], &p[n_head], &Ax[n_head], &Ar[n_head], &Ap[n_head], is_loaded_S_matrix, false, false);
        } else if ((iter % S_MAT_REST_LIMIT == 0) || is_large_difference) {
            num_gemm1_call = MakeMatrix_z_blas(local_size, num_active, Sa, Sb, (double*)temp_mat33, &x[n_head], &r[n_head], &p[n_head], &Ax[n_head], &Ar[n_head], &Ap[n_head], false, false, false);
            is_large_difference = false;
        } else {
            num_gemm1_call = MakeMatrix_z_blas(local_size, num_active, Sa, Sb, (double*)temp_mat33, &x[n_head], &r[n_head], &p[n_head], &Ax[n_head], &Ar[n_head], &Ap[n_head], true, true, true);

            required_reduce_H_S = false;

        }
        gyCheckError(gyGetLastError(), (std::string("DEV-ERROR") + std::to_string(__LINE__)).c_str());

        gy::Synchronize();
        watch.Record(17, num_gemm1_call);

        
#ifdef DEBUGPRINT3
        if (is_root) {
            for (int n = 0; n < N; ++n) {
                int idx = (n + 3 * N * n);
                printf("DEBUG-b-HS(%d):%d, %f, %f, %f, %f\n", proc_id, n, Sa[idx].r, Sa[idx].i, Sb[idx].r, Sb[idx].i);
            }
            fflush(stdout);
        }
#endif

        //functions to compress H matrix into the empty region of S matrix and to shurink the size of MPI_Allreduce//
        const int n2 = N * 2;
        

        const bool is_use_p = !((is_first_time && (iter == 0)) || is_large_difference);

        if (!is_use_p) {
            gyCheckError(gyGetLastError(), (std::string("DEV-ERROR") + std::to_string(__LINE__)).c_str());
            //Compress matrix for MPI_Reduce //
            {
                const auto* XtHX = Sa;
                const auto* XtHR = Sa + N * n3;
                auto* dstPtR = Sb + N * (n3 + 2);
                auto* dstRtX = Sb + N;                
                CopySubMat(N, N, dstPtR, n3, XtHX, n3);                
                CopySubMat(N, n2, dstRtX, n3, XtHR, n3);                
            }

            gy::Synchronize();
            watch.Record(10);

#if defined(USE_SCALAPACK) || defined(ZGEMM_3_MPI1D) || defined(ZGEMM_BLASMP)
            MPI_Allreduce(Sb, red_Sb, (N* N * 2) * 6, MPI_DOUBLE, MPI_SUM, l_grid.mpi_comm);
#else
            MPI_Reduce(Sb, red_Sb, (N * N * 2) * 6, MPI_DOUBLE, MPI_SUM, 0, l_grid.mpi_comm);
#endif

            
            watch.Record(14);

            //初回はp=0のため、3n x 3n領域を2n x 2n行列に縮小//
            // 通信ではred_Sbのみを使ったので、red_Sa(9nn領域中の8nn領域)
            //3x3領域を2x2領域に見せるためにとして,初回はPtRからPtPまでの4blockを利用する
            OneComplex* Sa2 = red_Sa;
            OneComplex* Sb2 = Sa2 + n2 * n2;

            CopySubMat(N, n2, Sa2 + N * n2, n2, red_Sb + N, n3);
            CopySubMat(N, N, Sa2, n2, red_Sb + N * (n3 + 2), n3);
            CopySubMat(N, n2, Sb2 + N * n2, n2, red_Sb + N * n3, n3);
            CopySubMat(N, N, Sb2, n2, red_Sb, n3);



            {//store reduced H and S matrix
                CopySubMat(N, N, Sa, n3, red_Sb + N * (n3 + 2), n3);
                CopySubMat(N, n2, Sa + N * n3, n3, red_Sb + N, n3);
                CopySubMat(N, N, Sb, n3, red_Sb, n3);
                CopySubMat(N, n2, Sb + N * n3, n3, red_Sb + N * n3, n3);
            }

                        
            watch.Record(10);


#ifdef DEBUG_PRINT_MATRIX
            if (proc_id == 0) {
                {
                    std::string filepath("lobpcg_H_matrix_re_");
                    filepath += std::to_string(SCF_step) + ".txt";
                    FILE* fp = fopen(filepath.c_str(), "w");
                    for (int k = 0; k < n2; ++k) {
                        for (int j = 0; j < n2 - 1; ++j) {
                            fprintf(fp, "%.10f\t", Sa2[j + n2 * k].r);
                        }
                        {
                            const int j = n2 - 1;
                            fprintf(fp, "%.10f\n", Sa2[j + n2 * k].r);
                        }
                    }
                    fclose(fp);
                }
                {
                    std::string filepath("lobpcg_H_matrix_im_");
                    filepath += std::to_string(SCF_step) + ".txt";
                    FILE* fp = fopen(filepath.c_str(), "w");
                    for (int k = 0; k < n2; ++k) {
                        for (int j = 0; j < n2 - 1; ++j) {
                            fprintf(fp, "%.10f\t", Sa2[j + n2 * k].i);
                        }
                        {
                            const int j = n2 - 1;
                            fprintf(fp, "%.10f\n", Sa2[j + n2 * k].i);
                        }
                    }
                    fclose(fp);
                }
                {
                    std::string filepath("lobpcg_S_matrix_re_");
                    filepath += std::to_string(SCF_step) + ".txt";
                    FILE* fp = fopen(filepath.c_str(), "w");
                    for (int k = 0; k < n2; ++k) {
                        for (int j = 0; j < n2 - 1; ++j) {
                            fprintf(fp, "%.10f\t", Sb2[j + n2 * k].r);
                        }
                        {
                            const int j = n2 - 1;
                            fprintf(fp, "%.10f\n", Sb2[j + n2 * k].r);
                        }
                    }
                    fclose(fp);
                }
                {
                    std::string filepath("lobpcg_S_matrix_im_");
                    filepath += std::to_string(SCF_step) + ".txt";
                    FILE* fp = fopen(filepath.c_str(), "w");
                    for (int k = 0; k < n2; ++k) {
                        for (int j = 0; j < n2 - 1; ++j) {
                            fprintf(fp, "%.10f\t", Sb2[j + n2 * k].i);
                        }
                        {
                            const int j = n2 - 1;
                            fprintf(fp, "%.10f\n", Sb2[j + n2 * k].i);
                        }
                    }
                    fclose(fp);
                }
            }
#endif
            gy::Synchronize();
#ifdef DEBUGPRINT3
            if (is_root) {
                for (int n = 0; n < 2 * N; ++n) {
                    int idx = (n + 2 * N * n);
                    printf("DEBUG-HS:%d, %f, %f, %f, %f\n", n, Sa2[idx].r, Sa2[idx].i, Sb2[idx].r, Sb2[idx].i);
                }
                fflush(stdout);
            }
#endif

#ifdef USE_SCALAPACK
            //int info = lapack_PZHEGVX_split(Sa2, Sb2, &(S_e_value[0]), &(S_e_vector[0]), 2 * N, 1, N, l_grid.mpi_comm, blacs_grid);
#ifdef USE_PZHEGVD_IMPL2
            int info = lapack_PZHEGVD_impl2_2('U', Sa2, Sb2, &(S_e_value[0]), &(S_e_vector[0]), 2 * N, blacs_grid);
#else
            int info = lapack_PZHEGVD_impl('U', Sa2, Sb2, &(S_e_value[0]), &(S_e_vector[0]), 2 * N, l_grid.mpi_comm, blacs_grid);
#endif
#elif defined(GY_WITH_CUDA_OR_HIP) && (! defined(GY_TEST_ZHEGVD_ORG) )
            //CUDAではZHEGVD_emuは遅いことをA30で確認//
            if (is_root) {
                //int info = lapack_ZHEGVX(Sa2, Sb2, &(S_e_value[0]), &(S_e_vector[0]), 2 * N, 1, N);
                Conj_U_to_L(2 * N, Sa2);
                Conj_U_to_L(2 * N, Sb2);
                //int info = lapack_ZHEGVD_emu('U', Sa2, Sb2, &(S_e_value[0]), &(S_e_vector[0]), Sa, 2 * N);
                int info = lapack_ZHEGVD_emu('U', Sa2, Sb2, &(S_e_value[0]), &(S_e_vector[0]), temp_mat33, 2 * N);
                
            }
#else
            if(is_root){

                int info = lapack_ZHEGVD('U', Sa2, Sb2, &(S_e_value[0]), &(S_e_vector[0]), 2 * N);

            }
#endif
            watch.Record(15);
            gy::Synchronize();
#ifdef DEBUGPRINT3
            if (is_root) {
                gy::Synchronize();
                for (int n = 0; n < N; ++n) {
                    printf("DEBUG-E:%d, %f\n", n, S_e_value[n]);
                }
                fflush(stdout);
            }
#endif

            MPI_Bcast(S_e_value, N, MPI_DOUBLE, 0, l_grid.mpi_comm);
            MPI_Bcast(S_e_vector, N* n2 * 2, MPI_DOUBLE, 0, l_grid.mpi_comm);
            watch.Record(14);
            //const int min_id = (S_e_value[0] < S_e_value[1]) ? 0 : 1;
            //固有値の小さい順に並んでいるもの考えてよい//


        } else {//(is_use_p==true)//


            auto StoreReducedMatrix33 = [](int N, OneComplex* Sb, const OneComplex* red_Sb) {
                auto n2 = N * 2;
                auto n3 = N * 3;

                OneComplex* XtX = Sb;
                OneComplex* XtR = Sb + N * n3;
                OneComplex* XtP = Sb + N * 2 * n3;
                const OneComplex* red_XtX = red_Sb;
                const OneComplex* red_XtR = red_Sb + N * n3;
                const OneComplex* red_XtP = red_Sb + N * 2 * n3;

                CopySubMat(N, N, XtX, n3, red_XtX, n3);
                CopySubMat(N, n2, XtR, n3, red_XtR, n3);
                CopySubMat(N, n3, XtP, n3, red_XtP, n3);;
                };
            if (required_reduce_H_S) {
                
                //compress matrix for MPI_reduce//
                {
                    const auto* XtHX = Sa;
                    const auto* XtX = Sb;
                    auto* dstPtHR = Sa + N * (n3 + 2);
                    auto* dstPtR = Sb + N * (n3 + 2);

                    CopySubMat(N, N, dstPtHR, n3, XtHX, n3);                    
                    CopySubMat(N, N, dstPtR, n3, XtX, n3);                    
                }

                gy::Synchronize();
                watch.Record(10);
#if defined(USE_SCALAPACK) || defined(ZGEMM_3_MPI1D) || defined(ZGEMM_BLASMP)
                MPI_Allreduce(Sa + N * n3, red_Sa + N * n3, (N * N * 2) * 6, MPI_DOUBLE, MPI_SUM, l_grid.mpi_comm);
                MPI_Allreduce(Sb + N * n3, red_Sb + N * n3, (N * N * 2) * 6, MPI_DOUBLE, MPI_SUM, l_grid.mpi_comm);
#else
                MPI_Reduce(Sa + N * n3, red_Sa + N * n3, (N * N * 2) * 6, MPI_DOUBLE, MPI_SUM, 0, l_grid.mpi_comm);
                MPI_Reduce(Sb + N * n3, red_Sb + N * n3, (N * N * 2) * 6, MPI_DOUBLE, MPI_SUM, 0, l_grid.mpi_comm);
#endif
                //expand matrix for MPI_reduce//
                {
                    auto* dstXtHX = red_Sa;
                    auto* dstXtX = red_Sb;
                    const auto* PtHR = red_Sa + N * (n3 + 2);
                    const auto* PtR = red_Sb + N * (n3 + 2);
                    CopySubMat(N, N, dstXtHX, n3, PtHR, n3);
                    CopySubMat(N, N, dstXtX, n3, PtR, n3);
                }
                watch.Record(14);

                StoreReducedMatrix33(N, Sa, red_Sa);
                StoreReducedMatrix33(N, Sb, red_Sb);


            } else {
                //この場合に通信するのは,XtHR,RtHR,RtHP,RtRの4つ//
                //ただしDIRECT_S_MATRIX_XR_RPが未定義の場合はXtR,RtPも通信が必要//

                const size_t offset_XtR = N * n3;
                const size_t offset_RtP = N * (2 * n3 + 1);
                const size_t offset_PtR = N * (n3 + 2);

                //reduce XtHR,RtHR,RtHP//
                CopySubMat(N, N, Sa + offset_PtR, n3, Sa + offset_RtP, n3);
                gy::Synchronize();
                watch.Record(10);
                MPI_Allreduce(Sa + offset_XtR, red_Sa + offset_XtR, (N * N * 2) * 3, MPI_DOUBLE, MPI_SUM, l_grid.mpi_comm);
                watch.Record(14);
                auto* RtHP2 = red_Sa + N * (2 * n3 + 1);
                auto* PtHR2 = red_Sa + N * (n3 + 2);
                CopySubMat(N, N, red_Sa + offset_RtP, n3, red_Sa + offset_PtR, n3);

                //store for direct calculation of matrix//
                CopySubMat(N, n2, Sa + offset_XtR, n3, red_Sa + offset_XtR, n3);
                CopySubMat(N, N, Sa + offset_RtP, n3, red_Sa + offset_RtP, n3);

                //load from direct calculated matrix//
                const size_t offset_XtP = N * 2 * n3;
                const size_t offset_PtP = N * (2 * n3 + 2);
                CopySubMat(N, N, red_Sa, n3, Sa, n3);
                CopySubMat(N, N, red_Sa + offset_XtP, n3, Sa + offset_XtP, n3);
                CopySubMat(N, N, red_Sa + offset_PtP, n3, Sa + offset_PtP, n3);

#ifdef DIRECT_S_MATRIX_XR_RP
                //transfer only RtR, XtR, RtP //
                const size_t offset_RtR = N * (n3 + 1);
                //auto* RtR = Sb + N * (n3 + 1);
                CopySubMat(N, N, red_Sb, N, Sb + offset_RtR, n3);
                gy::Synchronize();
                watch.Record(10);
                MPI_Allreduce(red_Sb, red_Sb + N * N, (N * N * 2) * 1, MPI_DOUBLE, MPI_SUM, l_grid.mpi_comm);
                watch.Record(14);
                //auto* RtR2 = red_Sb + N * (n3 + 1);
                CopySubMat(N, N, red_Sb + offset_RtR, n3, red_Sb + N * N, N);

                //store for direct calculation of matrix//
                CopySubMat(N, N, Sb + offset_RtR, n3, red_Sb + N * N, N);

                //load from direct calculated matrix//
                CopySubMat(N, N, red_Sb + offset_XtR, n3, Sb + offset_XtR, n3);
                CopySubMat(N, N, red_Sb + offset_RtP, n3, Sb + offset_RtP, n3);

#else
                //transfer RtR //
                auto* RtP = Sb + N * (2 * n3 + 1);
                auto* PtR = Sb + N * (n3 + 2);
                CopySubMat(N, N, PtR, n3, RtP, n3);
                watch.Record(10);
                MPI_Allreduce(Sb + N * n3, red_Sb + N * n3, (N * N * 2) * 3, MPI_DOUBLE, MPI_SUM, l_grid.mpi_comm);
                watch.Record(14);
                auto* RtP2 = red_Sb + N * (2 * n3 + 1);
                auto* PtR2 = red_Sb + N * (n3 + 2);
                CopySubMat(N, N, RtP2, n3, PtR2, n3);

                //store for direct calculation of matrix//
                CopySubMat(N, N * 2, Sb + N * n3, n3, red_Sb + N * n3, n3);
                CopySubMat(N, N, RtP, n3, RtP2, n3);
#endif

                //load from direct calculated matrix//
                CopySubMat(N, N, red_Sb, n3, Sb, n3);
                CopySubMat(N, N, red_Sb + offset_XtP, n3, Sb + offset_XtP, n3);
                CopySubMat(N, N, red_Sb + offset_PtP, n3, Sb + offset_PtP, n3);


            }
            watch.Record(10);

#ifdef DEBUG_PRINT_MATRIX
            if (proc_id == 0) {
                {
                    std::string filepath("lobpcg_H_matrix_re_");
                    filepath += std::to_string(SCF_step) + ".txt";
                    FILE* fp = fopen(filepath.c_str(), "w");
                    for (int k = 0; k < n3; ++k) {
                        for (int j = 0; j < n3-1; ++j) {
                            fprintf(fp, "%.10f\t", red_Sa[j + n3 * k].r);
                        }
                        {
                            const int j = n3 - 1;
                            fprintf(fp, "%.10f\n", red_Sa[j + n3 * k].r);
                        }
                    }
                    fclose(fp);
                }
                {
                    std::string filepath("lobpcg_H_matrix_im_");
                    filepath += std::to_string(SCF_step) + ".txt";
                    FILE* fp = fopen(filepath.c_str(), "w");
                    for (int k = 0; k < n3; ++k) {
                        for (int j = 0; j < n3 - 1; ++j) {
                            fprintf(fp, "%.10f\t", red_Sa[j + n3 * k].i);
                        }
                        {
                            const int j = n3 - 1;
                            fprintf(fp, "%.10f\n", red_Sa[j + n3 * k].i);
                        }
                    }
                    fclose(fp);
                }
                {
                    std::string filepath("lobpcg_S_matrix_re_");
                    filepath += std::to_string(SCF_step) + ".txt";
                    FILE* fp = fopen(filepath.c_str(), "w");
                    for (int k = 0; k < n3; ++k) {
                        for (int j = 0; j < n3 - 1; ++j) {
                            fprintf(fp, "%.10f\t", red_Sb[j + n3 * k].r);
                        }
                        {
                            const int j = n3 - 1;
                            fprintf(fp, "%.10f\n", red_Sb[j + n3 * k].r);
                        }
                    }
                    fclose(fp);
                }
                {
                    std::string filepath("lobpcg_S_matrix_im_");
                    filepath += std::to_string(SCF_step) + ".txt";
                    FILE* fp = fopen(filepath.c_str(), "w");
                    for (int k = 0; k < n3; ++k) {
                        for (int j = 0; j < n3 - 1; ++j) {
                            fprintf(fp, "%.10f\t", red_Sb[j + n3 * k].i);
                        }
                        {
                            const int j = n3 - 1;
                            fprintf(fp, "%.10f\n", red_Sb[j + n3 * k].i);
                        }
                    }
                    fclose(fp);
                }
            }
#endif

            gy::Synchronize();
#ifdef USE_SCALAPACK
            //int info = lapack_PZHEGVX_split(&(red_Sa[0]), &(red_Sb[0]), &(S_e_value[0]), &(S_e_vector[0]), 3 * N, 1, N, l_grid.mpi_comm, blacs_grid);
#ifdef USE_PZHEGVD_IMPL2
            int info = lapack_PZHEGVD_impl2_2('U', &(red_Sa[0]), &(red_Sb[0]), &(S_e_value[0]), &(S_e_vector[0]), 3 * N, blacs_grid);
#else
            int info = lapack_PZHEGVD_impl('U', &(red_Sa[0]), &(red_Sb[0]), &(S_e_value[0]), &(S_e_vector[0]), 3 * N, l_grid.mpi_comm, blacs_grid);
#endif
#elif defined(GY_WITH_CUDA_OR_HIP) && (! defined(GY_TEST_ZHEGVD_ORG) )
            //CUDAではZHEGVD_emuは遅いことをA30で確認//
            if (is_root) {                
                //int info = lapack_ZHEGVX(&(red_Sa[0]), &(red_Sb[0]), &(S_e_value[0]), &(S_e_vector[0]), 3 * N, 1, N);
                Conj_U_to_L(3 * N, &(red_Sa[0]));
                Conj_U_to_L(3 * N, &(red_Sb[0]));


#ifdef DEBUGPRINT3

                for (int n = 0; n < 3 * N; ++n) {
                    int idx = (n + 3 * N * n);
                    printf("DEBUG-HS3:%d, %f, %f, %f, %f\n", n, red_Sa[idx].r, red_Sa[idx].i, red_Sb[idx].r, red_Sb[idx].i);
                }
                fflush(stdout);

#endif
                int info = lapack_ZHEGVD_emu('U', &(red_Sa[0]), &(red_Sb[0]), &(S_e_value[0]), &(S_e_vector[0]), temp_mat33, 3 * N);
                
            }
#else
            if(is_root){

#ifdef DEBUGPRINT3

                for (int n = 0; n < 3 * N; ++n) {
                    int idx = (n + 3 * N * n);
                    printf("DEBUG-HS3:%d, %f, %f, %f, %f\n", n, red_Sa[idx].r, red_Sa[idx].i, red_Sb[idx].r, red_Sb[idx].i);
                }
                fflush(stdout);

#endif

                int info = lapack_ZHEGVD('U', &(red_Sa[0]), &(red_Sb[0]), &(S_e_value[0]), &(S_e_vector[0]), 3 * N);


            }
#endif
            watch.Record(15);
            gy::Synchronize();
#ifdef DEBUGPRINT3
            if (is_root) {
                for (int n = 0; n < N; ++n) {
                    printf("DEBUG-E3:%d, %f\n", n, S_e_value[n]);
                }
                fflush(stdout);
            }
#endif


            MPI_Bcast(S_e_value, N, MPI_DOUBLE, 0, l_grid.mpi_comm);
            MPI_Bcast(S_e_vector, N * n3 * 2, MPI_DOUBLE, 0, l_grid.mpi_comm);
            watch.Record(14);
            //固有値の小さい順に並んでいるもの考えてよい//
            //const int min_id = (S_e_value[0] < S_e_value[1]) ? (S_e_value[0] < S_e_value[2]) ? 0 : 2 : (S_e_value[1] < S_e_value[2]) ? 1 : 2;

        }



        /*
        X = {x[0], x[1], ..., x[num_solution-1]}
        X = X * Cx + R * Cr + P * Cp
        P = R * Cr + P * Cp
        AX = AX * Cx + AR * Cr + AP * Cp
        AP = AR * Cr + AP * Cp
        where Cx,Cr,CP which are solution S_e_vector of eigen value problem
        */
        
        int num_gemm2_call = NextVector_z2(local_size, N, S_e_vector, is_use_p, &x[n_head], &r[n_head], &p[n_head], &tmpM[0], temp_mat33);
        watch.Record(16, num_gemm2_call);

        
        //X_next = TMP;
        // 
        //check norm, here.

#if 1
        Norm_bundle(l_norms, N, tmpM[0].re, local_size);
        Norm_bundle(l_norms_p, N, p[n_head].re, local_size);
        
#else
        for (int j = 0; j < N; ++j) {
            l_norms[j] = SoAC::Norm(tmpM[j], local_size);
            l_norms_p[j] = SoAC::Norm(p[j+n_head], local_size);
        }
#endif
        gy::Synchronize();
        watch.Record(10);

        const int norm_size = N * 2;
/*
#ifdef GY_WITH_CUDA
        cudaMemPrefetchAsync(l_norms, num_solution * sizeof(double), cudaCpuDeviceId);
        cudaMemPrefetchAsync(sum_norms, num_solution * sizeof(double), cudaCpuDeviceId);
        gy::Synchronize();
#endif
*/
        MPI_Allreduce(l_norms, sum_norms, norm_size, MPI_DOUBLE, MPI_SUM, l_grid.mpi_comm);
        gy::Synchronize();
        watch.Record(14);

#ifdef LOBPCG_Z_PRINT_NORM
        //show norm
        {
            if (is_root) {
                for (int j = 0; j < norm_size; ++j) {
                    printf("[%d]sk=%d, norm[%d] = %g, %g\n", proc_id, sk, j, sum_norms[j], 1.0 / sqrt(sum_norms[j]));
                }
                fflush(stdout);
            }
        }
#endif



        //正しく解けた場合はxがnormalizeされている//
        //これがずれた場合は数値誤差が積もっているのでXをrollbackし、//
        //Escapeするべし(Hを更新して次のSCF-loopで再計算)//
        double max_diff_norm_x = 0.0;
        for (int k = 0; k < N; ++k) {
            double diff = fabs(1.0 - sum_norms[k]);
            if (max_diff_norm_x < diff)max_diff_norm_x = diff;
        }
        uint8_t check_flag = 0;
        if (max_diff_norm_x > LIMIT_Z_NORM_X_FOR_RESET_MATRIX) {
            check_flag = 1;
        }

        uint8_t check_flag_sum = 0;
        MPI_Allreduce(&check_flag, &check_flag_sum, 1, MPI_UINT8_T, MPI_BOR, l_grid.mpi_comm);
        gy::Synchronize();
        is_large_difference = (check_flag_sum == 1);

        if (is_large_difference) {
            if (is_root) {
                printf("LOBPCG numerical stability breaking: | 1 - |psi|^2| = %g > %g\n", max_diff_norm_x, LIMIT_Z_NORM_X_FOR_RESET_MATRIX);
                printf("Rollback psi to the previous state, and escape from LOBPCG loop\n");
                fflush(stdout);
            }

            //////////////////////////////////////
            //escape from this loop
            //////////////////////////////////////
            //keep previous X and R, and H and S matrix is completely calculated in next step and clear P//


            //idea copy R to P for next steps//
#if 1
            gy::CopyMemory(p[n_head].re, r[n_head].re, local_size * 2 * sizeof(double) * (num_solution - n_head));
            gy::Synchronize();
#else
            for (int k = n_head; k < num_solution; ++k) {
                SoAC::Copy(p[k], r[k], local_size);
            }
#endif
            //scaling for dVol, where previous X is already normalized to 1 in grid scale//
            //脱出するため規格化(dV倍も戻す)//
#if 1
            if (num_fix > 0) {
                MulC_bundle(num_fix, x[0].re, 1.0 / sqrt(dVol), local_size);
            }
            Mul_invSqV_bundle(N, x[n_head].re, sum_norms, dVol, local_size);
#else
            for (int j = 0; j < num_fix; ++j) {
                const double coef_x = 1.0 / sqrt(dVol);
                SoAC::MulC(x[j], coef_x, local_size);
            }
            for (int j = 0; j < N; ++j) {
                const double coef_x = 1.0 / sqrt(sum_norms[j] * dVol);
                SoAC::MulC(x[j+n_head], coef_x, local_size);
            }
#endif

            break;

        } else {//(!is_large_difference)

            //X_next = TMP;
#if 1
            gy::CopyMemory(x[n_head].re, tmpM[0].re, sizeof(double) * local_size * 2 * N);
#else
            for (int k = 0; k < N; ++k) {
                SoAC::Copy(x[k+n_head], tmpM[k], local_size);
            }
#endif

#if 1
            gy::CopyMemory(&ei[n_head], &S_e_value[0], sizeof(double) * N);
            gy::Synchronize();
#else
            for (int i = 0; i < N; ++i) {
                ei[i + n_head] = S_e_value[i];
            }
#endif

#ifdef DEBUG_PRINT_MATRIX
            if (proc_id == 0) {
                std::string filepath("lobpcg_eigen");
                filepath += std::string("0", 3 - (std::to_string(SCF_step).length())) + std::to_string(SCF_step) + ".txt";
                
            }
#endif

#ifdef EVERY_NORMALIZE_P
            //normalize p to supress nan//
            {
                for (int j = 0; j < N; ++j) {
                    const double coef_p = 1.0 / sqrt(sum_norms_p[j]);
                    sum_norms_p[j+n_head] = coef_p;
                    SoAC::MulC(p[j + n_head], coef_p, local_size);
                }

                for (int j = 0; j < N; ++j) {
                    SoAC::MulC(Ap[j + n_head], sum_norms_p[j + n_head], local_size);
                }

            }
#endif


            const bool is_final_step = (iter + 1 == iter_max);
            if (!is_final_step) {

                watch.Record(10);
                gy::Synchronize();//necessary because NextVector_z2 is CPU calculation
                int num_gemm2_call = NextVector_z2(local_size, N, S_e_vector, is_use_p, &Ax[n_head], &Ar[n_head], &Ap[n_head], &tmpM[0], temp_mat33);
                watch.Record(16, num_gemm2_call);

                //AX_next = TMP;
#if 1
                gy::CopyMemory(Ax[n_head].re, tmpM[0].re, sizeof(double)* local_size * 2 * N);
                gy::Synchronize();
#else
                for (int k = 0; k < N; ++k) {
                    SoAC::Copy(Ax[k + n_head], tmpM[k], local_size);
                }
#endif



#if 1
/*
                hipMemPrefetchAsync(Ax[0].re, sizeof(double)* 2 * local_size * num_solution, 0, 0);
                hipMemPrefetchAsync(x[0].re, sizeof(double)* 2 * local_size * num_solution, 0, 0);
                hipMemPrefetchAsync(r[0].re, sizeof(double)* 2 * local_size * num_solution, 0, 0);
                hipMemPrefetchAsync(ei, sizeof(double)* num_solution, 0, 0);

                ResidualNorm_bundle(&l_norms_r[n_head], num_solution - n_head, r[n_head].re, &ei[n_head], x[n_head].re, Ax[n_head].re, local_size);
                
                printf("TEST: NORM_R\n");
                for (int j = n_head; j < num_solution; ++j) {
                    printf("norm_r[%d] = %.10f\n", j, l_norms_r[j]);
                }
                fflush(stdout);
                */
//NOTE: なぜかこのカーネルを呼ぶと、rocm6.4.3のrocSOLVERと組み合わせた時に計算がおかしくなる.
//解決: 数値誤差によりXのノルムが崩れる.//
                Residual_bundle(num_solution - n_head, r[n_head].re, &ei[n_head], x[n_head].re, Ax[n_head].re, local_size);
#else
                for (int j = n_head; j < num_solution; ++j) {
                    for (size_t i = 0; i < local_size; ++i) {
                        r[j].re[i] = (Ax[j].re[i] - ei[j] * x[j].re[i]);
                        r[j].im[i] = (Ax[j].im[i] - ei[j] * x[j].im[i]);
                    }
                }
#endif

                watch.Record(10);
                int num_gemm3_call = 0;
#ifdef ZGEMM_BLASMP
                
#ifdef USE_SCALAPACK
                if (blacs_grid.mpi_comm != l_grid.mpi_comm) {                    
                    if (blacs_grid.mpi_comm != MPI_COMM_NULL) {
                        num_gemm3_call = NextMatrix_z_blasmp(blacs_grid.mpi_comm, N, Sa, Sb, S_e_vector, S_e_value, is_use_p, !is_final_step, temp_mat33);
                    }
                    MPI_Bcast(Sa, nn9, MPI_DOUBLE, 0, l_grid.mpi_comm);
                    MPI_Bcast(Sb, nn9, MPI_DOUBLE, 0, l_grid.mpi_comm);
                    
                } else {
                    num_gemm3_call = NextMatrix_z_blasmp(l_grid.mpi_comm, N, Sa, Sb, S_e_vector, S_e_value, is_use_p, !is_final_step, temp_mat33);
                }
#else
                num_gemm3_call = NextMatrix_z_blasmp(l_grid.mpi_comm, N, Sa, Sb, S_e_vector, S_e_value, is_use_p, !is_final_step, temp_mat33);
#endif                

#elif defined( ZGEMM_3_MPI1D)
                num_gemm3_call = NextMatrix_z_mpi1d(l_grid.mpi_comm, N, Sa, Sb, S_e_vector, S_e_value, is_use_p, !is_final_step, temp_mat33);
#else
                num_gemm3_call = NextMatrix_z(N, Sa, Sb, S_e_vector, S_e_value, is_use_p, !is_final_step);
#endif
                watch.Record(19, num_gemm3_call);

#ifdef EVERY_NORMALIZE_P
                //already not required//
                //normalize XtX ... in H- and S-matrix//
                ScaleHSMatrix_z(N, Sa, Sb, nullptr, sum_norms_p, nullptr, !is_final_step);
#endif

            } else {//(is_final_step) 
                //脱出するため規格化(dV倍も戻す)//
                gyCheckError(gyGetLastError(), (std::string("DEV-ERROR") + std::to_string(__LINE__)).c_str());
                //normalize of X vector//
                if (num_fix > 0) {
                    MulC_bundle(num_fix, x[0].re, 1.0 / sqrt(dVol), local_size);
                }
                gyCheckError(gyGetLastError(), (std::string("DEV-ERROR") + std::to_string(__LINE__)).c_str());
                Mul_invSqV_bundle(N, x[n_head].re, sum_norms, dVol, local_size);

                gyCheckError(gyGetLastError(), (std::string("DEV-ERROR") + std::to_string(__LINE__)).c_str());
            }




#ifdef LOBPCG_Z_PRINT_EIGEN
            if (is_root) {
                printf("LOBPCG iter = %d\n", iter + 1);
                if (is_final_step) {
                    for (int i = 0; i < std::min(20, num_solution); ++i) {
                        printf("eigen[%d] = %f\n", i, ei[i]);
                    }
                } else {
                    for (int i = 0; i < std::min(20, num_solution); ++i) {
                        printf("eigen[%d] = %f, %f\n", i, ei[i], S_e_value[i]);
                    }
                }
                fflush(stdout);
            }

#endif

            watch.Record(10);
            //breakしなくてもistepの最終ループなので抜ける//
        }


    }

    gyCheckError(gyGetLastError(), (std::string("DEV-ERROR") + std::to_string(__LINE__)).c_str());
    ////////////////////////////End of loop of Lanczos to create Symmetric Tridiagonal T// 

#ifdef S_MATRIX_RELOAD
    if ((keep_S_matrix != nullptr)) {
        int num_gemm3_call = 0;
        num_gemm3_call = NextMatrix_z(num_active, Sa, Sb, S_e_vector, S_e_value, (iter_max > 1), false);
        watch.Record(19, num_gemm3_call);
        //normalize XtX ... in H- and S-matrix//
        ScaleHSMatrix_z(num_active, Sa, Sb, sum_norms, sum_norms_p, sum_norms_r, false);
        SaveSMatrix_d(num_active, keep_S_matrix, Sb);
    }
#endif

    for (int i = 0; i < N; ++i) {
        eigen_values[i + n_head] = ei_mirror[i + n_head];
    }


    watch.Record(10);
}

//
// 波動関数のバンドルXにたいして、5倍の領域を必要とする
// HX, R, HR, HPの要領域とtemporaryなファイル領域を合わせたもの
// 
// 追加で必要なsub-work領域のサイズを返す
// 
// doubleでのサイズを返す
//
inline
size_t WorkSize_z_multi_mpi_keep(int local_size, int num_solution) {
    AlingedMemSizeCounter<GY_ALIGNMENT> counter;
    counter.Count<OneComplex>(local_size * num_solution, 4);
    counter.Count<double>(num_solution, 1);
    counter.Count<double>(num_solution * 3, 1);
    counter.Count<double>(num_solution, 7);
    counter.Count<OneComplex>(num_solution * num_solution * 9, 6);

    return counter.Total()/sizeof(double);
    
}

//for temporary region, which is not used when calling "OpeA" functor//
inline
size_t WorkSize_z_multi_mpi_temporary(int local_size, int num_solution) {
    AlingedMemSizeCounter<GY_ALIGNMENT> counter;
    size_t main_size = counter.Count<OneComplex>(local_size * num_solution, 1); 
    return main_size / sizeof(double);
}


}

#endif

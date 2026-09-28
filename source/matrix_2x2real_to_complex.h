#pragma once
#include "gyield/gyield.h"

#ifdef GY_WITH_CUDA_OR_HIP


//note: real2x2 is double2 type which is cast ftom real(double) array//
__global__ void aos_complex_from_2by2real(int N, double2* __restrict Sa, int lda, const double2* __restrict real2x2, int ldt) {

    const int tid = threadIdx.x;
    //const int num_threads = blockDim.x;
    const int m = blockIdx.x;


    for (int n = tid; n < N; n += blockDim.x) {
        Sa[n + lda * m].x = real2x2[n + ldt * (m * 2)].x + real2x2[n + ldt * (m * 2 + 1)].y;
        Sa[n + lda * m].y = -real2x2[n + ldt * (m * 2)].y + real2x2[n + ldt * (m * 2 + 1)].x;
    }

}



template<class MYCOMPLEX>
void AoSComplexFrom2by2Real(int N, MYCOMPLEX* __restrict outZ, int lda, const double* __restrict temp_mat_d_2n2n, int ldt) {
    const int Nth = GY_MAX_THREADS_PER_BLOCK;
    aos_complex_from_2by2real <<<N, Nth >>> (N, (double2*)outZ, lda, (double2*)temp_mat_d_2n2n, ldt/2);
};


//note: real2x2 is double2 type which is cast ftom real(double) array//
__global__ void aos_complex_trU_from_2by2real(int N, double2* __restrict Sa, int lda, const double2* __restrict real2x2, int ldt) {

    const int tid = threadIdx.x;
    //const int num_threads = blockDim.x;
    const int m = blockIdx.x;


    for (int n = tid; n < m; n += blockDim.x) {
        Sa[n + lda * m].x = real2x2[n + ldt * (m * 2)].x + real2x2[n + ldt * (m * 2 + 1)].y;
        Sa[n + lda * m].y = -real2x2[n + ldt * (m * 2)].y + real2x2[n + ldt * (m * 2 + 1)].x;
        Sa[m + lda * n].x = Sa[n + lda * m].x;
        Sa[m + lda * n].y = -Sa[n + lda * m].y;
    }
    Sa[m + lda * m].x = real2x2[m + ldt * (m * 2)].x + real2x2[m + ldt * (m * 2 + 1)].y;
    Sa[m + lda * m].y = 0.0;
}




template<class MYCOMPLEX>
void AoSComplexTrUFrom2by2Real(int N, MYCOMPLEX* __restrict outZ, int lda, const double* __restrict temp_mat_d_2n2n, int ldt) {
    const int Nth = GY_MAX_THREADS_PER_BLOCK;
    aos_complex_trU_from_2by2real <<<N, Nth >>> (N, (double2*)outZ, lda, (double2*)temp_mat_d_2n2n, ldt / 2);
};


#else

template<class MYCOMPLEX>
void AoSComplexFrom2by2Real(int N, MYCOMPLEX* __restrict outZ, int lda, const double* __restrict temp_mat_d_2n2n, int ldt) {
    struct alignas(16) double2 {
        double x; double y;
    };
    double2* Sa = (double2*)outZ;
    
#pragma ivdep
    for (int m = 0; m < N; ++m) {
        for (int n = 0; n < N; ++n) {
            Sa[n + lda * m].x = temp_mat_d_2n2n[(n * 2) + ldt * (m * 2)] + temp_mat_d_2n2n[(n * 2 + 1) + ldt * (m * 2 + 1)];
            Sa[n + lda * m].y = -temp_mat_d_2n2n[(n * 2 + 1) + ldt * (m * 2)] + temp_mat_d_2n2n[(n * 2) + ldt * (m * 2 + 1)];
        }
    }
};


template<class MYCOMPLEX>
void AoSComplexTrUFrom2by2Real(int N, MYCOMPLEX* __restrict outZ, int lda, const double* __restrict temp_mat_d_2n2n, int ldt) {
    struct alignas(16) double2 {
        double x; double y;
    };
    double2* Sa = (double2*)outZ;

#pragma ivdep
    for (int m = 0; m < N; ++m) {
        for (int n = 0; n < m; ++n) {
            Sa[n + lda * m].x = temp_mat_d_2n2n[(n * 2) + ldt * (m * 2)] + temp_mat_d_2n2n[(n * 2 + 1) + ldt * (m * 2 + 1)];
            Sa[n + lda * m].y = -temp_mat_d_2n2n[(n * 2 + 1) + ldt * (m * 2)] + temp_mat_d_2n2n[(n * 2) + ldt * (m * 2 + 1)];
            Sa[m + lda * n].x = Sa[n + lda * m].x;
            Sa[m + lda * n].y = -Sa[n + lda * m].y;
        }
        Sa[m + lda * m].x = temp_mat_d_2n2n[(m * 2) + ldt * (m * 2)] + temp_mat_d_2n2n[(m * 2 + 1) + ldt * (m * 2 + 1)];
        Sa[m + lda * m].y = 0.0;
    }

};

#endif


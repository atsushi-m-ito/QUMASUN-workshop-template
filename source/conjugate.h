#pragma once
#include "gyield/gy_for.h"

#ifdef GY_WITH_CUDA_OR_HIP

template<class T>
void Conj_L_to_U(int N, T* tL) {


    double2* L = (double2*)tL;

    gy::For_1d_bundle<int>(N, N, GY_LAMBDA(int j, int k){
        const int i = k + 1 + j;
        if (i < N) {
            L[k + N * i] = { L[i + N * k].x, -L[i + N * k].y };
        }
    });

}

template<class T>
void Conj_U_to_L(int N, T* tU) {
#if 0
    struct double2 {
        double x;
        double y;
    };
#endif
    double2* U = (double2*)tU;

    gy::For_1d_bundle<int>(N, N, GY_LAMBDA(int i, int k){
        if (i < k) {
            U[k + N * i] = { U[i + N * k].x, -U[i + N * k].y };
        }
    });

}

#else
template<class T>
void Conj_L_to_U(int N, T* tL) {

    struct double2 {
        double x;
        double y;
    };
    double2* L = (double2*)tL;
#if 1
    for (int k = 0; k < N; ++k) {
        for (int i = k + 1; i < N; ++i) {
            L[k + N * i] = { L[i + N * k].x, -L[i + N * k].y };
        }
    }

#else
    for (int k = 0; k < N; ++k) {
        for (int i = 0; i < k; ++i) {
            L[i + N * k] = { L[k + N * i].x, -L[k + N * i].y };
        }
    }
#endif
}

template<class T>
void Conj_U_to_L(int N, T* tU) {
    struct double2 {
        double x;
        double y;
    };
    double2* U = (double2*)tU;

#if 1
    for (int k = 0; k < N; ++k) {
        for (int i = 0; i < k; ++i) {
            U[k + N * i] = { U[i + N * k].x, -U[i + N * k].y };
        }
    }
#else
    for (int k = 0; k < N; ++k) {
        for (int i = k + 1; i < N; ++i) {
            U[i + N * k] = { U[k + N * i].x, -U[k + N * i].y };
        }
    }
#endif
}
#endif


template<class T>
void Conjugate(int N1, int N2, T* tA, int lda, T* tC, int ldc) {

#if !defined(GY_WITH_CUDA) && !defined(GY_WITH_HIP)
    struct alignas(16) double2 {
        double x;
        double y;
    };
#endif

    double2* A = (double2*)tA;
    double2* C = (double2*)tC;

    gy::For_1d_bundle<int>(N1, N2, GY_LAMBDA(int j, int k){
        C[k + ldc * j] = { A[j + lda * k].x, -A[j + lda * k].y };
    });

}


template<class T>
void ConjugateAdd(int N1, int N2, T* tA, int lda, T* tC, int ldc) {

#if !defined(GY_WITH_CUDA) && !defined(GY_WITH_HIP)
    struct alignas(16) double2 {
        double x;
        double y;
    };
#endif

    double2* A = (double2*)tA;
    double2* C = (double2*)tC;

    gy::For_1d_bundle<int>(N1, N2, GY_LAMBDA(int j, int k){
        C[k + ldc * j].x += A[j + lda * k].x;
        C[k + ldc * j].y += -A[j + lda * k].y;
    });

}

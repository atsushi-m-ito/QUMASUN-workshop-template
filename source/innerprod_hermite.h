#pragma once
#include <cstdint>
#include "soacomplex.h"
#include "gyield/gyield.h"
#include "gyield//gy_for.h"


#ifdef GY_WITH_CUDA_OR_HIP



__global__ void innerprod_bundle(int64_t i_end, double* __restrict xHx, const double* __restrict x_soa, const double* __restrict Hx_soa) {
    extern __shared__ double smem[];
    const int tid = threadIdx.x;
    //const int num_threads = blockDim.x;
    const int n = blockIdx.x;

    double l_xHx = 0.0;
    for (int64_t i = tid; i < i_end; i += blockDim.x) {
        l_xHx += x_soa[i + i_end * n] * Hx_soa[i + i_end * n];
    }

    double sum = gy::summarize_in_block(l_xHx, smem);
    if (tid == 0) {
        xHx[n] = sum;
    }

}


__global__ void norm_bundle(int64_t i_end, double* __restrict xHx, const double* __restrict x_soa) {
    extern __shared__ double smem[];
    const int tid = threadIdx.x;
    //const int num_threads = blockDim.x;
    const int n = blockIdx.x;

    double l_xHx = 0.0;
    for (int64_t i = tid; i < i_end; i += blockDim.x) {
        l_xHx += x_soa[i + i_end * n] * x_soa[i + i_end * n];
    }
    
    double sum = gy::summarize_in_block(l_xHx, smem);
    if (tid == 0) {
        xHx[n] = sum;        
    }

}


__global__ void residual_norm_bundle(int64_t i_end, double* __restrict norms, double* __restrict r_soa, const double* __restrict ei, const double* __restrict x_soa, const double* __restrict Hx_soa) {
    extern __shared__ double smem[];
    const int tid = threadIdx.x;
    //const int num_threads = blockDim.x;
    const int n = blockIdx.x;

    double l_rr = 0.0;
    for (int64_t i = tid; i < i_end; i += blockDim.x) {
        r_soa[i + i_end * n] = Hx_soa[i + i_end * n] - ei[n] * x_soa[i + i_end * n];
        l_rr += r_soa[i + i_end * n] * r_soa[i + i_end * n];
    }

    double sum = gy::summarize_in_block(l_rr, smem);
    if (tid == 0) {
        norms[n] = sum;
    }

}

inline
void InnerProd_Hermite_bundle(double* xHx, int num_bundle, const double* __restrict x_soa_re, const double* __restrict Hx_soa_re, int64_t local_size) {
    const int64_t i_end = local_size * 2;

    const int Nth = GY_MAX_THREADS_PER_BLOCK;
    const int64_t sh_mem_size = sizeof(double) * Nth;
    innerprod_bundle <<<num_bundle, Nth, sh_mem_size >>> (i_end, xHx, x_soa_re, Hx_soa_re);
    gy::Synchronize();
}

inline
void Norm_bundle(double* xHx, int num_bundle, const double* __restrict x_soa_re, int64_t local_size) {
    const int64_t i_end = local_size * 2;

    const int Nth = GY_MAX_THREADS_PER_BLOCK;
    const int64_t sh_mem_size = sizeof(double) * Nth;
    norm_bundle <<<num_bundle, Nth, sh_mem_size >>> (i_end, xHx, x_soa_re);
    gyDeviceSynchronize();
}

inline
void ResidualNorm_bundle(double* norms, int num_bundle, double* __restrict r_soa_re, const double* __restrict ei, const double* __restrict x_soa_re, const double* __restrict Hx_soa_re, int64_t local_size) {
    const int64_t i_end = local_size * 2;

    const int Nth = GY_MAX_THREADS_PER_BLOCK;
    const int64_t sh_mem_size = sizeof(double) * Nth;
    residual_norm_bundle <<<num_bundle, Nth, sh_mem_size >>> (i_end, norms, r_soa_re, ei, x_soa_re, Hx_soa_re);
    gyDeviceSynchronize();
}

#if 0

inline
__global__ void residual_bundle(int64_t i_end, double* __restrict r_soa, const double* __restrict ei, const double* __restrict x_soa, const double* __restrict Hx_soa) {
    
    const int tid = threadIdx.x;
    //const int num_threads = blockDim.x;
    const int n = blockIdx.x;

    
    for (int64_t i = tid; i < i_end; i += blockDim.x) {
        r_soa[i + i_end * n] = Hx_soa[i + i_end * n] - ei[n] * x_soa[i + i_end * n];
    }

}


inline
void Residual_bundle(int num_bundle, double*  r_soa_re, const double*  ei, const double*  x_soa_re, const double*  Hx_soa_re, int64_t local_size) {
    const int64_t i_end = local_size * 2;
    const int Nth = GY_MAX_THREADS_PER_BLOCK;
    residual_bundle <<<num_bundle, Nth >>> (i_end, r_soa_re, ei, x_soa_re, Hx_soa_re);
    gyDeviceSynchronize();
}

#elif 1
inline
void Residual_bundle(int num_bundle, double* __restrict r_soa_re, const double* __restrict ei, const double* __restrict x_soa_re, const double* __restrict Hx_soa_re, int64_t local_size) {
    const int64_t i_end = local_size * 2;

    gy::For_1d_bundle<int64_t>(i_end, num_bundle, GY_LAMBDA(int64_t i, int n){
        r_soa_re[i + i_end * n] = Hx_soa_re[i + i_end * n] - ei[n] * x_soa_re[i + i_end * n];
    });
    gyDeviceSynchronize();
}
#else

inline
void Residual_bundle(int num_bundle, double* __restrict r_soa_re, const double* __restrict ei, const double* __restrict x_soa_re, const double* __restrict Hx_soa_re, int64_t local_size) {
    const int64_t i_end = local_size * 2;
    for (int n = 0; n < num_bundle; ++n) {
        for (int64_t i = 0; i < i_end; ++i) {
            r_soa_re[i + i_end * n] = Hx_soa_re[i + i_end * n] - ei[n] * x_soa_re[i + i_end * n];
        }
    }
}
#endif


#else

inline
void InnerProd_Hermite_bundle(double* xHx, int num_bundle, const double* __restrict x_soa_re, const double* __restrict Hx_soa_re, int64_t local_size) {
    const int64_t i_end = local_size * 2;
    for (int n = 0; n < num_bundle; ++n) {
        xHx[n] = 0.0;
        for (int64_t i = 0; i < i_end; ++i) {
            xHx[n] += x_soa_re[i + i_end * n] * Hx_soa_re[i + i_end * n];
        }
    }
}

inline
void Norm_bundle(double* xHx, int num_bundle, const double* __restrict x_soa_re, int64_t local_size) {
    const int64_t i_end = local_size * 2;
    for (int n = 0; n < num_bundle; ++n) {
        xHx[n] = 0.0;
        for (int64_t i = 0; i < i_end; ++i) {
            xHx[n] += x_soa_re[i + i_end * n] * x_soa_re[i + i_end * n];
        }
    }
}

inline
void ResidualNorm_bundle(double* norms, int num_bundle, double* __restrict r_soa_re, const double* __restrict ei, const double* __restrict x_soa_re, const double* __restrict Hx_soa_re, int64_t local_size) {
    const int64_t i_end = local_size * 2;
    for (int n = 0; n < num_bundle; ++n) {
        norms[n] = 0.0;
        for (int64_t i = 0; i < i_end; ++i) {
            r_soa_re[i + i_end * n] = Hx_soa_re[i + i_end * n] - ei[n] * x_soa_re[i + i_end * n];
            norms[n] += r_soa_re[i + i_end * n] * r_soa_re[i + i_end * n];
        }
    }
}

inline
void Residual_bundle(int num_bundle, double* __restrict r_soa_re, const double* __restrict ei, const double* __restrict x_soa_re, const double* __restrict Hx_soa_re, int64_t local_size) {
    const int64_t i_end = local_size * 2;
    for (int n = 0; n < num_bundle; ++n) {
        for (int64_t i = 0; i < i_end; ++i) {
            r_soa_re[i + i_end * n] = Hx_soa_re[i + i_end * n] - ei[n] * x_soa_re[i + i_end * n];
        }
    }
}


#endif

inline
void MulC_bundle(int num_bundle, double* __restrict xd, double c, int64_t local_size) {
    const int64_t i_end = local_size * 2;
    gy::For_1d_bundle<int64_t>(i_end, num_bundle, GY_LAMBDA(int64_t i, int n){
        xd[i + i_end * n] *= c;
    });
    gy::Synchronize();
}


/*
* n-th orbital is scaled by (1.0/srqt(norm[n] * c))
*/
inline
void Mul_invSqV_bundle(int num_bundle, double* __restrict xd, const double* __restrict norms, double c, int64_t local_size) {
    const int64_t i_end = local_size * 2;
    gy::For_1d_bundle<int64_t>(i_end, num_bundle, GY_LAMBDA(int64_t i, int n){
        const double cc = 1.0 / sqrt(norms[n] * c);
        xd[i + i_end * n] *= cc;
    });
    gy::Synchronize();
}


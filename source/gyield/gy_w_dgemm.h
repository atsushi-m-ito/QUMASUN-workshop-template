#pragma once
#include "gy_blas_init.h"


namespace gy {
    namespace blas {

        /*
        * DGEMM wrapper
        * C = alpha*(A^t*B) + beta*C
        * matrix A is KxM grids (A[iy + K * ix] where ix in [0,M), iy in [0,K)
        * matrix B is KxN grids (A[iy + K * ix] where ix in [0,N), iy in [0,K)
        */
        inline
        void DGEMM_t(int M, int N, int K, const double* A, const double* B, double* C, int strideC, double alpha, double beta) {
#ifdef GY_WITH_CUDA
            auto&& handle = SingletonHandle::Get();
            cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                M, N, K,
                &alpha,
                A, K,
                B, K,
                &beta,
                C, strideC);
#elif defined(GY_WITH_HIP)
            auto&& handle = SingletonHandle::Get();
            rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none,
                M, N, K,
                &alpha,
                A, K,
                B, K,
                &beta,
                C, strideC);
#else
            //cpu
            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, M, N, K, alpha, A, K, B, K, beta, C, strideC);
#endif       
        }

        inline
        void SGEMM_t(int M, int N, int K, const float* A, const float* B, float* C, int strideC, float alpha, float beta) {
#ifdef GY_WITH_CUDA
            auto&& handle = SingletonHandle::Get();
            cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                M, N, K,
                &alpha,
                A, K,
                B, K,
                &beta,
                C, strideC);
#elif defined(GY_WITH_HIP)
            auto&& handle = SingletonHandle::Get();
            rocblas_sgemm(handle, rocblas_operation_transpose, rocblas_operation_none,
                M, N, K,
                &alpha,
                A, K,
                B, K,
                &beta,
                C, strideC);
#else
            //cpu
            cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, M, N, K, alpha, A, K, B, K, beta, C, strideC);
#endif       
        }



        inline
        void DGEMM_bt(int M, int N, int K, const double* A, const double* B, double* C, int strideC, double alpha, double beta) {
#ifdef GY_WITH_CUDA
            auto&& handle = SingletonHandle::Get();
            cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                M, N, K,
                &alpha,
                A, K,
                B, K,
                &beta,
                C, strideC);
#elif defined(GY_WITH_HIP)
            auto&& handle = SingletonHandle::Get();
            rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_transpose,
                M, N, K,
                &alpha,
                A, K,
                B, K,
                &beta,
                C, strideC);
#else
            //cpu
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, M, N, K, alpha, A, K, B, K, beta, C, strideC);
#endif
        }

        inline
        void SGEMM_bt(int M, int N, int K, const float* A, const float* B, float* C, int strideC, float alpha, float beta) {
#ifdef GY_WITH_CUDA
            auto&& handle = SingletonHandle::Get();
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                M, N, K,
                &alpha,
                A, K,
                B, K,
                &beta,
                C, strideC);
#elif defined(GY_WITH_HIP)
            auto&& handle = SingletonHandle::Get();
            rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_transpose,
                M, N, K,
                &alpha,
                A, K,
                B, K,
                &beta,
                C, strideC);
#else
            //cpu
            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, M, N, K, alpha, A, K, B, K, beta, C, strideC);
#endif
        }

        inline
        void DGEMM_n(int M, int N, int K, const double* A, const double* B, double* C, int strideC, double alpha, double beta) {
#ifdef GY_WITH_CUDA
            auto&& handle = SingletonHandle::Get();
            cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                M, N, K,
                &alpha,
                A, M,
                B, K,
                &beta,
                C, strideC);
#elif defined(GY_WITH_HIP)
            auto&& handle = SingletonHandle::Get();
            rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_none,
                M, N, K,
                &alpha,
                A, M,
                B, K,
                &beta,
                C, strideC);
#else
            //cpu
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, M, B, K, beta, C, strideC);
#endif       
        }


        inline
        void SGEMM_n(int M, int N, int K, const float* A, const float* B, float* C, int strideC, float alpha, float beta) {
#ifdef GY_WITH_CUDA
            auto&& handle = SingletonHandle::Get();
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                M, N, K,
                &alpha,
                A, M,
                B, K,
                &beta,
                C, strideC);
#elif defined(GY_WITH_HIP)
            auto&& handle = SingletonHandle::Get();
            rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none,
                M, N, K,
                &alpha,
                A, M,
                B, K,
                &beta,
                C, strideC);
#else
            //cpu
            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, M, B, K, beta, C, strideC);
#endif       
        }

        inline
        void DGEMV_t(int M, int K, const double* A, const double* V, double* C, double alpha, double beta) {
#ifdef GY_WITH_CUDA
            auto&& handle = SingletonHandle::Get();
            cublasDgemv(handle, CUBLAS_OP_T,
                M, K,
                &alpha,
                A, K,
                V, 1,
                &beta,
                C, 1);

#elif defined(GY_WITH_HIP)
            auto&& handle = SingletonHandle::Get();
            rocblas_dgemv(handle, rocblas_operation_transpose,
                M, K,
                &alpha,
                A, K,
                V, 1,
                &beta,
                C, 1);
#else
            cblas_dgemv(CblasRowMajor, CblasNoTrans, M, K, alpha, A, K, V, 1, beta, C, 1);

#endif
        }


        template<class T>
        T* wrapCopyToDevice(
            T* dev_work, T* A, int64_t offset, size_t width) {
            if (dev_work == nullptr) return A;
            T* d_A = dev_work + offset;
            gyCheckError(gyMemcpy(d_A, A, width * sizeof(T), gyMemcpyHostToDevice), "Failed to copy to device");
            return d_A;
        }
        
        //const version
        template<class T>
        T* wrapCopyToDevice( 
            T* dev_work, const T* A, int64_t offset, size_t width) {
            if (dev_work == nullptr) return const_cast<T*>(A);
            T* d_A = dev_work + offset;
            gyCheckError(gyMemcpy(d_A, A, width * sizeof(T), gyMemcpyHostToDevice), "Failed to copy to device");
            return d_A;
        }

        template<class T>
        T* wrapReferDevice(
            T* dev_work, T* C, int64_t offset, size_t width) {
            if (dev_work == nullptr) return C;
            T* d_C = dev_work + offset;
            return d_C;
        }

        template<class T>
        void wrapCopyBackFromDevice(
            T* C, T* d_C, size_t width) {
            if (C == d_C) return;  //host to host , or device to device(unified mem)
            gyCheckError(gyMemcpy(C, d_C, width * sizeof(T), gyMemcpyDeviceToHost), "Failed to copy back from device");
        }

        template <class T>
        void wrapGemm_t(int m, int n, int k,
            const T* A, const T* B, T* C, int strideC, T alpha, T beta, T* dev_work = nullptr)
        {
            T* d_A = wrapCopyToDevice(dev_work, A, 0, m * k);
            T* d_B = wrapCopyToDevice(dev_work, B, m * k, n * k);
            T* d_C = wrapReferDevice(dev_work, C, (m+n) * k, strideC * n);

            if constexpr (std::is_same_v<T, double>) {
                DGEMM_t( m, n, k, d_A, d_B, d_C, strideC, alpha, beta);
            } else if constexpr (std::is_same_v<T, float>) {
                SGEMM_t( m, n, k, d_A, d_B, d_C, strideC, alpha, beta);
            } else {
                static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>, "Unsupported type for wrapGemm_t");
            }

            wrapCopyBackFromDevice(C, d_C, strideC * n);
        }

        template <class T>
        void wrapGemm_bt(int m, int n, int k,
            const T* A, const T* B, T* C, int strideC, T alpha, T beta, T* dev_work = nullptr)
        {
            T* d_A = wrapCopyToDevice(dev_work, A, 0, m * k);
            T* d_B = wrapCopyToDevice(dev_work, B, m * k, n * k);
            T* d_C = wrapReferDevice(dev_work, C, (m + n) * k, strideC * n);

            if constexpr (std::is_same_v<T, double>) {
                DGEMM_bt(m, n, k, d_A, d_B, d_C, strideC, alpha, beta);
            } else if constexpr (std::is_same_v<T, float>) {
                SGEMM_bt(m, n, k, d_A, d_B, d_C, strideC, alpha, beta);
            } else {
                static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>, "Unsupported type for wrapGemm_bt");
            }
            wrapCopyBackFromDevice(C, d_C, strideC* n);
        }

        template <class T>
        void wrapGemm_n(int m, int n, int k,
            const T* A, const T* B, T* C, int strideC, T alpha, T beta, T* dev_work = nullptr)
        {
            T* d_A = wrapCopyToDevice(dev_work, A, 0, m * k);
            T* d_B = wrapCopyToDevice(dev_work, B, m * k, n * k);
            T* d_C = wrapReferDevice(dev_work, C, (m + n) * k, strideC * n);

            if constexpr (std::is_same_v<T, double>) {
                DGEMM_n(m, n, k, d_A, d_B, d_C, strideC, alpha, beta);
            } else if constexpr (std::is_same_v<T, float>) {
                SGEMM_n(m, n, k, d_A, d_B, d_C, strideC, alpha, beta);
            } else {
                static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>, "Unsupported type for wrapGemm_n");
            }
            wrapCopyBackFromDevice(C, d_C, strideC* n);
        }
    }//blas//

    

    template <class T>
    T* AllocWorkOnMemory(size_t width) {
        T* d_work = nullptr;
#if defined(GY_WITH_CUDA) || defined(GY_WITH_HIP)        
        gyCheckError(gyMalloc((void**)&d_work, width * sizeof(T)), "Failed to allocate device memory");
#endif
        return d_work;
    }


}//gy//




/*
* DGEMM wrapper
* C = alpha*(A^t*B) + beta*C
* matrix A is KxM grids (A[iy + K * ix] where ix in [0,M), iy in [0,K)
* matrix B is KxN grids (A[iy + K * ix] where ix in [0,N), iy in [0,K)
*/
inline
void blas_DGEMM_t(int M, int N, int K, const double* A, const double* B, double* C, int strideC, double alpha, double beta) {
    gy::blas::DGEMM_t(M, N, K, A, B, C, strideC, alpha, beta);
    //gyDeviceSynchronize();
}
#if 0
inline
void blas_DGEMM_t_ld(int M, int N, int K, double* A, int lda, double* B, int ldb, double* C, int ldc, double alpha, double beta) {
    gy::blas::blas_DGEMM_t_ld(CblasColMajor, CblasTrans, CblasNoTrans, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    //cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, M, B, K, beta, C, strideC);
    //cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, alpha, A, K, B, K, beta, C, strideC);

}
#endif



#ifdef USE_DGEMMT
inline
void blas_DGEMMT_t_up(int N, int K, double* A, double* B, double* C, int strideC, double alpha, double beta) {
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
    gy::blas::DGEMM_bt(M, N, K, A, B, C, strideC, alpha, beta);
    //gyDeviceSynchronize();
}




inline
void blas_DGEMM_n(int M, int N, int K, const double* A, const double* B, double* C, int strideC, double alpha, double beta) {
    gy::blas::DGEMM_n(M, N, K, A, B, C, strideC, alpha, beta);
    //gyDeviceSynchronize();
}



inline
void blas_DGEMM(char TransA, char TransB, int M, int N, int K, double alpha, const double* A, int lda, const double* B, int ldb, double beta, double* C, int ldc) {
#ifdef GY_WITH_CUDA
    auto&& handle = gy::blas::SingletonHandle::Get();
    cublasDgemm(handle, 
        TransA == 'T' ? CUBLAS_OP_T : CUBLAS_OP_N,
        TransB == 'T' ? CUBLAS_OP_T : CUBLAS_OP_N,
        M, N, K,
        &alpha,
        A, lda,
        B, ldb,
        &beta,
        C, ldc);
    //gyDeviceSynchronize();
#elif defined(GY_WITH_HIP)
    auto&& handle = gy::blas::SingletonHandle::Get();
    rocblas_dgemm(handle, 
        TransA == 'T' ? rocblas_operation_transpose : rocblas_operation_none,
        TransB == 'T' ? rocblas_operation_transpose : rocblas_operation_none,
        M, N, K,
        &alpha,
        A, lda,
        B, ldb,
        &beta,
        C, ldc);
    //gyDeviceSynchronize();
#else
    cblas_dgemm(CblasColMajor,
        TransA == 'T' ? CblasTrans : CblasNoTrans,
        TransB == 'T' ? CblasTrans : CblasNoTrans,
        M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}


inline
void blas_ZGEMM(char TransA, char TransB, int M, int N, int K, const void* alpha, const void* A, int lda,
    const void* B, int ldb, const void* beta, void* C, int ldc) {
#ifdef GY_WITH_CUDA
    auto&& handle = gy::blas::SingletonHandle::Get();
    cublasZgemm(handle, 
        TransA == 'C' ? CUBLAS_OP_C : TransA == 'T' ? CUBLAS_OP_T : CUBLAS_OP_N,
        TransB == 'C' ? CUBLAS_OP_C : TransB == 'T' ? CUBLAS_OP_T : CUBLAS_OP_N,
        M, N, K,
        (const cuDoubleComplex*)alpha,
        (const cuDoubleComplex*)A, lda,
        (const cuDoubleComplex*)B, ldb,
        (const cuDoubleComplex*)beta,
        (cuDoubleComplex*)C, ldc);
    //gyDeviceSynchronize();
#elif defined(GY_WITH_HIP)
    auto&& handle = gy::blas::SingletonHandle::Get();
    rocblas_zgemm(handle, 
        TransA == 'C' ? rocblas_operation_conjugate_transpose : TransA == 'T' ? rocblas_operation_transpose : rocblas_operation_none,
        TransB == 'C' ? rocblas_operation_conjugate_transpose : TransB == 'T' ? rocblas_operation_transpose : rocblas_operation_none,
        M, N, K,
        (const rocblas_double_complex*)alpha,
        (const rocblas_double_complex*)A, lda,
        (const rocblas_double_complex*)B, ldb,
        (const rocblas_double_complex*)beta,
        (rocblas_double_complex*)C, ldc);
    //gyDeviceSynchronize();
#else
    cblas_zgemm(CblasColMajor,
        TransA == 'C' ? CblasConjTrans : TransA == 'T' ? CblasTrans : CblasNoTrans,
        TransB == 'C' ? CblasConjTrans : TransB == 'T' ? CblasTrans : CblasNoTrans,
        M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}

template<class MYCOMPLEX>
void blas_ZGEMM_t(int M, int N, int K, const MYCOMPLEX* A, const MYCOMPLEX* B, MYCOMPLEX* C, int strideC, MYCOMPLEX alpha, MYCOMPLEX beta) {
    blas_ZGEMM('C', 'N', M, N, K, &alpha, A, K, B, K, &beta, C, strideC);
}

/*
* DGEMV wrapper
* C = alpha*(A^t*V) + beta*C
* matrix A is KxM grids (A[iy + K * ix] where ix in [0,M), iy in [0,K)
* matrix V is K grids (A[iy] where iy in [0,K)
*/
inline
void blas_DGEMV_t(int M, int K, const double* A, const double* V, double* C, double alpha, double beta) {

    gy::blas::DGEMV_t(M, K, A, V, C, alpha, beta);
    //gyDeviceSynchronize();

}
#if 0
inline
void blas_ZGEMV_t(int M, int K, void* A, void* V, void* C, OneComplex alpha, OneComplex beta) {

    cblas_zgemv(CblasRowMajor, CblasNoTrans, M, K, &alpha, A, K, V, 1, &beta, C, 1);

}
#endif


/*
* DSYRK wrapper
* C = alpha*(A^t*A) + beta*C
* matrix A is KxM grids (A[iy + K * ix] where ix in [0,M), iy in [0,K)
*/
inline
void blas_DSYRK_t(char UPLO, int N, int K, const double* A, double* C, int strideC, double alpha, double beta) {
#ifdef GY_WITH_CUDA
    auto&& handle = gy::blas::SingletonHandle::Get();
    cublasDsyrk(handle,
        (UPLO == 'U') ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_T,
        N, K,
        &alpha,
        (const double*)A, K,
        &beta,
        (double*)C, strideC);
    //gyDeviceSynchronize();
#elif defined(GY_WITH_HIP)
    auto&& handle = gy::blas::SingletonHandle::Get();
    rocblas_dsyrk(handle,
        (UPLO == 'U') ? rocblas_fill_upper : rocblas_fill_lower,
        rocblas_operation_transpose,
        N, K,
        &alpha,
        (const double*)A, K,
        &beta,
        (double*)C, strideC);
    //gyDeviceSynchronize();
#else
    cblas_dsyrk(CblasColMajor,
        (UPLO == 'U') ? CblasUpper : CblasLower,
        CblasTrans, N, K, alpha, A, K, beta, C, strideC);
#endif
    

}
inline
void blas_DSYRK_t(int N, int K, const double* A, double* C, int strideC, double alpha, double beta) {
    blas_DSYRK_t('U', N, K, A, C, strideC, alpha, beta);
}



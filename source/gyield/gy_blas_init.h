#pragma once
#ifdef GY_WITH_CUDA
#include "gy_cublas.h"

#elif defined(GY_WITH_HIP)
#include "gy_rocblas.h"

#else  //for CPU

#if defined(__INTEL_LLVM_COMPILER ) || defined(__INTEL_COMPILER )
#include <complex>
#include <mkl.h>
#include <mkl_cblas.h>
//#include <mkl_lapack.h>
#elif defined(_NEC)
#include <complex>
#include <cblas.h>

#else
#define HAVE_LAPACK_CONFIG_H
#define LAPACK_COMPLEX_CPP
#include <complex>
//#define lapack_complex_float std::complex<float>
//#define lapack_complex_double std::complex<double>
#define LAPACK_GLOBAL_PATTERN_UC
#define __EMSCRIPTEN__
#include <cblas.h>
//#include <f77blas.h>
//#include <lapack.h>
//#define MKL_Complex16 std::complex<double>
#endif


#endif

namespace gy {

    namespace blas {

#if defined(GY_WITH_CUDA) || defined(GY_WITH_HIP) 

        class SingletonHandle {
            gyblasHandle_t m_handle = 0;
        public:
            SingletonHandle() {
                // cuBLASハンドルの作成
                gyCheckBlasError(gyblasCreate(&m_handle), "Failed to create cuBLAS handle");
                // tensor coreの有効化, これを書かないとFP32とFP64で差が出ない //
                gyCheckBlasError(gyblasEnableTensorCore(m_handle), "Failed to set Tensor Core math mode");

            }

            ~SingletonHandle() {
                if(m_handle){
                    gyblasDestroy(m_handle);
                }
            }

            static gyblasHandle_t Get() {
                static SingletonHandle blas_common_handle;
                return blas_common_handle.m_handle;
            }
        };

        
#endif


    }




}


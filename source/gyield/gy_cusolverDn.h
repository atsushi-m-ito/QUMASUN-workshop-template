#pragma once
#ifdef GY_WITH_CUDA

#include <cusolverDn.h>
#include <cuda_runtime.h>

using gysolverDnHandle_t = cusolverDnHandle_t;
using gysolverStatus_t = cusolverStatus_t;

inline gysolverStatus_t gysolverDnCreate(gysolverDnHandle_t* p_handle) {
    return cusolverDnCreate(p_handle);
}
inline gysolverStatus_t gysolverDnDestroy(gysolverDnHandle_t handle) {
    return cusolverDnDestroy(handle);
}


inline
void gyCheckSolverDnError(gysolverStatus_t status, const char* msg) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << msg << ": cusolverDn error" << std::endl;
        exit(EXIT_FAILURE);
    }
}


namespace gy {

    namespace lapack {


        class SingletonHandle {
            gysolverDnHandle_t m_handle = 0;
        public:
            SingletonHandle() {
                // cuBLASハンドルの作成
                gyCheckSolverDnError(gysolverDnCreate(&m_handle), "Failed to create cuBLAS handle");
                // tensor coreの有効化, これを書かないとFP32とFP64で差が出ない //
                //gyCheckBlasError(gyblasEnableTensorCore(m_handle), "Failed to set Tensor Core math mode");

            }

            ~SingletonHandle() {
                if(m_handle){
                    gysolverDnDestroy(m_handle);
                }
            }

            static gysolverDnHandle_t Get() {
                static SingletonHandle lapack_common_handle;
                return lapack_common_handle.m_handle;
            }
        };

        


    }




}
#endif

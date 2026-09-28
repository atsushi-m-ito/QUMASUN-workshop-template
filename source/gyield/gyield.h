#pragma once
#ifdef GY_WITH_CUDA
#include "gy_cuda.h"
#define __GPU_GLOBAL__  __global__
#define __GPU_DEVICE__  __device__
#define GY_WITH_CUDA_OR_HIP 1

#elif defined(GY_WITH_HIP)
#include "gy_hip.h"
#define __GPU_GLOBAL__  __global__
#define __GPU_DEVICE__  __device__

inline constexpr bool REDEIAN_MODE = true;
#define GY_WITH_CUDA_OR_HIP 1

#elif defined(GY_WITH_OMP)


inline int GY_MAX_THREADS_PER_BLOCK = 2;



#else

//dummy
#define __GPU_GLOBAL__     
#define __GPU_DEVICE__   

inline int GY_MAX_THREADS_PER_BLOCK = 1;

#define gyCheckError( a, b)     

namespace gy {
    inline
    void Synchronize(void) {
    }
}

#endif

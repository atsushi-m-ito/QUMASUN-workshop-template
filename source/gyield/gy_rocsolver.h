#pragma once
#ifdef GY_WITH_HIP
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>
#include "gy_blas_init.h"

//rocsolver handle is same as rocblas handle




namespace gy {

    namespace lapack {


        using SingletonHandle = gy::blas::SingletonHandle;
            
    }




}
#endif

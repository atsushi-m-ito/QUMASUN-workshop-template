#pragma once
#include "gyield.h"

#if defined(GY_WITH_CUDA) || defined(GY_WITH_HIP)

namespace gy {

#define GY_LAMBDA   [=]__device__


    /*
    * 同一ブロック内のスレッド間で和を取る関数(カーネルから呼ぶ)
    *
    */

    __device__ __forceinline__
    double summarize_in_block(double val, double* __restrict smem) {
        const int tid = threadIdx.x;
        const int num_threads = blockDim.x;


        smem[tid] = val;

        __syncthreads();
        for (int d = num_threads >> 1; d >= warpSize; d >>= 1) { //here, warpSize is kernel constant//
            if (tid < d) {
                smem[tid] += smem[tid + d];
            }
            __syncthreads();
        }

        //summarize in warp
        double sum = 0.0;
        if (tid < warpSize) {
            sum = smem[tid];
            for (int d = warpSize >> 1; d > 0; d >>= 1) {
#ifdef GY_WITH_CUDA
                double tmp2 = __shfl_down_sync(0xffffffff, sum, d);
#else
                double tmp2 = __shfl_down(sum, d);
#endif
                //if (tid < d) {
                sum += tmp2; //f(tid%WARP_SIZE >= d ) count += tmp2;
                //}
            }
        }
        return sum;
    }


    template <class I, class FUNC>
    __global__ void for_1d(I Nx, FUNC func) {
        const I ix = threadIdx.x + (I)blockDim.x * blockIdx.x;
        if (ix >= Nx) return;
        func(ix);
        
    }

    template <class I, class FUNC>
    void For(I Nx, FUNC func) {
        const int Nth = GY_MAX_THREADS_PER_BLOCK;
        for_1d <<<(Nx + Nth - 1) / Nth, Nth >>> (Nx, func);
    }

    template <class I, class FUNC>
    __global__ void for_1d_bundle(I Nx, FUNC func) {
        const I ix = threadIdx.x + blockDim.x * blockIdx.x;
        const int ib = blockIdx.y;
        if (ix >= Nx) return;
        func(ix, ib);

    }

    template <class I, class FUNC>
    void For_1d_bundle(I Nx, int num_bundle, FUNC func) {
        const int Nth = GY_MAX_THREADS_PER_BLOCK;
        dim3 cu_blocks;
        cu_blocks.x = (int)((Nx + Nth - 1) / Nth);
        cu_blocks.y = num_bundle;
        cu_blocks.z = 1;
        for_1d_bundle <<<cu_blocks, Nth >>> (Nx, func);
    }




    template <class I, class D, class FUNC>
    __global__ void for_1d_bundle_reduce(I i_end, D* __restrict results, D scale, FUNC func) {
        extern __shared__ double smem[];
        const int tid = threadIdx.x;
        //const int num_threads = blockDim.x;
        const int n = blockIdx.x;

        double l_sum = 0.0;
        for (int64_t i = tid; i < i_end; i += blockDim.x) {
            l_sum += func(i, n);;
        }

        double sum = summarize_in_block(l_sum, smem);
        if (tid == 0) {
            results[n] = sum * scale;
        }

    }


    template <class I, class D, class FUNC>
    void For_1d_bundle_reduce(const I Nx, int num_bundle, D* __restrict results, D scale, FUNC func) {

        const int Nth = GY_MAX_THREADS_PER_BLOCK;
        const int64_t sh_mem_size = sizeof(double) * Nth;
        for_1d_bundle_reduce <<<num_bundle, Nth, sh_mem_size >>> (Nx, results, scale, func);

    }

    template <class FUNC>
    __global__ void for_2d(int Nx, int Ny, FUNC func) {
        const int ix = threadIdx.x + blockDim.x * blockIdx.x;
        const int iy = threadIdx.y + blockDim.y * blockIdx.y;
        if (ix >= Nx || iy >= Ny) return;
        func(ix, iy);        
    }

    template <class FUNC>
    void For(int Nx, int Ny, FUNC func) {

        dim3 cu_threads;
#if 1
        cu_threads.x = GY_MAX_THREADS_PER_BLOCK;
#elif defined(GY_APU)
        cu_threads.x = 64;
#else
        cu_threads.x = 32;
#endif
        cu_threads.y = GY_MAX_THREADS_PER_BLOCK / cu_threads.x;
        cu_threads.z = 1;
        dim3 cu_blocks;
        cu_blocks.x = (Nx + cu_threads.x - 1) / cu_threads.x;
        cu_blocks.y = (Ny + cu_threads.y - 1) / cu_threads.y;
        cu_blocks.z = 1;
        for_2d <<<cu_blocks, cu_threads >>> (Nx, Ny, func);
    }


    template <class FUNC>
    __global__ void for_3d(int Nx, int Ny, int Nz, FUNC func) {
        const int ix = threadIdx.x + blockDim.x * blockIdx.x;
        const int iy = threadIdx.y + blockDim.y * blockIdx.y;
        const int iz = threadIdx.z + blockDim.z * blockIdx.z;
        if (ix >= Nx || iy >= Ny || iz >= Nz) return;
        func(ix, iy, iz);
        
    }

    template <class FUNC>
    void For(int Nx, int Ny, int Nz, FUNC func) {

        dim3 cu_threads;
#ifdef GY_APU
        cu_threads.x = 64;
#else
        cu_threads.x = 32;
#endif
        cu_threads.y = GY_MAX_THREADS_PER_BLOCK / cu_threads.x;
        cu_threads.z = 1;
        dim3 cu_blocks;
        cu_blocks.x = (Nx + cu_threads.x - 1) / cu_threads.x;
        cu_blocks.y = (Ny + cu_threads.y - 1) / cu_threads.y;
        cu_blocks.z = (Nz + cu_threads.z - 1) / cu_threads.z;
        
        for_3d <<<cu_blocks, cu_threads >>> (Nx, Ny, Nz, func);
    }

    template <class FUNC>
    __global__ void for_3d_bundle(int Nx, int Ny, int Nz, FUNC func) {
        const int tid = threadIdx.x;
        const int iy = blockIdx.x;
        const int iz = blockIdx.y;
        const int n = blockIdx.z;

        //const int i_end = Nx * Ny * Nz;
        for (int ix = tid; ix < Nx; ix += blockDim.x) {            
            func(ix, iy, iz, n);
        }

    }

    template <class FUNC>
    void For_3d_bundle(int Nx, int Ny, int Nz, int num_bundle, FUNC func) {

        dim3 cu_threads;
        cu_threads.x = 256; //GY_MAX_THREADS_PER_BLOCK
        cu_threads.y = 1;
        cu_threads.z = 1;
        dim3 cu_blocks;
        cu_blocks.x = Ny;
        cu_blocks.y = Nz;
        cu_blocks.z = num_bundle;

        for_3d_bundle <<<cu_blocks, cu_threads >>> (Nx, Ny, Nz, func);
    }


}

#else


namespace gy {

#define GY_LAMBDA   [&]


    template <class FUNC>
    void For(int Nx, FUNC func) {
        for (int ix = 0; ix < Nx; ++ix) {
            func(ix);
        }
    }

    template <class I, class FUNC>
    void For_1d_bundle(I Nx, int num_bundle, FUNC func) {
        for (int iy = 0; iy < num_bundle; ++iy) {
            for (I ix = 0; ix < Nx; ++ix) {
                func(ix, iy);
            }
        }
    }

    template <class I, class D, class FUNC>
    void For_1d_bundle_reduce(const I Nx, int num_bundle, D* __restrict results, D scale, FUNC func) {
        for (int n = 0; n < num_bundle; ++n) {
            double sum = 0.0;
            for (I i = 0; i < Nx; ++i) {
                sum += func(i, n);
            }
            results[n] = sum * scale;
        }
    }


    template <class FUNC>
    void For(int Nx, int Ny, FUNC func) {
        for (int iy = 0; iy < Ny; ++iy) {
            for (int ix = 0; ix < Nx; ++ix) {
                func(ix, iy);
            }
        }
    }

    template <class FUNC>
    void For(int Nx, int Ny, int Nz, FUNC func) {
        for (int iz = 0; iz < Nz; ++iz) {
            for (int iy = 0; iy < Ny; ++iy) {
                for (int ix = 0; ix < Nx; ++ix) {
                    func(ix, iy, iz);
                }
            }
        }
    }

    template <class FUNC>
    void For_3d_bundle(int Nx, int Ny, int Nz, int num_bundle, FUNC func) {
        for (int n = 0; n < num_bundle; ++n) {
            for (int iz = 0; iz < Nz; ++iz) {
                for (int iy = 0; iy < Ny; ++iy) {
                    for (int ix = 0; ix < Nx; ++ix) {
                        func(ix, iy, iz, n);
                    }
                }
            }
        }
    }
}



#endif

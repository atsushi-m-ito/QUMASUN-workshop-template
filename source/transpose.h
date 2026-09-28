#pragma once


template <typename T>
void transpose(size_t width1, size_t width2,
    const T* pSrc, int srcLineStride,
    T* pDst, int dstLineStride) noexcept
{
    if ((width1 == 0) || (width2 == 0) || (pSrc == nullptr) || (pDst == nullptr) || (srcLineStride < width1) || (dstLineStride < width2)) {
        return;
    }


    const size_t bw = 256;
    const size_t bh = 64;
    const size_t bx_end = (width1 / bw) * bw;
    const size_t by_end = (width2 / bh) * bh;
    for (size_t by = 0; by < by_end; by += bh) {
        for (size_t bx = 0; bx < bx_end; bx += bw) {
            auto offset_s = by * srcLineStride + bx;
            auto offset_d = bx * dstLineStride + by;

#pragma _NEC ivdep
#pragma ivdep
            for (size_t iy = 0; iy < bh; ++iy) {
#pragma vector
                for (size_t ix = 0; ix < bw; ++ix) {
                    auto src = pSrc[offset_s + iy * srcLineStride + ix];
                    pDst[offset_d + ix * dstLineStride + iy] = src;
                }
            }
        }
    }

    const size_t widthRemain = width1 - bx_end;
    const size_t heightRemain = width2 - by_end;
    {
        size_t by = by_end;
        for (size_t bx = 0; bx < bx_end; bx += bw) {
            auto offset_s = by * srcLineStride + bx;
            auto offset_d = bx * dstLineStride + by;

#pragma _NEC ivdep
#pragma ivdep
            for (size_t iy = 0; iy < heightRemain; ++iy) {
#pragma vector
                for (size_t ix = 0; ix < bw; ++ix) {
                    auto src = pSrc[offset_s + iy * srcLineStride + ix];
                    pDst[offset_d + ix * dstLineStride + iy] = src;
                }
            }
        }
    }
    {
        for (size_t by = 0; by < by_end; by += bh) {
            size_t bx = bx_end;
            auto offset_s = by * srcLineStride + bx;
            auto offset_d = bx * dstLineStride + by;

#pragma _NEC ivdep
#pragma ivdep
            for (size_t iy = 0; iy < bh; ++iy) {
#pragma vector
                for (size_t ix = 0; ix < widthRemain; ++ix) {
                    auto src = pSrc[offset_s + iy * srcLineStride + ix];
                    pDst[offset_d + ix * dstLineStride + iy] = src;
                }
            }
        }
    }
    {
        size_t by = by_end;
        size_t bx = bx_end;
        auto offset_s = by * srcLineStride + bx;
        auto offset_d = bx * dstLineStride + by;

#pragma _NEC ivdep
#pragma ivdep
        for (size_t iy = 0; iy < heightRemain; ++iy) {
            for (size_t ix = 0; ix < widthRemain; ++ix) {
                auto src = pSrc[offset_s + iy * srcLineStride + ix];
                pDst[offset_d + ix * dstLineStride + iy] = src;
            }
        }
    }
}


#ifdef GY_WITH_CUDA_OR_HIP
/*
* 3次元グリッドの並びを x,y,z から y,z,xへ循環置換させる
* y,z,x to z,x,y や z,x,y to x,y,z にも使える
*/
namespace gy {

    template<class T>
    __global__ void transpose3_gpu(int local_size_x, int local_size_y, int local_size_z, T* __restrict dest, const T* __restrict src) {
        const int64_t x0 = blockDim.x * blockIdx.x;
        const int64_t y0 = blockDim.y * blockIdx.y;
        const int TILE = 32;         //転置する最小単位のタイルは32*32とする。これ以上だと、shared memooryの上限サイズ(64KB)を超える可能性がある(T次第)
        const int x = threadIdx.x;   //32幅が前提
        const int y = threadIdx.y;   //[0-()]が前提
        const int64_t ioz = blockIdx.z;   //[0-()]が前提
        __shared__ T smem[TILE * TILE];

        //タイルへsrcから読み込み//
        for (int w = y; w < TILE && w < local_size_y - y0 ; w += blockDim.y) {
            const int ix = x + x0;
            const int iy = w + y0;
            const int sx = (x + w) % TILE; //yが進むごとにxを1グリッドずらすことで、次に読み込むときのキャッシュのバンクコンフリクトを防ぐ//
            if (ix < local_size_x) {
                smem[sx + TILE * w] = src[ix + local_size_x * (iy + local_size_y * ioz)];
            }
        }
        __syncthreads();

        //タイルからdestへ書き込み//
        for (int w = y; w < TILE; w += blockDim.y) {
            const int oy = x + y0;
            const int ox = w + x0;
            const int sy = x;
            const int sx = (w + x) % TILE; //yが進むごとにxを1グリッドずらすことで、次に読み込むときのキャッシュのバンクコンフリクトを防ぐ//

            const T dat = smem[sx + TILE * sy];

            if ((oy < local_size_y) && (ox < local_size_x)) {
                dest[oy + local_size_y * (ioz + local_size_z * ox)] = dat;
            }
        }

    }
    


    template<class T>
    __global__ void transpose3_back_gpu(int local_size_x, int local_size_y, int local_size_z, T* __restrict dest, const T* __restrict src) {
        const int64_t x0 = blockDim.x * blockIdx.x;
        const int64_t z0 = blockDim.y * blockIdx.y;
        const int TILE = 32;         //転置する最小単位のタイルは32*32とする。これ以上だと、shared memoryの上限サイズ(64KB)を超える可能性がある(T次第)
        const int x = threadIdx.x;   //32幅が前提
        const int z = threadIdx.y;   //[0-()]が前提
        const int64_t ioy = blockIdx.z;   //[0-()]が前提
        __shared__ T smem[TILE * TILE];

        //タイルへsrcから読み込み//
        for (int w = z; w < TILE && w < local_size_z - z0; w += blockDim.y) {
            const int ix = x + x0;
            const int iz = w + z0;
            const int sx = (x + w) % TILE; //yが進むごとにxを1グリッドずらすことで、次に読み込むときのキャッシュのバンクコンフリクトを防ぐ//
            if (ix < local_size_x) {
                smem[sx + TILE * w] = src[ix + local_size_x * (ioy + local_size_y * iz)];
            }
        }
        __syncthreads();

        //タイルからdestへ書き込み//
        for (int w = z; w < TILE; w += blockDim.y) {
            const int oz = x + z0;
            const int ox = w + x0;
            const int sz = x;
            const int sx = (w + x) % TILE; //yが進むごとにxを1グリッドずらすことで、次に読み込むときのキャッシュのバンクコンフリクトを防ぐ//

            const T dat = smem[sx + TILE * sz];

            if ((oz < local_size_z) && (ox < local_size_x)) {
                dest[oz + local_size_z * (ox + local_size_x * ioy)] = dat;
            }
        }

    }
}

#if 1
template<class T>
void Transpose3(int local_size_x, int local_size_y, int local_size_z, T* __restrict dest, const T* __restrict src)
{

    dim3 cu_threads;
    cu_threads.x = 32;
    cu_threads.y = 8;
    cu_threads.z = 1;
    dim3 cu_blocks;
    cu_blocks.x = (int)((local_size_x + cu_threads.x - 1) / cu_threads.x);
    cu_blocks.y = (int)((local_size_y + cu_threads.y - 1) / cu_threads.y);
    cu_blocks.z = local_size_z;
    gy::transpose3_gpu <<< cu_blocks, cu_threads >>> (local_size_x, local_size_y, local_size_z, dest, src);
    gy::Synchronize();
}


template<class T>
void Transpose3_back(int local_size_x, int local_size_y, int local_size_z, T* dest, T* src)
{

    dim3 cu_threads;
    cu_threads.x = 32;
    cu_threads.y = 8;
    cu_threads.z = 1;
    dim3 cu_blocks;
    cu_blocks.x = (int)((local_size_x + cu_threads.x - 1) / cu_threads.x);
    cu_blocks.y = (int)((local_size_z + cu_threads.y - 1) / cu_threads.y);
    cu_blocks.z = local_size_y;
    gy::transpose3_back_gpu <<< cu_blocks, cu_threads >>> (local_size_x, local_size_y, local_size_z, dest, src);
    gy::Synchronize();
}

#else
/*
* 3次元グリッドの並びを x,y,z から y,z,xへ循環置換させる
* y,z,x to z,x,y や z,x,y to x,y,z にも使える
*/
template<class T>
void Transpose3(int local_size_x, int local_size_y, int local_size_z, T* __restrict dest, const T* __restrict src)
{
    gy::For(local_size_x, local_size_y, local_size_z, GY_LAMBDA(int ix, int iy, int iz){

        if ((iz >= local_size_z) || (iy >= local_size_y) || (ix >= local_size_x)) return;
        dest[iy + local_size_y * (iz + local_size_z * ix)] = src[ix + local_size_x * (iy + local_size_y * iz)];

    });
    gy::Synchronize();
}

/*
* 3次元グリッドの並びを x,y,z から z,x,y へ循環置換させる
* z,x,y to y,z,x や y,z,x to x,y,z にも使える
*/
template<class T>
void Transpose3_back(int local_size_x, int local_size_y, int local_size_z, T* dest, T* src)
{

    gy::For(local_size_x, local_size_y, local_size_z, GY_LAMBDA(int ix, int iy, int iz){

        if ((iz >= local_size_z) || (iy >= local_size_y) || (ix >= local_size_x)) return;
        dest[iz + local_size_z * (ix + local_size_x * iy)] = src[ix + local_size_x * (iy + local_size_y * iz)];
    });
    gy::Synchronize();
}

#endif

#else
/*
* 3次元グリッドの並びを x,y,z から y,z,xへ循環置換させる
* y,z,x to z,x,y や z,x,y to x,y,z にも使える
*/
template<class T>
void Transpose3(int local_size_x, int local_size_y, int local_size_z, T* __restrict dest, const T* __restrict src)
{

    for (int iz = 0; iz < local_size_z; ++iz) {
        for (int iy = 0; iy < local_size_y; ++iy) {
            for (int ix = 0; ix < local_size_x; ++ix) {
                dest[iy + local_size_y * (iz + local_size_z * ix)] = src[ix + local_size_x * (iy + local_size_y * iz)];
            }
        }
    }
}

#if 1
/*
* 3次元グリッドの並びを x,y,z から z,x,y へ循環置換させる
* z,x,y to y,z,x や y,z,x to x,y,z にも使える
*/

template<class T>
void Transpose3_back(int local_size_x, int local_size_y, int local_size_z, T* dest, T* src)
{

    for (int iz = 0; iz < local_size_z; ++iz) {
        for (int iy = 0; iy < local_size_y; ++iy) {
            for (int ix = 0; ix < local_size_x; ++ix) {
                dest[iz + local_size_z * (ix + local_size_x * iy)] = src[ix + local_size_x * (iy + local_size_y * iz)];
            }
        }
    }
}
#else
/*
* 3次元グリッドの並びを y,z,x から x,y,z へ循環置換させる
* z,x,y to y,z,x や x,y,z to z,x,y にも使える
*/
template<class T>
void Transpose3_back(int local_size_y, int local_size_z, int local_size_x, T* dest, T* src)
{

    for (int iz = 0; iz < local_size_z; ++iz) {
        for (int iy = 0; iy < local_size_y; ++iy) {
            for (int ix = 0; ix < local_size_x; ++ix) {
                dest[ix + local_size_x * (iy + local_size_y * iz)] = src[iy + local_size_y * (iz + local_size_z * ix)];
            }
        }
    }
}
#endif

#endif

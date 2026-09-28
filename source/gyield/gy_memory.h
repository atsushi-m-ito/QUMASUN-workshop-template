#pragma once

#include "gyield.h"
#include <cstring>

#ifdef GY_WITH_CUDA
#ifndef GY_ALIGNMENT
#define GY_ALIGNMENT   (256)
#endif


#elif defined(GY_WITH_HIP)
#ifndef GY_ALIGNMENT
#define GY_ALIGNMENT   (128)
#endif


#else
#ifndef GY_ALIGNMENT
//for SIMD AVD512
#define GY_ALIGNMENT   (64)
#endif

#ifdef _WIN32
#include <malloc.h>
#endif
#endif

#include <memory>

namespace gy {

    template<class T>
    size_t AlignedBytesize(size_t N) {
        size_t size_in_bytes = N * sizeof(T);
        return ((size_in_bytes + GY_ALIGNMENT - 1) / GY_ALIGNMENT) * GY_ALIGNMENT;
    }

    template<class T>
    size_t AlignedLeadingDimension(size_t N) {
        size_t padded_bytes = AlignedBytesize<T>(N);
        return padded_bytes / sizeof(T);
    }

    template<class T>
    T* AlignedAlloc(size_t N) {
#ifdef GY_WITH_CUDA
        T* buf;
/*#ifdef GY_APU
* cudaMallocManaged should be used also even if native unifie dmemory is supported on GH200.
        gyError_t err = cudaMalloc(&buf, sizeof(T) * N);
        if (err != gySuccess) {
            printf("ERROR: cudaMalloc: %p, %zd\n", buf, sizeof(T) * N);
            gyCheckError(err, "ERROR: cudaMalloc: ");
        }
#else
*/
        gyError_t err = cudaMallocManaged(&buf, sizeof(T) * N);
        if (err != gySuccess) {
            printf("ERROR: cudaMallocManaged: %p, %zd\n", buf, sizeof(T) * N);
            gyCheckError(err, "ERROR: cudaMallocManaged: ");
        }
//#endif
        return buf;
#elif defined (GY_WITH_HIP)
        T* buf;
#ifdef GY_APU
        gyError_t err = hipMalloc(&buf, sizeof(T) * N);
        if (err != gySuccess) {
            printf("ERROR: hipMalloc: %p, %zd\n", buf, sizeof(T) * N);
            gyCheckError(err, "ERROR: hipMalloc: ");
        }
#else
        gyError_t err = hipMallocManaged(&buf, sizeof(T) * N);
        if (err != gySuccess) {
            printf("ERROR: hipMallocManaged: %p, %zd\n", buf, sizeof(T) * N);
            gyCheckError(err, "ERROR: hipMallocManaged: ");
        }
#endif
        return buf;
#else  //CPU
#ifdef _WIN32
        return (T*)_aligned_malloc(AlignedBytesize<T>(N), GY_ALIGNMENT);
#else
        return (T*)(std::aligned_alloc(GY_ALIGNMENT, AlignedBytesize<T>(N)));
#endif
#endif
    }

    template<class T>
    void AlignedFree(T* ptr) {
#ifdef GY_WITH_CUDA
        cudaFree(ptr);
#elif defined(GY_WITH_HIP)
        hipFree(ptr);
#else
#ifdef _WIN32
        ::_aligned_free(ptr);
#else
        std::free(ptr);
#endif
#endif
        ptr = nullptr;
    }

    template <class T>
    struct Deleter {
    public:
        using U = std::remove_extent_t<T>;
        void operator()(U* ptr) {
            AlignedFree<U>(ptr);
        }
    };


    template <class T>
    using unique_aligned_ptr = std::unique_ptr < T, Deleter<T> >;

    template<class T>
    std::enable_if_t<std::is_array_v<T>&& std::extent_v<T> == 0, unique_aligned_ptr<T>>
        make_unique_aligned(size_t N) {
        using U = std::remove_extent_t<T>;
        return unique_aligned_ptr<T>(AlignedAlloc<U>(N));
    }



    template<class T>
    void CopyMemory(T* dest, const T* src, size_t byte_size) {
#ifdef GY_WITH_CUDA_OR_HIP
        gyMemcpy(dest, src, byte_size, gyMemcpyDeviceToDevice);
#else
        std::memcpy(dest, src, byte_size);
#endif
    }

    template<class T>
    void CopyMemoryToHost(T* dest, const T* src, size_t byte_size) {
#ifdef GY_WITH_CUDA_OR_HIP
        gyMemcpy(dest, src, byte_size, gyMemcpyDeviceToHost);
#else
        std::memcpy(dest, src, byte_size);
#endif
    }

    template<class T>
    void SetMemory(T* dest, int val, size_t byte_size) {
#ifdef GY_WITH_CUDA_OR_HIP
        gyMemset(dest, val, byte_size);
#else
        std::memset(dest, val, byte_size);
#endif
    }

    template<class T>
    void ZeroClear(T* dest, size_t size) {
        SetMemory(dest, 0, sizeof(T)*size);
    }

}



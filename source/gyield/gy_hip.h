#pragma once 
#include <hip/hip_runtime.h>
#include <iostream>

inline int GY_MAX_THREADS_PER_BLOCK = 1024;
inline int GY_WARP_SIZE = 64;

using gyError_t = hipError_t;
constexpr gyError_t gySuccess = hipSuccess;

using gyDeviceProp = hipDeviceProp_t;

inline
gyError_t gyGetDeviceProperties(gyDeviceProp *prop, int deviceId)
{
    return hipGetDeviceProperties(prop, deviceId);
}

inline gyError_t gyGetDeviceCount(int* count) {
    return hipGetDeviceCount(count);
}

inline gyError_t gySetDevice(int device) {
    return hipSetDevice(device);
}

inline
gyError_t gyMalloc(void** ptr, size_t size)
{
    return hipMalloc(ptr, size);
}

inline
gyError_t gyMallocManaged(void** ptr, size_t size)
{
    return hipMallocManaged(ptr, size);
}

inline
gyError_t gyDeviceSynchronize(void)
{
    return hipDeviceSynchronize();
}

namespace gy {
    inline
    void Synchronize(void) {
        auto a = gyDeviceSynchronize();
    }
}

using gyMemcpyKind = hipMemcpyKind;
inline const gyMemcpyKind gyMemcpyHostToHost = hipMemcpyHostToHost;
inline const gyMemcpyKind gyMemcpyHostToDevice = hipMemcpyHostToDevice;
inline const gyMemcpyKind gyMemcpyDeviceToHost = hipMemcpyDeviceToHost;
inline const gyMemcpyKind gyMemcpyDeviceToDevice = hipMemcpyDeviceToDevice;
inline const gyMemcpyKind gyMemcpyDefault = hipMemcpyDefault;

inline
gyError_t gyMemcpy(void* dst, const void* src, size_t count, gyMemcpyKind kind)
{
    return hipMemcpy( dst, src, count, kind);
}

inline
gyError_t gyMemcpyAsync(void* dst, const void* src, size_t count, gyMemcpyKind kind)
{
    return hipMemcpyAsync(dst, src, count, kind);
}

inline
gyError_t gyMemset(void* devPtr, int value, size_t count){
    return hipMemset(devPtr, value, count);    
}

inline
gyError_t gyFree(void* p){
    return hipFree(p);
}



inline const hipDeviceAttribute_t gyDevAttrWarpSize = hipDeviceAttributeWarpSize;

inline
gyError_t gyDeviceGetAttribute(int* value, hipDeviceAttribute_t attr, int  device) {
    return hipDeviceGetAttribute(value, attr, device);
}

gyError_t gyGetLastError() {
    return hipGetLastError();
}


void gyCheckError(gyError_t err, const char* msg) {
    if (err != hipSuccess) {
        std::cerr << msg << ":(" << err << ") " << hipGetErrorString(err) << "(" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

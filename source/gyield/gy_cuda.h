#pragma once 
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

inline int GY_MAX_THREADS_PER_BLOCK = 1024;
inline int GY_WARP_SIZE = 32;


using gyError_t = cudaError_t;
using gyDeviceProp = cudaDeviceProp;
constexpr gyError_t gySuccess = cudaSuccess;

inline 
cudaError_t gyGetDeviceProperties(gyDeviceProp *prop, int deviceId)
{
    return cudaGetDeviceProperties(prop, deviceId);
}

inline cudaError_t gyGetDeviceCount(int* count) {
    return cudaGetDeviceCount(count);
}

inline cudaError_t gySetDevice(int device) {
    return cudaSetDevice(device);
}

inline
gyError_t gyMalloc(void** ptr, size_t size)
{
    return cudaMalloc(ptr, size);
}

inline
gyError_t gyMallocManaged(void** ptr, size_t size)
{
    return cudaMallocManaged(ptr, size);
}

inline
gyError_t gyDeviceSynchronize(void)
{
    return cudaDeviceSynchronize();
}

namespace gy {
    inline
    void Synchronize(void) {
        gyDeviceSynchronize();
    }
}


using gyMemcpyKind = cudaMemcpyKind;
inline const gyMemcpyKind gyMemcpyHostToHost = cudaMemcpyHostToHost;
inline const gyMemcpyKind gyMemcpyHostToDevice = cudaMemcpyHostToDevice;
inline const gyMemcpyKind gyMemcpyDeviceToHost = cudaMemcpyDeviceToHost;
inline const gyMemcpyKind gyMemcpyDeviceToDevice = cudaMemcpyDeviceToDevice;
inline const gyMemcpyKind gyMemcpyDefault = cudaMemcpyDefault;

inline
gyError_t gyMemcpy(void* dst, const void* src, size_t count, gyMemcpyKind kind)
{
    return cudaMemcpy(dst, src, count, kind);
}

inline
gyError_t gyMemcpyAsync(void* dst, const void* src, size_t count, gyMemcpyKind kind)
{
    return cudaMemcpyAsync(dst, src, count, kind);
}

inline
gyError_t gyMemset(void* devPtr, int value, size_t count){
    return cudaMemset(devPtr, value, count);    
}

inline
gyError_t gyFree(void* p){
    return cudaFree(p);
}


inline const cudaDeviceAttr gyDevAttrWarpSize = cudaDevAttrWarpSize;

inline
gyError_t gyDeviceGetAttribute(int* value, cudaDeviceAttr attr, int  device){
    return cudaDeviceGetAttribute(value, attr, device);
}

gyError_t gyGetLastError() {
    return cudaGetLastError();
}

void gyCheckError(gyError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ":(" << err << ") " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}


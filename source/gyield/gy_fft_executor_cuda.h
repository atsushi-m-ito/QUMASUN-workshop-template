#pragma once
#include <cstring>
#include <utility>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

#ifndef FFTW_ESTIMATE
#define FFTW_ESTIMATE (1U << 6)
#endif

template <typename Z, bool SYNC>
class FFTW_Executor_cuda {
private:
    int Nx = 0;
    int Ny = 0;
    int Nz = 0;    
    cufftDoubleComplex* buffer = nullptr;
    cufftHandle plan;
public:
    ~FFTW_Executor_cuda() {
        if (buffer) {
            cufftDestroy(plan);
            cudaFree(buffer);
        }
    }

    void Initialize(int grid_x, int grid_y, int grid_z, int fftw_flags) {
        Nx = grid_x;
        Ny = grid_y;
        Nz = grid_z;
        cudaMallocManaged(&buffer, sizeof(cufftDoubleComplex) * Nx * Ny * Nz);
        cufftPlan3d(&plan, Nz, Ny, Nx, CUFFT_Z2Z);
    }

    Z* GetBuffer() { return (Z*)buffer; };

    Z* ForwardExecute(Z* in, Z* out) {
        auto res = cufftExecZ2Z(plan, 
            (in == nullptr) ? buffer : (cufftDoubleComplex*)in,
            (out == nullptr) ? buffer : (cufftDoubleComplex*)out,
            CUFFT_FORWARD);
        if constexpr (SYNC) cudaDeviceSynchronize();
        return (out == nullptr) ? (Z*)buffer : out;
    }

    Z* BackwardExecute(Z* in, Z* out) {
        auto res = cufftExecZ2Z(plan,
            (in == nullptr) ? buffer : (cufftDoubleComplex*)in,
            (out == nullptr) ? buffer : (cufftDoubleComplex*)out,
            CUFFT_INVERSE);
        if constexpr (SYNC) cudaDeviceSynchronize();
        return (out == nullptr) ? (Z*)buffer : out;
    }


    Z* ForwardDirect(Z* inout) {
        auto res = cufftExecZ2Z(plan, (cufftDoubleComplex*)inout, (cufftDoubleComplex*)inout, CUFFT_FORWARD);
        if constexpr (SYNC) cudaDeviceSynchronize();
        return (Z*)inout;
    }


    Z* BackwardDirect(Z* inout) {
        auto res = cufftExecZ2Z(plan, (cufftDoubleComplex*)inout, (cufftDoubleComplex*)inout, CUFFT_INVERSE);
        if constexpr (SYNC) cudaDeviceSynchronize();
        return (Z*)inout;
    }

};


template <typename Z, bool SYNC>
class FFTW_Executor1D_cuda {
private:
    int Nx = 0;
    cufftDoubleComplex* buffer = nullptr;
    cufftHandle plan;
public:
    ~FFTW_Executor1D_cuda() {
        if (buffer) {
            cufftDestroy(plan);
            cudaFree(buffer);
        }
    }

    void Initialize(int grid_x, int fftw_flags) {
        Nx = grid_x;
        cudaMallocManaged(&buffer, sizeof(cufftDoubleComplex) * Nx );
        cufftPlan1d(&plan, Nx, CUFFT_Z2Z, 1);
    }

    Z* GetBuffer() { return (Z*)buffer; };

    Z* ForwardExecute(Z* in, Z* out) {
        auto res = cufftExecZ2Z(plan,
            (in == nullptr) ? buffer : (cufftDoubleComplex*)in,
            (out == nullptr) ? buffer : (cufftDoubleComplex*)out,
            CUFFT_FORWARD);
        if constexpr (SYNC) cudaDeviceSynchronize();
        return (out == nullptr) ? (Z*)buffer : out;
    }

    Z* BackwardExecute(Z* in, Z* out) {
        auto res = cufftExecZ2Z(plan,
            (in == nullptr) ? buffer : (cufftDoubleComplex*)in,
            (out == nullptr) ? buffer : (cufftDoubleComplex*)out,
            CUFFT_INVERSE);
        if constexpr (SYNC) cudaDeviceSynchronize();
        return (out == nullptr) ? (Z*)buffer : out;
    }


    Z* ForwardDirect(Z* inout) {
        auto res = cufftExecZ2Z(plan, (cufftDoubleComplex*)inout, (cufftDoubleComplex*)inout, CUFFT_FORWARD);
        if constexpr (SYNC) cudaDeviceSynchronize();
        return (Z*)inout;
    }


    Z* BackwardDirect(Z* inout) {
        auto res = cufftExecZ2Z(plan, (cufftDoubleComplex*)inout, (cufftDoubleComplex*)inout, CUFFT_INVERSE);
        if constexpr (SYNC) cudaDeviceSynchronize();
        return (Z*)inout;
    }

};


template <typename Z, bool SYNC>
class FFTW_ExecutorMany1D_cuda {
private:
    int Nx = 0;
    int Nbundle = 0;
    //cufftDoubleComplex* buffer = nullptr;
    cufftHandle plan;
public:
    ~FFTW_ExecutorMany1D_cuda() {
        if (Nx > 0) {
            cufftDestroy(plan);
        }
    }

    //ld: leading dimension of each data, which may be useful for SIMD and GPU//
    void Initialize(int grid_x, int num_lines) {
        Nx = grid_x;
        Nbundle = num_lines;
        int ld = Nx;        
        cufftPlanMany(&plan, 1, &Nx, &Nx, 1, ld, &Nx, 1, ld, CUFFT_Z2Z, Nbundle);
    }


    Z* ForwardDirect(Z* inout) {
        auto res = cufftExecZ2Z(plan, (cufftDoubleComplex*)inout, (cufftDoubleComplex*)inout, CUFFT_FORWARD);
        if constexpr (SYNC) cudaDeviceSynchronize();
        return (Z*)inout;
    }


    Z* BackwardDirect(Z* inout) {
        auto res = cufftExecZ2Z(plan, (cufftDoubleComplex*)inout, (cufftDoubleComplex*)inout, CUFFT_INVERSE);
        if constexpr (SYNC) cudaDeviceSynchronize();
        return (Z*)inout;
    }


};



template <typename Z, bool SYNC>
class FFTW_ExecutorMany3D_cuda {
private:
    int Nxyz[3]{ 0,0,0 };
    int Nbundle = 0;
    //cufftDoubleComplex* buffer = nullptr;
    cufftHandle plan;
public:
    ~FFTW_ExecutorMany3D_cuda() {
        if (Nbundle > 0) {
            cufftDestroy(plan);
        }
    }

    //ld: leading dimension of each data, which may be useful for SIMD and GPU//
    void Initialize(int grid_x, int grid_y, int grid_z, int num_bundle_, Z* buffer = nullptr, int ld = 0) {
        if (Nbundle > 0) {
            cufftDestroy(plan);
            Nbundle = 0;
        }
        Nxyz[2] = grid_x;
        Nxyz[1] = grid_y;
        Nxyz[0] = grid_z;
        Nbundle = num_bundle_;
        if (ld == 0) {
            ld = Nxyz[0] * Nxyz[1] * Nxyz[2];
        }
        cufftPlanMany(&plan, 3, Nxyz, Nxyz, 1, ld, Nxyz, 1, ld, CUFFT_Z2Z, Nbundle);

    }

    int GetNumBundle() {
        return Nbundle;
    }


    Z* ForwardExecute(Z* in, Z* out) {
        auto res = cufftExecZ2Z(plan, (cufftDoubleComplex*)in, (cufftDoubleComplex*)out, CUFFT_FORWARD);
        if constexpr (SYNC) cudaDeviceSynchronize();
        return out;
    }

    Z* BackwardExecute(Z* in, Z* out) {
        auto res = cufftExecZ2Z(plan, (cufftDoubleComplex*)in, (cufftDoubleComplex*)out, CUFFT_INVERSE);
        if constexpr (SYNC) cudaDeviceSynchronize();
        return in;
    }


    Z* ForwardDirect(Z* inout) {
        auto res = cufftExecZ2Z(plan, (cufftDoubleComplex*)inout, (cufftDoubleComplex*)inout, CUFFT_FORWARD);
        if constexpr (SYNC) cudaDeviceSynchronize();
        return (Z*)inout;
    }


    Z* BackwardDirect(Z* inout) {
        auto res = cufftExecZ2Z(plan, (cufftDoubleComplex*)inout, (cufftDoubleComplex*)inout, CUFFT_INVERSE);
        if constexpr (SYNC) cudaDeviceSynchronize();
        return (Z*)inout;
    }


};

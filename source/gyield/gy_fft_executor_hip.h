/***********************************
* 
* FFT wrapper for hipFFT for AMD GPU
* 
* Pay attention:
* The in-buffer may be overwritten, even if fft is executed in out-of-place mode.
* This is official behavior.
* 
************************************/

#pragma once
#include <cstring>
#include <utility>

#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>

#ifndef FFTW_ESTIMATE
#define FFTW_ESTIMATE (1U << 6)
#endif

template <typename Z, bool SYNC>
class FFTW_Executor_hip {
private:
    int Nx = 0;
    int Ny = 0;
    int Nz = 0;    
    hipfftDoubleComplex* buffer = nullptr;
    hipfftHandle plan;
public:
    ~FFTW_Executor_hip() {
        if (buffer) {
            hipfftDestroy(plan);
            hipFree(buffer);
        }
    }

    void Initialize(int grid_x, int grid_y, int grid_z, int fftw_flags) {
        Nx = grid_x;
        Ny = grid_y;
        Nz = grid_z;
#ifdef GY_APU
        hipMalloc(&buffer, sizeof(hipfftDoubleComplex) * Nx * Ny * Nz);
#else
        hipMallocManaged(&buffer, sizeof(hipfftDoubleComplex) * Nx * Ny * Nz);
#endif
        hipfftPlan3d(&plan, Nz, Ny, Nx, HIPFFT_Z2Z);
    }

    Z* GetBuffer() { return (Z*)buffer; };
#if 0
    Z* ForwardExecute(Z* in, Z* out) {
        auto res = hipfftExecZ2Z(plan, 
            (in == nullptr) ? buffer : (hipfftDoubleComplex*)in,
            (out == nullptr) ? buffer : (hipfftDoubleComplex*)out,
            HIPFFT_FORWARD);
        if constexpr (SYNC) hipDeviceSynchronize();
        return (out == nullptr) ? (Z*)buffer : out;
    }

    Z* BackwardExecute(Z* in, Z* out) {
        auto res = hipfftExecZ2Z(plan,
            (in == nullptr) ? buffer : (hipfftDoubleComplex*)in,
            (out == nullptr) ? buffer : (hipfftDoubleComplex*)out,
            HIPFFT_BACKWARD);
        if constexpr (SYNC) hipDeviceSynchronize();
        return (out == nullptr) ? (Z*)buffer : out;
    }
#endif

    Z* ForwardDirect(Z* inout) {
        auto res = hipfftExecZ2Z(plan, (hipfftDoubleComplex*)inout, (hipfftDoubleComplex*)inout, HIPFFT_FORWARD);
        if constexpr (SYNC) hipDeviceSynchronize();
        return (Z*)inout;
    }


    Z* BackwardDirect(Z* inout) {
        auto res = hipfftExecZ2Z(plan, (hipfftDoubleComplex*)inout, (hipfftDoubleComplex*)inout, HIPFFT_BACKWARD);
        if constexpr (SYNC) hipDeviceSynchronize();
        return (Z*)inout;
    }

};


template <typename Z, bool SYNC>
class FFTW_Executor1D_hip {
private:
    int Nx = 0;
    hipfftDoubleComplex* buffer = nullptr;
    hipfftHandle plan;
public:
    ~FFTW_Executor1D_hip() {
        if (buffer) {
            hipfftDestroy(plan);
            hipFree(buffer);
        }
    }

    void Initialize(int grid_x, int fftw_flags) {
        Nx = grid_x;
#ifdef GY_APU
        auto err = hipMalloc(&buffer, sizeof(hipfftDoubleComplex) * Nx);
#else        
        auto err = hipMallocManaged(&buffer, sizeof(hipfftDoubleComplex) * Nx );
#endif
        hipfftPlan1d(&plan, Nx, HIPFFT_Z2Z, 1);
    }

    Z* GetBuffer() { return (Z*)buffer; };
#if 0
    Z* ForwardExecute(Z* in, Z* out) {
        auto res = hipfftExecZ2Z(plan,
            (in == nullptr) ? buffer : (hipfftDoubleComplex*)in,
            (out == nullptr) ? buffer : (hipfftDoubleComplex*)out,
            HIPFFT_FORWARD);
        if constexpr (SYNC) hipDeviceSynchronize();
        return (out == nullptr) ? (Z*)buffer : out;
    }

    Z* BackwardExecute(Z* in, Z* out) {
        auto res = hipfftExecZ2Z(plan,
            (in == nullptr) ? buffer : (hipfftDoubleComplex*)in,
            (out == nullptr) ? buffer : (hipfftDoubleComplex*)out,
            HIPFFT_BACKWARD);
        if constexpr (SYNC) hipDeviceSynchronize();
        return (out == nullptr) ? (Z*)buffer : out;
    }
#endif

    Z* ForwardDirect(Z* inout) {
        auto res = hipfftExecZ2Z(plan, (hipfftDoubleComplex*)inout, (hipfftDoubleComplex*)inout, HIPFFT_FORWARD);
        if constexpr (SYNC) hipDeviceSynchronize();
        return (Z*)inout;
    }


    Z* BackwardDirect(Z* inout) {
        auto res = hipfftExecZ2Z(plan, (hipfftDoubleComplex*)inout, (hipfftDoubleComplex*)inout, HIPFFT_BACKWARD);
        if constexpr (SYNC) hipDeviceSynchronize();
        return (Z*)inout;
    }

};


template <typename Z, bool SYNC>
class FFTW_ExecutorMany1D_hip {
private:
    int Nx = 0;
    int Nbundle = 0;
    //hipfftDoubleComplex* buffer = nullptr;
    hipfftHandle plan;
public:
    ~FFTW_ExecutorMany1D_hip() {
        hipfftDestroy(plan);
    }

    //ld: leading dimension of each data, which may be useful for SIMD and GPU//
    void Initialize(int grid_x, int num_lines) {
        Nx = grid_x;
        Nbundle = num_lines;
        int ld = Nx;
        hipfftPlanMany(&plan, 1, &Nx, &Nx, 1,ld, &Nx, 1, ld, HIPFFT_Z2Z, num_lines);
    }



    Z* ForwardDirect(Z* inout) {
        auto res = hipfftExecZ2Z(plan, (hipfftDoubleComplex*)inout, (hipfftDoubleComplex*)inout, HIPFFT_FORWARD);
        if constexpr (SYNC) hipDeviceSynchronize();
        return (Z*)inout;
    }


    Z* BackwardDirect(Z* inout) {
        auto res = hipfftExecZ2Z(plan, (hipfftDoubleComplex*)inout, (hipfftDoubleComplex*)inout, HIPFFT_BACKWARD);
        if constexpr (SYNC) hipDeviceSynchronize();
        return (Z*)inout;
    }


};


template <typename Z, bool SYNC>
class FFTW_ExecutorMany3D_hip {
private:
    int Nxyz[3]{ 0,0,0 };
    int Nbundle = 0;
    //hipfftDoubleComplex* buffer = nullptr;
    hipfftHandle plan;
public:
    ~FFTW_ExecutorMany3D_hip() {
        if (Nbundle > 0) {
            hipfftDestroy(plan);
        }
    }

    //ld: leading dimension of each data, which may be useful for SIMD and GPU//
    void Initialize(int grid_x, int grid_y, int grid_z, int num_bundle_, Z* buffer = nullptr, int ld = 0) {
        if (Nbundle > 0) {
            hipfftDestroy(plan);
            Nbundle = 0;
        }
        Nxyz[2] = grid_x;
        Nxyz[1] = grid_y;
        Nxyz[0] = grid_z;
        Nbundle = num_bundle_;
        if (ld == 0) {
            ld = Nxyz[0] * Nxyz[1] * Nxyz[2];
        }
        hipfftPlanMany(&plan, 3, Nxyz, Nxyz, 1, ld, Nxyz, 1, ld, HIPFFT_Z2Z, Nbundle);
    }

    int GetNumBundle() {
        return Nbundle;
    }

    Z* ForwardExecute(Z* in, Z* out) {
        auto res = hipfftExecZ2Z(plan, (hipfftDoubleComplex*)in, (hipfftDoubleComplex*)out, HIPFFT_FORWARD);
        if constexpr (SYNC) hipDeviceSynchronize();
        return out;
    }

    Z* BackwardExecute(Z* in, Z* out) {
        auto res = hipfftExecZ2Z(plan, (hipfftDoubleComplex*)in, (hipfftDoubleComplex*)out, HIPFFT_BACKWARD);
        if constexpr (SYNC) hipDeviceSynchronize();
        return in;
    }


    Z* ForwardDirect(Z* inout) {
        auto res = hipfftExecZ2Z(plan, (hipfftDoubleComplex*)inout, (hipfftDoubleComplex*)inout, HIPFFT_FORWARD);
        if constexpr (SYNC) hipDeviceSynchronize();
        return (Z*)inout;
    }


    Z* BackwardDirect(Z* inout) {
        auto res = hipfftExecZ2Z(plan, (hipfftDoubleComplex*)inout, (hipfftDoubleComplex*)inout, HIPFFT_BACKWARD);
        if constexpr (SYNC) hipDeviceSynchronize();
        return (Z*)inout;
    }


};

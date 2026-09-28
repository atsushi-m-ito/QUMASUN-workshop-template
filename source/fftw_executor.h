#pragma once
#ifdef GY_WITH_CUDA
#include "gyield/gy_fft_executor_cuda.h"
#include "soacomplex.h"
using FFTW_Executor = FFTW_Executor_cuda<OneComplex, true>;
using FFTW_Executor1D = FFTW_Executor1D_cuda<OneComplex, true>;
using FFTW_ExecutorMany1D = FFTW_ExecutorMany1D_cuda<OneComplex, true>;
using FFTW_ExecutorMany3D = FFTW_ExecutorMany3D_cuda<OneComplex, true>;
#elif defined(GY_WITH_HIP)
#include "gyield/gy_fft_executor_hip.h"
#include "soacomplex.h"
using FFTW_Executor = FFTW_Executor_hip<OneComplex, true>;
using FFTW_Executor1D = FFTW_Executor1D_hip<OneComplex, true>;
using FFTW_ExecutorMany1D = FFTW_ExecutorMany1D_hip<OneComplex, true>;
using FFTW_ExecutorMany3D = FFTW_ExecutorMany3D_hip<OneComplex, true>;
#else
//for CPU/////////////////////////////////////////////////
#include <cstring>
#include "soacomplex.h"

#ifdef _NEC
#include <aslfftw3.h>
#else
#include <fftw3.h>
#endif


class FFTW_Executor {
private:
    int Nx = 0;
    int Ny = 0;
    int Nz = 0;    
    fftw_complex* buffer = nullptr;
    fftw_plan plan_forward{ 0 };
    fftw_plan plan_backward{ 0 };
public:
    ~FFTW_Executor() {
        if (buffer) {
            fftw_destroy_plan(plan_forward);
            fftw_destroy_plan(plan_backward);
            fftw_free(buffer);
        }
    }

    void Initialize(int grid_x, int grid_y, int grid_z, int fftw_flags) {
        Nx = grid_x;
        Ny = grid_y;
        Nz = grid_z;
        buffer = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * Nx * Ny * Nz);
        plan_forward = fftw_plan_dft_3d(Nz, Ny, Nx, buffer, buffer, FFTW_FORWARD, fftw_flags);
        plan_backward = fftw_plan_dft_3d(Nz, Ny, Nx, buffer, buffer, FFTW_BACKWARD, fftw_flags);
    }

    OneComplex* GetBuffer() { return (OneComplex*)buffer; };
#if 0
    OneComplex* ForwardExecute(OneComplex* in, OneComplex* out) {
        if (in && (in != (OneComplex*)buffer)) {
            std::memcpy(buffer, in, sizeof(fftw_complex) * Nx * Ny * Nz);
        }
        fftw_execute(plan_forward);
        if (out && (out != (OneComplex*)buffer)) {
            std::memcpy(out, buffer, sizeof(fftw_complex) * Nx * Ny * Nz);
        }
        return (OneComplex*)buffer;
    }

    OneComplex* BackwardExecute(OneComplex* in, OneComplex* out) {
        if (in && (in != (OneComplex*)buffer)) {
            if (in == out) {
                return (OneComplex*)buffer;
            }
            std::memcpy(buffer, in, sizeof(fftw_complex) * Nx * Ny * Nz);
        }
        fftw_execute(plan_backward);
        if (out && (out != (OneComplex*)buffer)) {
            std::memcpy(out, buffer, sizeof(fftw_complex) * Nx * Ny * Nz);
        }
        return (OneComplex*)buffer;
    }
#endif
    //inとoutは同一である必要がある//
    OneComplex* ForwardDirect(OneComplex* inout) {
        fftw_execute_dft(plan_forward, (fftw_complex*)inout, (fftw_complex*)inout);
        return inout;
    }

    //inとoutは同一である必要がある//
    OneComplex* BackwardDirect(OneComplex* inout) {
        fftw_execute_dft(plan_backward, (fftw_complex*)inout, (fftw_complex*)inout);
        return (OneComplex*)inout;
    }

};


#if 0
class FFTW_R2C_Executor {
private:
    int Nx = 0;
    int Ny = 0;
    int Nz = 0;
    fftw_complex* buffer = nullptr;
    fftw_plan plan_forward{ 0 };
    fftw_plan plan_backward{ 0 };
public:
    ~FFTW_R2C_Executor() {
        if (buffer) {
            fftw_destroy_plan(plan_forward);
            fftw_destroy_plan(plan_backward);
            fftw_free(buffer);
        }
    }

    void Initialize(int grid_x, int grid_y, int grid_z, int fftw_flags) {
        Nx = grid_x;
        Ny = grid_y;
        Nz = grid_z;
        buffer = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (Nx/2+1) * Ny * Nz);
        plan_forward = fftw_plan_dft_r2c_3d(Nz, Ny, Nx, (double*)buffer, buffer, fftw_flags);
        plan_backward = fftw_plan_dft_c2r_3d(Nz, Ny, Nx, buffer, (double*)buffer, fftw_flags);
    }

    fftw_complex* GetBuffer() { return buffer; };

    fftw_complex* ForwardExecute(double* in, fftw_complex* out) {
        if (in && (in != (double*)buffer)) {
            double* dbuf = (double*)buffer;
            const int stride = (Nx / 2 + 1)*2;
            for (int iyz = 0; iyz < Ny * Nz; ++iyz) {
                for (int ix = 0; ix < Nx; ++ix) {
                    dbuf[ix + stride * iyz] = in[ix + Nx * iyz];
                }
            }
        }
        fftw_execute(plan_forward);
        if (out && (out != buffer)) {
            std::memcpy(out, buffer, sizeof(fftw_complex) * (Nx / 2 + 1) * Ny * Nz);
        }
        return buffer;
    }

    fftw_complex* BackwardExecute(fftw_complex* in, double* out) {
        if (in && (in != buffer)) {
            std::memcpy(buffer, in, sizeof(fftw_complex) * (Nx / 2 + 1) * Ny * Nz);
        }
        fftw_execute(plan_backward);
        if (out && (out != (double*)buffer)) {
            const double* dbuf = (const double*)buffer;
            const int stride = (Nx / 2 + 1) * 2;
            for (int iyz = 0; iyz < Ny * Nz; ++iyz) {
                for (int ix = 0; ix < Nx; ++ix) {
                    out[ix + Nx * iyz] = dbuf[ix + stride * iyz];
                }
            }            
        }
        return buffer;
    }

};
#endif


class FFTW_Executor1D {
private:
    int Nx = 0;
    fftw_complex* buffer = nullptr;
    fftw_plan plan_forward{ 0 };
    fftw_plan plan_backward{ 0 };
public:
    ~FFTW_Executor1D() {
        if (buffer) {
            fftw_destroy_plan(plan_forward);
            fftw_destroy_plan(plan_backward);
            fftw_free(buffer);
        }
    }

    void Initialize(int grid_x, int fftw_flags) {
        Nx = grid_x;
        buffer = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * Nx);
        plan_forward = fftw_plan_dft_1d(Nx, buffer, buffer, FFTW_FORWARD, fftw_flags);
        plan_backward = fftw_plan_dft_1d(Nx, buffer, buffer, FFTW_BACKWARD, fftw_flags);
    }

    OneComplex* GetBuffer() { return (OneComplex*)buffer; };
#if 0
    OneComplex* ForwardExecute(OneComplex* in, OneComplex* out) {
        if ((in == (OneComplex*)buffer) && (out == (OneComplex*)buffer)) {
            fftw_execute(plan_forward);
            return (OneComplex*)buffer;
        } else if (in == out) {
            fftw_execute_dft(plan_forward, (fftw_complex*)in, (fftw_complex*)out);
            return out;
        }else{
            memcpy(out, in, sizeof(fftw_complex) * Nx); //because fftw_plan is created as in-place mode.
            fftw_execute_dft(plan_forward, (fftw_complex*)out, (fftw_complex*)out);
            return out;
        }
    }

    OneComplex* BackwardExecute(OneComplex* in, OneComplex* out) {
        if ((in == (OneComplex*)buffer) && (out == (OneComplex*)buffer)) {
            fftw_execute(plan_backward);
            return (OneComplex*)buffer;
        } else if (in == out) {
            fftw_execute_dft(plan_backward, (fftw_complex*)in, (fftw_complex*)out);
            return out;
        } else {
            memcpy(out, in, sizeof(fftw_complex) * Nx); //because fftw_plan is created as in-place mode.
            fftw_execute_dft(plan_backward, (fftw_complex*)out, (fftw_complex*)out);
            return out;
        }
    }
#endif

    OneComplex* ForwardDirect(OneComplex* inout) {
        fftw_execute_dft(plan_forward, (fftw_complex*)inout, (fftw_complex*)inout);
        return inout;
    }


    OneComplex* BackwardDirect(OneComplex* inout) {
        fftw_execute_dft(plan_backward, (fftw_complex*)inout, (fftw_complex*)inout);
        return (OneComplex*)inout;
    }
};




class FFTW_ExecutorMany1D {
private:
    int Nx = 0;
    fftw_plan plan_forward{ 0 };
    fftw_plan plan_backward{ 0 };
public:
    ~FFTW_ExecutorMany1D() {
        if (Nx > 0) {
            fftw_destroy_plan(plan_forward);
            fftw_destroy_plan(plan_backward);
        }
    }

    void Initialize(int grid_x, int num_lines) {
        Nx = grid_x;
        
        
        plan_forward = fftw_plan_many_dft(1, &Nx, num_lines, nullptr, &Nx, 1, (int)grid_x, nullptr, &Nx, 1, (int)grid_x, FFTW_FORWARD, FFTW_ESTIMATE);
        plan_backward = fftw_plan_many_dft(1, &Nx, num_lines, nullptr, &Nx, 1, (int)grid_x, nullptr, &Nx, 1, (int)grid_x, FFTW_BACKWARD, FFTW_ESTIMATE);

    }


    OneComplex* ForwardDirect(OneComplex* inout) {
        fftw_execute_dft(plan_forward, (fftw_complex*)inout, (fftw_complex*)inout);
        return inout;
    }


    OneComplex* BackwardDirect(OneComplex* inout) {
        fftw_execute_dft(plan_backward, (fftw_complex*)inout, (fftw_complex*)inout);
        return (OneComplex*)inout;
    }
};



#if 0
/*
* 奇関数を前提に, バッファを倍にとってmirrorすることで偶関数として処理できるようにするもの
* 
* 
*/
class FFTW_Mirror_Executor {
private:
    int Nx = 0;
    int Ny = 0;
    int Nz = 0;
    fftw_complex* buffer = nullptr;
    fftw_plan plan_forward[8];
    fftw_plan plan_backward[8];
public:
    ~FFTW_Mirror_Executor() {
        if (buffer) {
            for (int i = 0; i < 8; ++i) {
                fftw_destroy_plan(plan_forward[i]);
                fftw_destroy_plan(plan_backward[i]);
            }
            fftw_free(buffer);
        }
    }

    void Initialize(int grid_x, int grid_y, int grid_z, int fftw_flags) {
        Nx = grid_x;
        Ny = grid_y;
        Nz = grid_z;
        buffer = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * Nx * Ny * Nz * 8);
        for (int i = 0; i < 8; ++i) {
            const int mirror_size_x = (i & 0x1) ? Nx * 2 : Nx;
            const int mirror_size_y = (i & 0x2) ? Ny * 2 : Ny;
            const int mirror_size_z = (i & 0x4) ? Nz * 2 : Nz;
            plan_forward[i] = fftw_plan_dft_3d(mirror_size_z, mirror_size_y, mirror_size_x, buffer, buffer, FFTW_FORWARD, fftw_flags);
            plan_backward[i] = fftw_plan_dft_3d(mirror_size_z, mirror_size_y, mirror_size_x, buffer, buffer, FFTW_BACKWARD, fftw_flags);
        }
    }

    fftw_complex* GetBuffer() { return buffer; };

    fftw_complex* ForwardExecute(fftw_complex* in, fftw_complex* out) {
        if (in && (in != buffer)) {
            std::memcpy(buffer, in, sizeof(fftw_complex) * Nx * Ny * Nz);
        }
        fftw_execute(plan_forward[0]);
        if (out && (out != buffer)) {
            std::memcpy(out, buffer, sizeof(fftw_complex) * Nx * Ny * Nz);
        }
        return buffer;
    }

    fftw_complex* BackwardExecute(fftw_complex* in, fftw_complex* out) {
        if (in && (in != buffer)) {
            if (in == out) {
                return buffer;
            }
            std::memcpy(buffer, in, sizeof(fftw_complex) * Nx * Ny * Nz);
        }
        fftw_execute(plan_backward[0]);
        if (out && (out != buffer)) {
            std::memcpy(out, buffer, sizeof(fftw_complex) * Nx * Ny * Nz);
        }
        return buffer;
    }


    fftw_complex* ForwardDirect(fftw_complex* inout) {
        fftw_execute_dft(plan_forward[0], inout, inout);
        return inout;
    }

    fftw_complex* BackwardDirect(fftw_complex* inout) {
        fftw_execute_dft(plan_backward[0], inout, inout);
        return inout;
    }


    fftw_complex* ForwardDirectMirror(int odd_flag, fftw_complex* inout) {
        fftw_execute_dft(plan_forward[odd_flag], inout, inout);
        return inout;
    }

    fftw_complex* BackwardDirectMirror(int odd_flag, fftw_complex* inout) {
        fftw_execute_dft(plan_backward[odd_flag], inout, inout);
        return inout;
    }
};
#endif


class FFTW_ExecutorMany3D {
private:
    int Nxyz[3]{ 0,0,0 };
    int Nbundle = 0;
    
    fftw_plan plan_forward{ 0 };
    fftw_plan plan_backward{ 0 };
public:
    ~FFTW_ExecutorMany3D() {
        if (Nbundle > 0) {
            fftw_destroy_plan(plan_forward);
            fftw_destroy_plan(plan_backward);            
        }
    }

    //ld: leading dimension of each data, which may be useful for SIMD and GPU//
    void Initialize(int grid_x, int grid_y, int grid_z, int num_bundle_, OneComplex* buffer = nullptr, int ld=0) {
        if (Nbundle > 0) {
            fftw_destroy_plan(plan_forward);
            fftw_destroy_plan(plan_backward);
            Nbundle = 0;
        }
        Nxyz[2] = grid_x;
        Nxyz[1] = grid_y;
        Nxyz[0] = grid_z;
        Nbundle = num_bundle_;
        if (ld == 0) {
            ld = Nxyz[0] * Nxyz[1] * Nxyz[2];
        }

        plan_forward = fftw_plan_many_dft(3, Nxyz, Nbundle, (fftw_complex*)buffer, Nxyz, 1, ld,
            (fftw_complex*)buffer, Nxyz, 1, ld, FFTW_FORWARD, FFTW_ESTIMATE);
        plan_backward = fftw_plan_many_dft(3, Nxyz, Nbundle, (fftw_complex*)buffer, Nxyz, 1, ld,
            (fftw_complex*)buffer, Nxyz, 1, ld, FFTW_BACKWARD, FFTW_ESTIMATE);
    }

    int GetNumBundle() {
        return Nbundle;
    }


    //OneComplex* GetBuffer() { return (OneComplex*)buffer; };

    OneComplex* ForwardExecute(OneComplex* in, OneComplex* out) {
        fftw_execute_dft(plan_forward, (fftw_complex*)in, (fftw_complex*)out);
        return out;
    }

    OneComplex* BackwardExecute(OneComplex* in, OneComplex* out) {
        fftw_execute_dft(plan_backward, (fftw_complex*)in, (fftw_complex*)out);
        return (OneComplex*)out;
    }


    OneComplex* ForwardDirect(OneComplex* inout) {
        //fftw_execute(plan_forward);
        fftw_execute_dft(plan_forward, (fftw_complex*)inout, (fftw_complex*)inout);
        return inout;
    }


    OneComplex* BackwardDirect(OneComplex* inout) {
        //fftw_execute(plan_backward);
        fftw_execute_dft(plan_backward, (fftw_complex*)inout, (fftw_complex*)inout);
        return (OneComplex*)inout;
    }

};


#endif

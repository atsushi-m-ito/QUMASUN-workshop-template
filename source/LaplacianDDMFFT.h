#pragma once
#include "fftw_executor.h"
#include "DDMExchange.h"
#include "GridRange.h"
#include "soacomplex.h"
#include "transpose.h"

#define WATCH_DDMFFT

/*
* DDM分割された通常グリッド(non-HRグリッド)の軌道をbundleで受け取り、
* X,Y,Zそれぞれの方向ごとに、通信を行ってFFTでLaplacianを演算して逆FFTで戻すことを行う
*/
template <class WATCH >
void LaplacianDDMFFT(const GridRangeMPI& l_grid, const GridRange& global_grid,
    DDMExchange& ddm_ex, 
#ifndef USE_FFT_MANY
    FFTW_Executor1D& my_fft_x,
    FFTW_Executor1D& my_fft_y,
    FFTW_Executor1D& my_fft_z,
#endif
    //double* work_buffer,
    SoAComplex* l_Hp_set, const SoAComplex* l_psi_set, int num_bundle,
    double m_dx, double m_dy, double m_dz,
    double gx, double gy, double gz,
    WATCH& watch    )
{
    

    auto& range = l_grid;
    const int local_size_x = range.end_x - range.begin_x;
    const int local_size_y = range.end_y - range.begin_y;
    const int local_size_z = range.end_z - range.begin_z;
    const int Nx = global_grid.SizeX();
    const int Ny = global_grid.SizeY();
    const int Nz = global_grid.SizeZ();

#ifdef USE_FFT_MANY
    FFTW_ExecutorMany1D my_fft_many_x;
    FFTW_ExecutorMany1D my_fft_many_y;
    FFTW_ExecutorMany1D my_fft_many_z;
    const int num_lines_x = ddm_ex.GetNumLinesExchangedX(local_size_y * local_size_z * num_bundle);
    const int num_lines_y = ddm_ex.GetNumLinesExchangedY(local_size_z * local_size_x * num_bundle);
    const int num_lines_z = ddm_ex.GetNumLinesExchangedZ(local_size_x * local_size_y * num_bundle);
    my_fft_many_x.Initialize(Nx, num_lines_x);
    my_fft_many_y.Initialize(Ny, num_lines_y);
    my_fft_many_z.Initialize(Nz, num_lines_z);
#endif

    const double coef_x = 2.0 * M_PI / ((double)Nx * m_dx); //hr_Nx * (dx/m_HR_ratio_x) == Nx*dx//
    const double coef_y = 2.0 * M_PI / ((double)Ny * m_dy);
    const double coef_z = 2.0 * M_PI / ((double)Nz * m_dz);
    //const double invN = 1.0 / (double)(Nx* Ny* Nz);


#ifdef WATCH_DDMFFT
    MPI_Barrier(l_grid.mpi_comm);
    watch.Record(18);
#endif


    const int64_t local_size = local_size_x * local_size_y * local_size_z;
    const int64_t enough_size = std::max(ddm_ex.GetExcangeBufferSize(Nx, Ny, Nz), local_size) * num_bundle;


    auto modified_buffer = std::make_unique<double[]>(enough_size*6);
    double* fft_buffer = &modified_buffer[0] + enough_size * 2;
    double* trans_buffer = &modified_buffer[0] + enough_size * 2 * 2;
    

    //initial buffer set//
    //const int64_t hr_local_size = local_size * m_HR_ratio_x * m_HR_ratio_y * m_HR_ratio_z;
    //const size_t hr_size_3d = m_size_3d * m_HR_ratio_x * m_HR_ratio_y * m_HR_ratio_z;
    //const double invN_HR = 1.0 / (double)(hr_size_3d);

    for (int64_t n = 0; n < num_bundle; ++n) {
        for (int64_t i = 0; i < local_size; ++i) {
            modified_buffer[n * local_size * 2 + i * 2] = l_psi_set[n].re[i];
            modified_buffer[n * local_size * 2 + i * 2 + 1] = l_psi_set[n].im[i];
        }
    }
#ifdef WATCH_DDMFFT
    watch.Record(90);
#endif

    //x exchange////////////////////////////////////////////////////////
    // x方向に領域分割されたバッファをExchangeして、x方向に連続なバッファに並べ替える//
    int my_line_yz = ddm_ex.ForwardExchangeX(Nx, local_size_y * local_size_z * num_bundle, &modified_buffer[0], &fft_buffer[0], &trans_buffer[0], 2);//2 means complex
#ifdef WATCH_DDMFFT
    watch.Record(91);
#endif
    //such as fft////////////
#ifdef USE_FFT_MANY
    my_fft_many_x.ForwardDirect((OneComplex*)&fft_buffer[0]);
    for (int l = 0; l < my_line_yz; ++l) {
        auto* line = (OneComplex*)&fft_buffer[l * Nx * 2];
        for (int ix = 0; ix < Nx; ++ix) {
            const double kx = coef_x * (double)(ix * 2 > Nx ? ix - Nx : ix);
            const double kx2_2 = (kx + gx) * (kx + gx) / (2.0 * (double)Nx);
            const double re = line[ix].r;
            const double im = line[ix].i;
            line[ix].r = re * kx2_2;
            line[ix].i = im * kx2_2;
        }
    }
    my_fft_many_x.BackwardDirect((OneComplex*)&fft_buffer[0]);

#else
    for (int l = 0; l < my_line_yz; ++l) {
        auto* line = (OneComplex*)&fft_buffer[l * Nx * 2];
        my_fft_x.ForwardDirect(line);
        //auto* l_dfdx = (fftw_complex*)&fft_buffer[l * Nx * 2];
        for (int ix = 0; ix < Nx; ++ix) {
            const double kx = coef_x * (double)(ix * 2 > Nx ? ix - Nx : ix);
            const double kx2_2 = (kx + gx) * (kx + gx) / (2.0 * (double)Nx);
            const double re = line[ix].r;
            const double im = line[ix].i;
            line[ix].r = re * kx2_2;
            line[ix].i = im * kx2_2;
        }
        my_fft_x.BackwardDirect(line);

    }
#endif
#ifdef WATCH_DDMFFT
    watch.Record(92);
#endif

    ddm_ex.BackwardExchangeX(Nx, local_size_y * local_size_z * num_bundle, &fft_buffer[0], &fft_buffer[0], &trans_buffer[0], 2);//2 means complex
#ifdef WATCH_DDMFFT
    watch.Record(93);
#endif
    //add 
    for (int64_t n = 0; n < num_bundle; ++n) {
        for (int64_t i = 0; i < local_size; ++i) {
            l_Hp_set[n].re[i] += fft_buffer[n * local_size * 2 + i * 2];
            l_Hp_set[n].im[i] += fft_buffer[n * local_size * 2 + i * 2 + 1];
        }
    }
#ifdef WATCH_DDMFFT
    watch.Record(90);
#endif


    //y exchange////////////////////////////////////////////////////////


    //転置 x,y,z to y,z,x//
    for (int n = 0; n < num_bundle; ++n) {
        Transpose3<OneComplex>(local_size_x, local_size_y, local_size_z, ((OneComplex*)&fft_buffer[0])+local_size*n, ((OneComplex*)&modified_buffer[0])+local_size * n);
    }
#ifdef WATCH_DDMFFT
    watch.Record(94);
#endif


    // y方向に領域分割されたバッファをExchangeして、y方向に連続なバッファに並べ替える//    
    int my_line_xz = ddm_ex.ForwardExchangeY(Ny, local_size_x * local_size_z * num_bundle, &fft_buffer[0], &fft_buffer[0], &trans_buffer[0], 2);//2 means complex
    //int hr_Ny = Ny * m_HR_ratio_y;
#ifdef WATCH_DDMFFT
    watch.Record(91);
#endif
    //such as fft////////////
#ifdef USE_FFT_MANY
    my_fft_many_y.ForwardDirect((OneComplex*)&fft_buffer[0]);
    for (int l = 0; l < my_line_xz; ++l) {
        auto* line = (OneComplex*)&fft_buffer[l * Ny * 2];
        for (int iy = 0; iy < Ny; ++iy) {
            const double ky = coef_y * (double)(iy * 2 > Ny ? iy - Ny : iy);
            const double ky2_2 = (ky + gy) * (ky + gy) / (2.0 * (double)Ny);
            const double re = line[iy].r;
            const double im = line[iy].i;
            line[iy].r = re * ky2_2;
            line[iy].i = im * ky2_2;
        }        
    }
    my_fft_many_y.BackwardDirect((OneComplex*)&fft_buffer[0]);

#else
    for (int l = 0; l < my_line_xz; ++l) {
        auto* line = (OneComplex*)&fft_buffer[l * Ny * 2];
        my_fft_y.ForwardDirect(line);
        for (int iy = 0; iy < Ny; ++iy) {
            const double ky = coef_y * (double)(iy * 2 > Ny ? iy - Ny : iy);
            const double ky2_2 = (ky + gy) * (ky + gy) / (2.0 * (double)Ny);
            const double re = line[iy].r;
            const double im = line[iy].i;
            line[iy].r = re * ky2_2;
            line[iy].i = im * ky2_2;
        }
        my_fft_y.BackwardDirect(line);
    }
#endif
#ifdef WATCH_DDMFFT
    watch.Record(92);
#endif

    ddm_ex.BackwardExchangeY(Ny, local_size_x * local_size_z * num_bundle, &fft_buffer[0], &fft_buffer[0], &trans_buffer[0], 2);//2 means complex
#ifdef WATCH_DDMFFT
    watch.Record(93);
#endif

    //転置 y,z,x to x,y,z//
    for (int n = 0; n < num_bundle; ++n) {
        Transpose3_back<OneComplex>(local_size_y, local_size_z, local_size_x, ((OneComplex*)&trans_buffer[0]) + local_size * n, ((OneComplex*)&fft_buffer[0]) + local_size * n);
    }
#ifdef WATCH_DDMFFT
    watch.Record(94);
#endif

    //add
    for (int64_t n = 0; n < num_bundle; ++n) {
        for (int64_t i = 0; i < local_size; ++i) {
            l_Hp_set[n].re[i] += trans_buffer[n * local_size * 2 + i * 2];
            l_Hp_set[n].im[i] += trans_buffer[n * local_size * 2 + i * 2 + 1];
        }
    }
#ifdef WATCH_DDMFFT
    watch.Record(90);
#endif
    //z exchange////////////////////////////////////////////////////////


    //転置 x,y,z to z,x,y//
    for (int n = 0; n < num_bundle; ++n) {
        Transpose3_back<OneComplex>(local_size_x, local_size_y, local_size_z, ((OneComplex*)&fft_buffer[0]) + local_size * n, ((OneComplex*)&modified_buffer[0]) + local_size * n);
    }
#ifdef WATCH_DDMFFT
    watch.Record(94);
#endif
    // z方向に領域分割されたバッファをExchangeして、z方向に連続なバッファに並べ替える//    
    int my_line_xy = ddm_ex.ForwardExchangeZ(Nz, local_size_x * local_size_y * num_bundle, &fft_buffer[0], &fft_buffer[0], &trans_buffer[0], 2);//2 means complex
#ifdef WATCH_DDMFFT
    watch.Record(91);
#endif

    //such as fft////////////
#ifdef USE_FFT_MANY
    my_fft_many_z.ForwardDirect((OneComplex*)&fft_buffer[0]);
    for (int l = 0; l < my_line_xy; ++l) {
        auto* line = (OneComplex*)&fft_buffer[l * Nz * 2];
        for (int iz = 0; iz < Nz; ++iz) {
            const double kz = coef_z * (double)(iz * 2 > Nz ? iz - Nz : iz);
            const double kz2_2 = (kz + gz) * (kz + gz) / (2.0 * (double)Nz);
            const double re = line[iz].r;
            const double im = line[iz].i;
            line[iz].r = re * kz2_2;
            line[iz].i = im * kz2_2;
        }        
    }
    my_fft_many_z.BackwardDirect((OneComplex*)&fft_buffer[0]);
#else
    for (int l = 0; l < my_line_xy; ++l) {
        auto* line = (OneComplex*)&fft_buffer[l * Nz * 2];
        my_fft_z.ForwardDirect(line);
        for (int iz = 0; iz < Nz; ++iz) {
            const double kz = coef_z * (double)(iz * 2 > Nz ? iz - Nz : iz);
            const double kz2_2 = (kz + gz) * (kz + gz) / (2.0 * (double)Nz);
            const double re = line[iz].r;
            const double im = line[iz].i;
            line[iz].r = re * kz2_2;
            line[iz].i = im * kz2_2;
        }
        my_fft_z.BackwardDirect(line);
    }
#endif
#ifdef WATCH_DDMFFT
    watch.Record(92);
#endif

    ddm_ex.BackwardExchangeZ(Nz, local_size_x * local_size_y * num_bundle, &fft_buffer[0], &fft_buffer[0], &trans_buffer[0], 2 );//2 means complex
#ifdef WATCH_DDMFFT
    watch.Record(93);
#endif

    //転置 z,x,y to x,y,z//
    for (int n = 0; n < num_bundle; ++n) {
        Transpose3<OneComplex>(local_size_z, local_size_x, local_size_y, ((OneComplex*)&trans_buffer[0]) + local_size * n, ((OneComplex*)&fft_buffer[0]) + local_size * n);
    }
#ifdef WATCH_DDMFFT
    watch.Record(94);
#endif

    //add
    for (int64_t n = 0; n < num_bundle; ++n) {
        for (int64_t i = 0; i < local_size; ++i) {
            l_Hp_set[n].re[i] += trans_buffer[n * local_size * 2 + i * 2];
            l_Hp_set[n].im[i] += trans_buffer[n * local_size * 2 + i * 2 + 1];
        }
    }
#ifdef WATCH_DDMFFT
    watch.Record(90);
#endif

}


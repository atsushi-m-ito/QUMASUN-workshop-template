#pragma once
#include "fftw_executor.h"
#include "DDMExchange.h"
#include "GridRange.h"
#include "soacomplex.h"
#include "transpose.h"


/*
* DDM分割された通常グリッド(non-HRグリッド)の電子密度l_Vを受けとり、
* l_dVdx等にnullptr以外の値がセットされた場合は、ポテンシャルのgradientを返す
*/
inline
void GradientDDMFFT(const GridRangeMPI& l_grid, const GridRange& global_grid,
    DDMExchange& ddm_ex, 
#ifdef USE_FFT_MANY
    FFTW_ExecutorMany1D& my_fft_many_x,
    FFTW_ExecutorMany1D& my_fft_many_y,
    FFTW_ExecutorMany1D& my_fft_many_z,
#else
    FFTW_Executor1D& my_fft_x,
    FFTW_Executor1D& my_fft_y,
    FFTW_Executor1D& my_fft_z, 
#endif
    //double* work_buffer,
    const double* l_V, double* l_dV_dx, double* l_dV_dy, double* l_dV_dz, double coef,
    double m_dx, double m_dy, double m_dz, bool is_complex) {
    //const bool is_root = IsRoot(m_mpi_comm);




    auto& range = l_grid;
    const int local_size_x = range.end_x - range.begin_x;
    const int local_size_y = range.end_y - range.begin_y;
    const int local_size_z = range.end_z - range.begin_z;
    const int Nx = global_grid.SizeX();
    const int Ny = global_grid.SizeY();
    const int Nz = global_grid.SizeZ();
    const int split_x = range.num_split_x;
    const int split_y = range.num_split_y;
    const int split_z = range.num_split_z;

    const double coef_x = coef * 2.0 * M_PI / ((double)Nx * m_dx); //hr_Nx * (dx/m_HR_ratio_x) == Nx*dx//
    const double coef_y = coef * 2.0 * M_PI / ((double)Ny * m_dy);
    const double coef_z = coef * 2.0 * M_PI / ((double)Nz * m_dz);
    const double invN = 1.0 / (double)(Nx* Ny* Nz);


    const int64_t local_size = local_size_x * local_size_y * local_size_z;
    const int64_t enough_size = std::max(ddm_ex.GetExcangeBufferSize(Nx, Ny, Nz), local_size) ;


    auto modified_buffer = gy::make_unique_aligned<double[]>(enough_size*6);
    double* fft_buffer = &modified_buffer[0] + enough_size * 2;
    double* trans_buffer = &modified_buffer[0] + enough_size * 2 * 2;



    //initial buffer set//
    if (is_complex) {
        for (int64_t i = 0; i < local_size; ++i) {
            modified_buffer[i * 2] = l_V[i*2];
            modified_buffer[i * 2 + 1] = l_V[i * 2 + 1];
        }
    } else {
        for (int64_t i = 0; i < local_size; ++i) {
            modified_buffer[i * 2] = l_V[i];
            modified_buffer[i * 2 + 1] = 0.0;
        }
    }
    //watch.Record(57);

    //x exchange////////////////////////////////////////////////////////
    // x方向に領域分割されたバッファをExchangeして、x方向に連続なバッファに並べ替える//
    int my_line_yz = ddm_ex.ForwardExchangeX(Nx, local_size_y * local_size_z, &modified_buffer[0], &fft_buffer[0], &trans_buffer[0], 2);//2 means complex
    //const int hr_Nx = Nx * m_HR_ratio_x;
    //watch.Record(53);

    //such as fft////////////
#ifdef USE_FFT_MANY
    my_fft_many_x.ForwardDirect((OneComplex*)&fft_buffer[0]);
    for (int l = 0; l < my_line_yz; ++l) {
        auto* line = (OneComplex*)&fft_buffer[l * Nx * 2];
        //auto* l_dfdx = (fftw_complex*)&fft_buffer[l * Nx * 2];
        for (int ix = 0; ix < Nx; ++ix) {
            const double kx = coef_x * (double)(ix * 2 > Nx ? ix - Nx : ix) / (double)Nx;
            const double re = line[ix].r;
            const double im = line[ix].i;
            line[ix].r = -im * kx;
            line[ix].i = re * kx;
        }        
    }
    my_fft_many_x.BackwardDirect((OneComplex*)&fft_buffer[0]);
#else
    for (int l = 0; l < my_line_yz; ++l) {
        auto* line = (OneComplex*)&fft_buffer[l * Nx * 2];
        my_fft_x.ForwardDirect(line);
        //auto* l_dfdx = (fftw_complex*)&fft_buffer[l * Nx * 2];
        for (int ix = 0; ix < Nx; ++ix) {
            const double kx = coef_x * (double)(ix * 2 > Nx ? ix - Nx : ix) / (double)Nx;
            const double re = line[ix].r;
            const double im = line[ix].i;
            line[ix].r = -im * kx;
            line[ix].i = re * kx;
        }
        my_fft_x.BackwardDirect(line);

    }
#endif
    //watch.Record(58);

    ddm_ex.BackwardExchangeX(Nx, local_size_y * local_size_z, &fft_buffer[0], &fft_buffer[0], &trans_buffer[0], 2);//2 means complex
    if (is_complex) {
        for (int64_t i = 0; i < local_size; ++i) {
            l_dV_dx[i*2] = fft_buffer[i * 2];
            l_dV_dx[i * 2+1] = fft_buffer[i * 2 + 1];
        }
    } else {
        for (int64_t i = 0; i < local_size; ++i) {
            l_dV_dx[i] = fft_buffer[i * 2];
        }
    }
    //watch.Record(51);

#if 0 //def _DEBUG
    printf("[%d]after BackwardExchangeX:\n", GetProcessID(m_ddm_comm));
    fflush(stdout);
    MPI_Barrier(m_ddm_comm);
#endif

    //転置 x,y,z to y,z,x//
    Transpose3<OneComplex>(local_size_x, local_size_y, local_size_z, (OneComplex*)&fft_buffer[0], (OneComplex*)&modified_buffer[0]);
    //watch.Record(52);

#if 0 //def _DEBUG
    printf("[%d]after Transpose3-1: %d, %d\n", GetProcessID(m_ddm_comm));
    fflush(stdout);
    MPI_Barrier(m_ddm_comm);
#endif

    //y exchange////////////////////////////////////////////////////////
    // y方向に領域分割されたバッファをExchangeして、y方向に連続なバッファに並べ替える//    
    int my_line_xz = ddm_ex.ForwardExchangeY(Ny, local_size_x * local_size_z, &fft_buffer[0], &fft_buffer[0], &trans_buffer[0], 2);//2 means complex
    //int hr_Ny = Ny * m_HR_ratio_y;
    //watch.Record(53);
    //such as fft////////////
#ifdef USE_FFT_MANY
    my_fft_many_y.ForwardDirect((OneComplex*)&fft_buffer[0]);
    for (int l = 0; l < my_line_xz; ++l) {
        auto* line = (OneComplex*)&fft_buffer[l * Ny * 2];
        for (int iy = 0; iy < Ny; ++iy) {
            const double ky = coef_y * (double)(iy * 2 > Ny ? iy - Ny : iy) / (double)Ny;
            const double re = line[iy].r;
            const double im = line[iy].i;
            line[iy].r = -im * ky;
            line[iy].i = re * ky;
        }        
    }
    my_fft_many_y.BackwardDirect((OneComplex*)&fft_buffer[0]);
#else
    for (int l = 0; l < my_line_xz; ++l) {
        auto* line = (OneComplex*)&fft_buffer[l * Ny * 2];
        my_fft_y.ForwardDirect(line);
        for (int iy = 0; iy < Ny; ++iy) {
            const double ky = coef_y * (double)(iy * 2 > Ny ? iy - Ny : iy) / (double)Ny;
            const double re = line[iy].r;
            const double im = line[iy].i;
            line[iy].r = -im * ky;
            line[iy].i = re * ky;
        }
        my_fft_y.BackwardDirect(line);
    }
#endif
    //watch.Record(58);

    ddm_ex.BackwardExchangeY(Ny, local_size_x * local_size_z, &fft_buffer[0], &fft_buffer[0], &trans_buffer[0], 2);//2 means complex
    //watch.Record(51);

    //転置 y,z,x to x,y,z//
    Transpose3_back<OneComplex>(local_size_y, local_size_z, local_size_x, (OneComplex*)&trans_buffer[0], (OneComplex*)&fft_buffer[0]);
    if (is_complex) {
        for (int64_t i = 0; i < local_size; ++i) {
            l_dV_dy[i * 2] = trans_buffer[i * 2];
            l_dV_dy[i * 2 + 1] = trans_buffer[i * 2 + 1];
        }
    } else {
        for (int64_t i = 0; i < local_size; ++i) {
            l_dV_dy[i] = trans_buffer[i * 2];
        }
    }
    //watch.Record(52);

#if 0 //def _DEBUG
    printf("[%d]after Transpose3-2: %d, %d\n", GetProcessID(m_ddm_comm));
    fflush(stdout);
    MPI_Barrier(m_ddm_comm);
#endif

    //z exchange////////////////////////////////////////////////////////
    //転置 x,y,z to z,x,y to //
    Transpose3_back<OneComplex>(local_size_x, local_size_y, local_size_z, (OneComplex*)&fft_buffer[0], (OneComplex*)&modified_buffer[0]);
    //watch.Record(52);

    // z方向に領域分割されたバッファをExchangeして、z方向に連続なバッファに並べ替える//    
    int my_line_xy = ddm_ex.ForwardExchangeZ(Nz, local_size_x * local_size_y, &fft_buffer[0], &fft_buffer[0], &trans_buffer[0], 2);//2 means complex
    //int my_line_begin_xy = DDM::GridBegin(local_size_x * local_size_y, ddm_ex.GetProccessID_Z(), split_z);
    //int hr_Nz = Nz * m_HR_ratio_z;
    //watch.Record(53);

    //such as fft////////////
#ifdef USE_FFT_MANY
    my_fft_many_z.ForwardDirect((OneComplex*)&fft_buffer[0]);
    for (int l = 0; l < my_line_xy; ++l) {
        auto* line = (OneComplex*)&fft_buffer[l * Nz * 2];        
        for (int iz = 0; iz < Nz; ++iz) {
            const double kz = coef_z * (double)(iz * 2 > Nz ? iz - Nz : iz) / (double)Nz;
            const double re = line[iz].r;
            const double im = line[iz].i;
            line[iz].r = -im * kz;
            line[iz].i = re * kz;
        }        
    }
    my_fft_many_z.BackwardDirect((OneComplex*)&fft_buffer[0]);
#else
    for (int l = 0; l < my_line_xy; ++l) {
        auto* line = (OneComplex*)&fft_buffer[l * Nz * 2];
        my_fft_z.ForwardDirect(line);
        for (int iz = 0; iz < Nz; ++iz) {
            const double kz = coef_z * (double)(iz * 2 > Nz ? iz - Nz : iz) / (double)Nz;
            const double re = line[iz].r;
            const double im = line[iz].i;
            line[iz].r = -im * kz;
            line[iz].i = re * kz;
        }
        my_fft_z.BackwardDirect(line);
    }
#endif
    //watch.Record(58);

    ddm_ex.BackwardExchangeZ(Nz, local_size_x * local_size_y, &fft_buffer[0], &fft_buffer[0], &trans_buffer[0], 2 );//2 means complex
    //watch.Record(51);
    //転置 z,x,y to x,y,z//
    Transpose3<OneComplex>(local_size_z, local_size_x, local_size_y, (OneComplex*)&trans_buffer[0], (OneComplex*)&fft_buffer[0]);
    if (is_complex) {
        for (int64_t i = 0; i < local_size; ++i) {
            l_dV_dz[i*2] = trans_buffer[i * 2];
            l_dV_dz[i * 2+1] = trans_buffer[i * 2 + 1];
        }
    } else {
        for (int64_t i = 0; i < local_size; ++i) {
            l_dV_dz[i] = trans_buffer[i * 2];
        }
    }
    //watch.Record(52);


}




/*
* DDM分割された通常グリッド(non-HRグリッド)の電子密度l_Vを受けとり、
* l_dVdx等にnullptr以外の値がセットされた場合は、ポテンシャルのgradientを返す
*/
inline
void GradientDDMFFT_HR2(const GridRangeMPI& l_grid, const GridRange& global_grid,
    DDMExchange& ddm_ex,
    FFTW_Executor1D& my_fft_x,
    FFTW_Executor1D& my_fft_y,
    FFTW_Executor1D& my_fft_z,
    //double* work_buffer,
    const double* l_V, double* l_dV_dx, double* l_dV_dy, double* l_dV_dz, double coef,
    double m_dx, double m_dy, double m_dz, bool is_complex, int HR_ratio_x, int HR_ratio_y, int HR_ratio_z) {
    //const bool is_root = IsRoot(m_mpi_comm);




    auto& range = l_grid;
    const int local_size_x = range.end_x - range.begin_x;
    const int local_size_y = range.end_y - range.begin_y;
    const int local_size_z = range.end_z - range.begin_z;
    const int Nx = global_grid.SizeX();
    const int Ny = global_grid.SizeY();
    const int Nz = global_grid.SizeZ();
    const int split_x = range.num_split_x;
    const int split_y = range.num_split_y;
    const int split_z = range.num_split_z;

    const double coef_x = coef * 2.0 * M_PI / ((double)Nx * m_dx); //hr_Nx * (dx/m_HR_ratio_x) == Nx*dx//
    const double coef_y = coef * 2.0 * M_PI / ((double)Ny * m_dy);
    const double coef_z = coef * 2.0 * M_PI / ((double)Nz * m_dz);
    const double invN = 1.0 / (double)(Nx * Ny * Nz);


    const int64_t local_size = local_size_x * local_size_y * local_size_z;
    const int64_t enough_size = std::max(ddm_ex.GetExcangeBufferSize(Nx, Ny, Nz), local_size);


    auto modified_buffer = gy::make_unique_aligned<double[]>(enough_size * 6);
    double* fft_buffer = &modified_buffer[0] + enough_size * 2;
    double* trans_buffer = &modified_buffer[0] + enough_size * 2 * 2;


    
    //initial buffer set//
    if (is_complex) {
        for (int64_t i = 0; i < local_size; ++i) {
            modified_buffer[i * 2] = l_V[i * 2];
            modified_buffer[i * 2 + 1] = l_V[i * 2 + 1];
        }
    } else {
        for (int64_t i = 0; i < local_size; ++i) {
            modified_buffer[i * 2] = l_V[i];
            modified_buffer[i * 2 + 1] = 0.0;
        }
    }
    //watch.Record(57);

    //x exchange////////////////////////////////////////////////////////
    // x方向に領域分割されたバッファをExchangeして、x方向に連続なバッファに並べ替える//
    int my_line_yz = ddm_ex.ForwardExchangeX(Nx/ HR_ratio_x, local_size_y * local_size_z, &modified_buffer[0], &fft_buffer[0], &trans_buffer[0], 2* HR_ratio_x);//2 means complex
    //const int hr_Nx = Nx * m_HR_ratio_x;
    //watch.Record(53);

    //such as fft////////////
    for (int l = 0; l < my_line_yz; ++l) {
        auto* line = (OneComplex*)&fft_buffer[l * Nx * 2];
        my_fft_x.ForwardDirect(line);
        //auto* l_dfdx = (fftw_complex*)&fft_buffer[l * Nx * 2];
        for (int ix = 0; ix < Nx; ++ix) {
            const double kx = coef_x * (double)(ix * 2 > Nx ? ix - Nx : ix) / (double)Nx;
            const double re = line[ix].r;
            const double im = line[ix].i;
            line[ix].r = -im * kx;
            line[ix].i = re * kx;
        }
        my_fft_x.BackwardDirect(line);

    }
    //watch.Record(58);

    ddm_ex.BackwardExchangeX(Nx/ HR_ratio_x, local_size_y * local_size_z, &fft_buffer[0], &fft_buffer[0], &trans_buffer[0], 2* HR_ratio_x);//2 means complex
    if (is_complex) {
        for (int64_t i = 0; i < local_size; ++i) {
            l_dV_dx[i * 2] = fft_buffer[i * 2];
            l_dV_dx[i * 2 + 1] = fft_buffer[i * 2 + 1];
        }
    } else {
        for (int64_t i = 0; i < local_size; ++i) {
            l_dV_dx[i] = fft_buffer[i * 2];
        }
    }
    //watch.Record(51);

#if 0 //def _DEBUG
    printf("[%d]after BackwardExchangeX:\n", GetProcessID(m_ddm_comm));
    fflush(stdout);
    MPI_Barrier(m_ddm_comm);
#endif

    //転置 x,y,z to y,z,x//
    Transpose3<OneComplex>(local_size_x, local_size_y, local_size_z, (OneComplex*)&fft_buffer[0], (OneComplex*)&modified_buffer[0]);
    //watch.Record(52);

#if 0 //def _DEBUG
    printf("[%d]after Transpose3-1: %d, %d\n", GetProcessID(m_ddm_comm));
    fflush(stdout);
    MPI_Barrier(m_ddm_comm);
#endif

    //y exchange////////////////////////////////////////////////////////
    // y方向に領域分割されたバッファをExchangeして、y方向に連続なバッファに並べ替える//    
    int my_line_xz = ddm_ex.ForwardExchangeY(Ny/ HR_ratio_y, local_size_x * local_size_z, &fft_buffer[0], &fft_buffer[0], &trans_buffer[0], 2 * HR_ratio_y);//2 means complex
    //int hr_Ny = Ny * m_HR_ratio_y;
    //watch.Record(53);
    //such as fft////////////
    for (int l = 0; l < my_line_xz; ++l) {
        auto* line = (OneComplex*)&fft_buffer[l * Ny * 2];
        my_fft_y.ForwardDirect(line);
        for (int iy = 0; iy < Ny; ++iy) {
            const double ky = coef_y * (double)(iy * 2 > Ny ? iy - Ny : iy) / (double)Ny;
            const double re = line[iy].r;
            const double im = line[iy].i;
            line[iy].r = -im * ky;
            line[iy].i = re * ky;
        }
        my_fft_y.BackwardDirect(line);
    }
    //watch.Record(58);

    ddm_ex.BackwardExchangeY(Ny/ HR_ratio_y, local_size_x * local_size_z, &fft_buffer[0], &fft_buffer[0], &trans_buffer[0], 2* HR_ratio_y);//2 means complex
    //watch.Record(51);

    //転置 y,z,x to x,y,z//
    Transpose3_back<OneComplex>(local_size_y, local_size_z, local_size_x, (OneComplex*)&trans_buffer[0], (OneComplex*)&fft_buffer[0]);
    if (is_complex) {
        for (int64_t i = 0; i < local_size; ++i) {
            l_dV_dy[i * 2] = trans_buffer[i * 2];
            l_dV_dy[i * 2 + 1] = trans_buffer[i * 2 + 1];
        }
    } else {
        for (int64_t i = 0; i < local_size; ++i) {
            l_dV_dy[i] = trans_buffer[i * 2];
        }
    }
    //watch.Record(52);

#if 0 //def _DEBUG
    printf("[%d]after Transpose3-2: %d, %d\n", GetProcessID(m_ddm_comm));
    fflush(stdout);
    MPI_Barrier(m_ddm_comm);
#endif

    //z exchange////////////////////////////////////////////////////////
    //転置 x,y,z to z,x,y to //
    Transpose3_back<OneComplex>(local_size_x, local_size_y, local_size_z, (OneComplex*)&fft_buffer[0], (OneComplex*)&modified_buffer[0]);
    //watch.Record(52);

    // z方向に領域分割されたバッファをExchangeして、z方向に連続なバッファに並べ替える//    
    int my_line_xy = ddm_ex.ForwardExchangeZ(Nz/ HR_ratio_z, local_size_x * local_size_y, &fft_buffer[0], &fft_buffer[0], &trans_buffer[0], 2 * HR_ratio_z);//2 means complex
    //int my_line_begin_xy = DDM::GridBegin(local_size_x * local_size_y, ddm_ex.GetProccessID_Z(), split_z);
    //int hr_Nz = Nz * m_HR_ratio_z;
    //watch.Record(53);

    //such as fft////////////
    for (int l = 0; l < my_line_xy; ++l) {
        auto* line = (OneComplex*)&fft_buffer[l * Nz * 2];
        my_fft_z.ForwardDirect(line);
        for (int iz = 0; iz < Nz; ++iz) {
            const double kz = coef_z * (double)(iz * 2 > Nz ? iz - Nz : iz) / (double)Nz;
            const double re = line[iz].r;
            const double im = line[iz].i;
            line[iz].r = -im * kz;
            line[iz].i = re * kz;
        }
        my_fft_z.BackwardDirect(line);
    }
    //watch.Record(58);

    ddm_ex.BackwardExchangeZ(Nz/ HR_ratio_z, local_size_x * local_size_y, &fft_buffer[0], &fft_buffer[0], &trans_buffer[0], 2* HR_ratio_z);//2 means complex
    //watch.Record(51);
    //転置 z,x,y to x,y,z//
    Transpose3<OneComplex>(local_size_z, local_size_x, local_size_y, (OneComplex*)&trans_buffer[0], (OneComplex*)&fft_buffer[0]);
    if (is_complex) {
        for (int64_t i = 0; i < local_size; ++i) {
            l_dV_dz[i * 2] = trans_buffer[i * 2];
            l_dV_dz[i * 2 + 1] = trans_buffer[i * 2 + 1];
        }
    } else {
        for (int64_t i = 0; i < local_size; ++i) {
            l_dV_dz[i] = trans_buffer[i * 2];
        }
    }
    //watch.Record(52);


}

#pragma once
#include "qumasun_base1.h"

#ifdef DDM_FFT
/*
* DDM分割されたHRグリッドの電子密度l_hr_rhoを受けとり、
* Poisson方程式をDDM-FFTで解き、
* 同じくDDM分割されたHRグリッドのポテンシャルl_Vを返す.
* nullptr以外の値がセットされた場合は、l_hr_dVdx等にgradient Vを返す
*/
inline
void QUMASUN_BASE1::mPoissonDDMFFT_HR(const double* l_hr_rho, double* l_hr_Vout, double* l_hr_dVdx_out, double* l_hr_dVdy_out, double* l_hr_dVdz_out) {
    const bool is_root = IsRoot(m_mpi_comm);


    auto& m_pp_integrator = m_pp_SvF;



    auto& range = ml_grid;
    const int local_size_x = range.end_x - range.begin_x;
    const int local_size_y = range.end_y - range.begin_y;
    const int local_size_z = range.end_z - range.begin_z;
    const int& Nx = m_size_x;
    const int& Ny = m_size_y;
    const int& Nz = m_size_z;
    const int split_x = range.num_split_x;
    const int split_y = range.num_split_y;
    const int split_z = range.num_split_z;

    auto& ddm_ex = *m_ddm_exchanger;

    const int64_t local_size = local_size_x * local_size_y * local_size_z;
    const int64_t enough_size = std::max(ddm_ex.GetExcangeBufferSize(Nx, Ny, Nz), local_size) * m_HR_ratio_x * m_HR_ratio_y * m_HR_ratio_z;


    double* modified_buffer = m_work;
    double* fft_buffer = m_work + enough_size * 2;
    double* trans_buffer = m_work + enough_size * 2 * 2;


    //initial buffer set//
    const int64_t hr_local_size = local_size * m_HR_ratio_x * m_HR_ratio_y * m_HR_ratio_z;
    const size_t hr_size_3d = m_size_3d * m_HR_ratio_x * m_HR_ratio_y * m_HR_ratio_z;
    const double invN = 1.0 / (double)(hr_size_3d);

    for (int64_t i = 0; i < hr_local_size; ++i) {
        modified_buffer[i * 2] = l_hr_rho[i];
        modified_buffer[i * 2 + 1] = 0.0;
    }
    watch.Record(37);

    //x exchange////////////////////////////////////////////////////////
    // x方向に領域分割されたバッファをExchangeして、x方向に連続なバッファに並べ替える//
    int my_line_yz = ddm_ex.ForwardExchangeX(Nx, local_size_y * local_size_z * m_HR_ratio_y * m_HR_ratio_z, &modified_buffer[0], &fft_buffer[0], &trans_buffer[0], 2 * m_HR_ratio_x);//2 means complex
    const int hr_Nx = Nx * m_HR_ratio_x;
    watch.Record(33);

    //such as fft////////////

#ifdef USE_FFT_MANY
    m_HR_fft_many_x.ForwardDirect((OneComplex*)&fft_buffer[0]);
#else
    for (int l = 0; l < my_line_yz; ++l) {
        auto* line = (OneComplex*)&fft_buffer[l * hr_Nx * 2];
        m_HR_fft_x.ForwardDirect(line);
    }
#endif
    watch.Record(38);

    ddm_ex.BackwardExchangeX(Nx, local_size_y * local_size_z * m_HR_ratio_y * m_HR_ratio_z, &modified_buffer[0], &fft_buffer[0], &trans_buffer[0], 2 * m_HR_ratio_x);//2 means complex
    watch.Record(35);

#if 0//def _DEBUG
    printf("[%d]after BackwardExchangeX:\n", GetProcessID(m_ddm_comm));
    fflush(stdout);
    MPI_Barrier(m_ddm_comm);
#endif
    //転置 x,y,z to y,z,x//
    Transpose3<OneComplex>(local_size_x * m_HR_ratio_x, local_size_y * m_HR_ratio_y, local_size_z * m_HR_ratio_z, (OneComplex*)&fft_buffer[0], (OneComplex*)&modified_buffer[0]);
    watch.Record(32);

#if 0//def _DEBUG
    printf("[%d]after Transpose3-1: %d, %d\n", GetProcessID(m_ddm_comm));
    fflush(stdout);
    MPI_Barrier(m_ddm_comm);
#endif

    //y exchange////////////////////////////////////////////////////////
    // y方向に領域分割されたバッファをExchangeして、y方向に連続なバッファに並べ替える//    
    int my_line_xz = ddm_ex.ForwardExchangeY(Ny, local_size_x * local_size_z * m_HR_ratio_x * m_HR_ratio_z, &fft_buffer[0], &fft_buffer[0], &trans_buffer[0], 2 * m_HR_ratio_y);//2 means complex
    int hr_Ny = Ny * m_HR_ratio_y;
    watch.Record(33);
    //such as fft////////////

#ifdef USE_FFT_MANY
    m_HR_fft_many_y.ForwardDirect((OneComplex*)&fft_buffer[0]);
#else
    for (int l = 0; l < my_line_xz; ++l) {
        auto* line = (OneComplex*)&fft_buffer[l * hr_Ny * 2];
        m_HR_fft_y.ForwardDirect(line);
    }
#endif
    watch.Record(38);

    ddm_ex.BackwardExchangeY(Ny, local_size_x * local_size_z * m_HR_ratio_x * m_HR_ratio_z, &modified_buffer[0], &fft_buffer[0], &trans_buffer[0], 2 * m_HR_ratio_y);//2 means complex
    watch.Record(35);

    //転置 y,z,x to z,x,y//
    Transpose3<OneComplex>(local_size_y * m_HR_ratio_y, local_size_z * m_HR_ratio_z, local_size_x * m_HR_ratio_x, (OneComplex*)&fft_buffer[0], (OneComplex*)&modified_buffer[0]);
    watch.Record(32);

#if 0//def _DEBUG
    printf("[%d]after Transpose3-2: %d, %d\n", GetProcessID(m_ddm_comm));
    fflush(stdout);
    MPI_Barrier(m_ddm_comm);
#endif

    //z exchange////////////////////////////////////////////////////////
    // z方向に領域分割されたバッファをExchangeして、z方向に連続なバッファに並べ替える//    
    int my_line_xy = ddm_ex.ForwardExchangeZ(Nz, local_size_x * local_size_y * m_HR_ratio_x * m_HR_ratio_y, &fft_buffer[0], &fft_buffer[0], &trans_buffer[0], 2 * m_HR_ratio_z);//2 means complex
    int my_line_begin_xy = DDM::GridBegin(local_size_x * local_size_y * m_HR_ratio_x * m_HR_ratio_y, ddm_ex.GetProccessID_Z(), split_z);
    int hr_Nz = Nz * m_HR_ratio_z;
    watch.Record(33);

    //such as fft////////////
#ifdef USE_FFT_MANY
    m_HR_fft_many_z.ForwardDirect((OneComplex*)&fft_buffer[0]);
#else
    for (int l = 0; l < my_line_xy; ++l) {
        auto* line = (OneComplex*)&fft_buffer[l * hr_Nz * 2];
        m_HR_fft_z.ForwardDirect(line);
    }
#endif
    watch.Record(38);

#if 0//def _DEBUG
    printf("[%d]after fft-forward-3: %d, %d, %d, %d\n", GetProcessID(m_ddm_comm), my_line_xy, my_line_begin_xy, ml_grid.begin_x * m_HR_ratio_x, ml_grid.begin_y * m_HR_ratio_y);
    fflush(stdout);
    MPI_Barrier(m_ddm_comm);
#endif
    const double coef_x = 2.0 * M_PI / ((double)Nx * m_dx); //hr_Nx * (dx/m_HR_ratio_x) == Nx*dx//
    const double coef_y = 2.0 * M_PI / ((double)Ny * m_dy);
    const double coef_z = 2.0 * M_PI / ((double)Nz * m_dz);


    for (int l = 0; l < my_line_xy; ++l) {
        auto* line = (OneComplex*)&fft_buffer[l * hr_Nz * 2];

        const int ix = ((l + my_line_begin_xy) % (local_size_x * m_HR_ratio_x)) + ml_grid.begin_x * m_HR_ratio_x;
        const int iy = ((l + my_line_begin_xy) / (local_size_x * m_HR_ratio_x)) + ml_grid.begin_y * m_HR_ratio_y;
        const double kx = coef_x * (double)(ix * 2 > hr_Nx ? ix - hr_Nx : ix);
        const double ky = coef_y * (double)(iy * 2 > hr_Ny ? iy - hr_Ny : iy);

        if ((ix == 0) && (iy == 0)) {
            {
                const int iz = 0;
                line[iz].r = 0.0;
                line[iz].i = 0.0;
            }
            for (int iz = 1; iz < hr_Nz; ++iz) {
                const double kz = coef_z * (double)(iz * 2 > hr_Nz ? iz - hr_Nz : iz);
                line[iz].r *= 4.0 * M_PI / (kx * kx + ky * ky + kz * kz) * invN;
                line[iz].i *= 4.0 * M_PI / (kx * kx + ky * ky + kz * kz) * invN;
            }
        } else {
            for (int iz = 0; iz < hr_Nz; ++iz) {
                const double kz = coef_z * (double)(iz * 2 > hr_Nz ? iz - hr_Nz : iz);
                line[iz].r *= 4.0 * M_PI / (kx * kx + ky * ky + kz * kz) * invN;
                line[iz].i *= 4.0 * M_PI / (kx * kx + ky * ky + kz * kz) * invN;
            }
        }
    }
#if 0//def _DEBUG
    printf("[%d]after Poisson in k:\n", GetProcessID(m_ddm_comm));
    fflush(stdout);
    MPI_Barrier(m_ddm_comm);
#endif
    watch.Record(34);


#ifdef USE_FFT_MANY
    m_HR_fft_many_z.BackwardDirect((OneComplex*)&fft_buffer[0]);
#else
    for (int l = 0; l < my_line_xy; ++l) {
        auto* line = (OneComplex*)&fft_buffer[l * hr_Nz * 2];
        m_HR_fft_z.BackwardDirect(line);
    }
#endif
#if 0//def _DEBUG
    printf("[%d]after fft-backward-3: %d, %d\n", GetProcessID(m_ddm_comm));
    fflush(stdout);
    MPI_Barrier(m_ddm_comm);
#endif
    watch.Record(38);


    ddm_ex.BackwardExchangeZ(Nz, local_size_x * local_size_y * m_HR_ratio_x * m_HR_ratio_y, &modified_buffer[0], &fft_buffer[0], &trans_buffer[0], 2 * m_HR_ratio_z);//2 means complex
    watch.Record(35);
    //転置 z,x,y to y,z,x//
    Transpose3_back<OneComplex>(local_size_z * m_HR_ratio_z, local_size_x * m_HR_ratio_x, local_size_y * m_HR_ratio_y, (OneComplex*)&fft_buffer[0], (OneComplex*)&modified_buffer[0]);
    watch.Record(32);

#if 0//def _DEBUG
    printf("[%d]after Transpose3-3: %d, %d\n", GetProcessID(m_ddm_comm));
    fflush(stdout);
    MPI_Barrier(m_ddm_comm);
#endif




    //y exchange ////////////////////////////////////////////////////////
    // y方向に領域分割されたバッファをExchangeして、y方向に連続なバッファに並べ替える//            
    ddm_ex.ForwardExchangeY(Ny, local_size_x * local_size_z * m_HR_ratio_x * m_HR_ratio_z, &fft_buffer[0], &fft_buffer[0], &trans_buffer[0], 2 * m_HR_ratio_y);//2 means complex
    watch.Record(33);

#ifdef USE_FFT_MANY
    m_HR_fft_many_y.BackwardDirect((OneComplex*)&fft_buffer[0]);
#else
    for (int l = 0; l < my_line_xz; ++l) {
        auto* line = (OneComplex*)&fft_buffer[l * hr_Ny * 2];
        m_HR_fft_y.BackwardDirect(line);
    }
#endif
    watch.Record(38);

    ////////////such as fft//
    ddm_ex.BackwardExchangeY(Ny, local_size_x * local_size_z * m_HR_ratio_x * m_HR_ratio_z, &modified_buffer[0], &fft_buffer[0], &trans_buffer[0], 2 * m_HR_ratio_y);//2 means complex
    watch.Record(35);
    //転置 y,z,x to x,y,z//
    Transpose3_back<OneComplex>(local_size_y * m_HR_ratio_y, local_size_z * m_HR_ratio_z, local_size_x * m_HR_ratio_x, (OneComplex*)&fft_buffer[0], (OneComplex*)&modified_buffer[0]);
    watch.Record(32);

#if 0//def _DEBUG
    printf("[%d]after Transpose3-5: %d, %d\n", GetProcessID(m_ddm_comm));
    fflush(stdout);
    MPI_Barrier(m_ddm_comm);
#endif

    //x exchange////////////////////////////////////////////////////////
    // x方向に領域分割されたバッファをExchangeして、x方向に連続なバッファに並べ替える//
    ddm_ex.ForwardExchangeX(Nx, local_size_y * local_size_z * m_HR_ratio_y * m_HR_ratio_z, &fft_buffer[0], &fft_buffer[0], &trans_buffer[0], 2 * m_HR_ratio_x);//2 means complex
    watch.Record(33);


    if (l_hr_dVdx_out != nullptr) {
        //x微分//

#ifdef USE_FFT_MANY
        for (int l = 0; l < my_line_yz; ++l) {
            auto* line = (OneComplex*)&fft_buffer[l * hr_Nx * 2];
            auto* l_dVdx = (OneComplex*)&modified_buffer[l * hr_Nx * 2];
            for (int ix = 0; ix < hr_Nx; ++ix) {
                const double kx = coef_x * (double)(ix * 2 > hr_Nx ? ix - hr_Nx : ix);
                l_dVdx[ix].r = -line[ix].i * kx;
                l_dVdx[ix].i = line[ix].r * kx;
            }
        }
        m_HR_fft_many_x.BackwardDirect((OneComplex*)&modified_buffer[0]);
#else
        for (int l = 0; l < my_line_yz; ++l) {
            auto* line = (OneComplex*)&fft_buffer[l * hr_Nx * 2];
            auto* l_dVdx = (OneComplex*)&modified_buffer[l * hr_Nx * 2];
            for (int ix = 0; ix < hr_Nx; ++ix) {
                const double kx = coef_x * (double)(ix * 2 > hr_Nx ? ix - hr_Nx : ix);
                l_dVdx[ix].r = -line[ix].i * kx;
                l_dVdx[ix].i = line[ix].r * kx;
            }
            m_HR_fft_x.BackwardDirect(l_dVdx);
        }
#endif
        watch.Record(34);
        ddm_ex.BackwardExchangeX(Nx, local_size_y * local_size_z * m_HR_ratio_y * m_HR_ratio_z, &modified_buffer[0], &modified_buffer[0], &trans_buffer[0], 2 * m_HR_ratio_x);//2 means complex
        watch.Record(35);

        for (int64_t i = 0; i < hr_local_size; ++i) {
            l_hr_dVdx_out[i] = modified_buffer[i * 2];
        }
        watch.Record(32);

    }
   

    //引き続きVをkx空間から実空間に戻す//

#ifdef USE_FFT_MANY
    m_HR_fft_many_x.BackwardDirect((OneComplex*)&fft_buffer[0]);
#else
    for (int l = 0; l < my_line_yz; ++l) {
        auto* line = (OneComplex*)&fft_buffer[l * hr_Nx * 2];
        m_HR_fft_x.BackwardDirect(line);
    }
#endif
    watch.Record(38);
    ddm_ex.BackwardExchangeX(Nx, local_size_y * local_size_z * m_HR_ratio_y * m_HR_ratio_z, &modified_buffer[0], &fft_buffer[0], &trans_buffer[0], 2 * m_HR_ratio_x);//2 means complex
    watch.Record(35);

#if 0//def _DEBUG
    printf("[%d]after Transpose3-7: %d, %d\n", GetProcessID(m_ddm_comm));
    fflush(stdout);
    MPI_Barrier(m_ddm_comm);
#endif


    for (int64_t i = 0; i < hr_local_size; ++i) {
        l_hr_Vout[i] = modified_buffer[i * 2];
    }
    watch.Record(32);



    if (l_hr_dVdy_out != nullptr) {
        //y微分/////////////////////////////////////////////////////////////
        //転置 x,y,z to y,z,x //
        Transpose3<OneComplex>(local_size_x * m_HR_ratio_x, local_size_y * m_HR_ratio_y, local_size_z * m_HR_ratio_z, (OneComplex*)&fft_buffer[0], (OneComplex*)&modified_buffer[0]);
        watch.Record(32);
        // y方向に領域分割されたバッファをExchangeして、y方向に連続なバッファに並べ替える//            
        ddm_ex.ForwardExchangeY(Ny, local_size_x * local_size_z * m_HR_ratio_x * m_HR_ratio_z, &fft_buffer[0], &fft_buffer[0], &trans_buffer[0], 2 * m_HR_ratio_y);//2 means complex
        watch.Record(33);

#ifdef USE_FFT_MANY
        m_HR_fft_many_y.ForwardDirect((OneComplex*)&fft_buffer[0]);
#else
        for (int l = 0; l < my_line_xz; ++l) {
            auto* line = (OneComplex*)&fft_buffer[l * hr_Ny * 2];
            m_HR_fft_y.ForwardDirect(line);
        }
#endif
        watch.Record(38);
        for (int l = 0; l < my_line_xz; ++l) {
            auto* line = (OneComplex*)&fft_buffer[l * hr_Ny * 2];
            //auto* l_dVdy = (fftw_complex*)&dVdy_buffer[l * hr_Ny * 2];
            for (int iy = 0; iy < hr_Ny; ++iy) {
                const double ky = coef_y * (double)(iy * 2 > hr_Ny ? iy - hr_Ny : iy);
                double re = line[iy].r;
                double im = line[iy].i;
                line[iy].r = -im * ky;
                line[iy].i = re * ky;
            }
        }
        watch.Record(34);

#ifdef USE_FFT_MANY
        m_HR_fft_many_y.BackwardDirect((OneComplex*)&fft_buffer[0]);
#else
        for (int l = 0; l < my_line_xz; ++l) {
            auto* line = (OneComplex*)&fft_buffer[l * hr_Ny * 2];
            m_HR_fft_y.BackwardDirect(line);
        }
#endif
        watch.Record(38);
        ddm_ex.BackwardExchangeY(Ny, local_size_x * local_size_z * m_HR_ratio_x * m_HR_ratio_z, &fft_buffer[0], &fft_buffer[0], &trans_buffer[0], 2 * m_HR_ratio_y);//2 means complex
        watch.Record(35);
        //転置 y,z,x to x,y,z//
        Transpose3_back<OneComplex>(local_size_y * m_HR_ratio_y, local_size_z * m_HR_ratio_z, local_size_x * m_HR_ratio_x, (OneComplex*)&trans_buffer[0], (OneComplex*)&fft_buffer[0]);


        const double invNY = 1.0 / (double)hr_Ny;
        for (int64_t i = 0; i < hr_local_size; ++i) {
            l_hr_dVdy_out[i] = trans_buffer[i * 2] * invNY;
        }
        watch.Record(32);

    }

    if (l_hr_dVdz_out != nullptr) {
        //z微分/////////////////////////////////////////////////////////////
        //転置 x,y,z to z,x,y //
        Transpose3_back<OneComplex>(local_size_x * m_HR_ratio_x, local_size_y * m_HR_ratio_y, local_size_z * m_HR_ratio_z, (OneComplex*)&fft_buffer[0], (OneComplex*)&modified_buffer[0]);
        watch.Record(32);
        // z方向に領域分割されたバッファをExchangeして、z方向に連続なバッファに並べ替える//            
        ddm_ex.ForwardExchangeZ(Nz, local_size_x * local_size_y * m_HR_ratio_x * m_HR_ratio_y, &fft_buffer[0], &fft_buffer[0], &trans_buffer[0], 2 * m_HR_ratio_z);//2 means complex
        watch.Record(33);

#ifdef USE_FFT_MANY
        m_HR_fft_many_z.ForwardDirect((OneComplex*)&fft_buffer[0]);
#else
        for (int l = 0; l < my_line_xy; ++l) {
            auto* line = (OneComplex*)&fft_buffer[l * hr_Nz * 2];
            m_HR_fft_z.ForwardDirect(line);
        }
#endif
        watch.Record(38);
        for (int l = 0; l < my_line_xy; ++l) {
            auto* line = (OneComplex*)&fft_buffer[l * hr_Nz * 2];
            //auto* l_dVdy = (fftw_complex*)&dVdy_buffer[l * hr_Ny * 2];
            for (int iz = 0; iz < hr_Nz; ++iz) {
                const double kz = coef_z * (double)(iz * 2 > hr_Nz ? iz - hr_Nz : iz);
                double re = line[iz].r;
                double im = line[iz].i;
                line[iz].r = -im * kz;
                line[iz].i = re * kz;
            }
        }
        watch.Record(34);


#ifdef USE_FFT_MANY
        m_HR_fft_many_z.BackwardDirect((OneComplex*)&fft_buffer[0]);
#else
        for (int l = 0; l < my_line_xy; ++l) {
            auto* line = (OneComplex*)&fft_buffer[l * hr_Nz * 2];
            m_HR_fft_z.BackwardDirect(line);
        }
#endif
        watch.Record(38);
        ddm_ex.BackwardExchangeZ(Nz, local_size_x * local_size_y * m_HR_ratio_x * m_HR_ratio_y, &fft_buffer[0], &fft_buffer[0], &trans_buffer[0], 2 * m_HR_ratio_z);//2 means complex
        watch.Record(35);
        //転置 z,x,y to x,y,z//
        Transpose3<OneComplex>(local_size_z * m_HR_ratio_z, local_size_x * m_HR_ratio_x, local_size_y * m_HR_ratio_y, (OneComplex*)&trans_buffer[0], (OneComplex*)&fft_buffer[0]);

        const double invNZ = 1.0 / (double)hr_Nz;
        for (int64_t i = 0; i < hr_local_size; ++i) {
            l_hr_dVdz_out[i] = trans_buffer[i * 2] * invNZ;
        }
        watch.Record(32);

    }


}

#endif

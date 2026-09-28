#pragma once
#include "qumasun_base1.h"
#include "shift_via_fft.h"

#if 0 //def DDM_FFT


/*
* DDM分割されたHRグリッドの場l_hr_Vを受けとり、
* DownConvertして、通常グリッドのポテンシャルl_Vを返す.
* (1次元ずつ、SvF_interpolationを利用して重ねていく
*/
inline
void QUMASUN_BASE1::mDDMFFT_Downconvert_SvF(const double* l_hr_V, double* l_V) {
    const bool is_root = IsRoot(m_mpi_comm);



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
    const int hr_Nx = Nx * m_HR_ratio_x;
    const int hr_Ny = Ny * m_HR_ratio_y;
    const int hr_Nz = Nz * m_HR_ratio_z;

    const int hr_dx = m_dx / (double)m_HR_ratio_x;
    const int hr_dy = m_dy / (double)m_HR_ratio_y;
    const int hr_dz = m_dz / (double)m_HR_ratio_z;


    auto& ddm_ex = *m_ddm_exchanger;

    const int64_t local_size = local_size_x * local_size_y * local_size_z;
    const int64_t enough_size = std::max(ddm_ex.GetExcangeBufferSize(Nx, Ny, Nz), local_size) * m_HR_ratio_x * m_HR_ratio_y * m_HR_ratio_z;

    double* modified_buffer = m_work;
    double* fft_buffer = m_work + enough_size;
    double* trans_buffer = m_work + enough_size * 2;
    auto buf1d = std::make_unique<OneComplex[]>(std::max(std::max(m_size_x, m_size_y), m_size_z));

    //initial buffer set//
    const int64_t hr_local_size = local_size * m_HR_ratio_x * m_HR_ratio_y * m_HR_ratio_z;
    const size_t hr_size_3d = m_size_3d * m_HR_ratio_x * m_HR_ratio_y * m_HR_ratio_z;
    const double invN = 1.0 / (double)(m_size_3d);
    const double invN_HR = 1.0 / (double)(hr_size_3d);
   

    for (int64_t i = 0; i < hr_local_size; ++i) {
        fft_buffer[i] = l_hr_V[i];
    }
    watch.Record(57);

    //x exchange////////////////////////////////////////////////////////
    // x方向に領域分割されたバッファをExchangeして、x方向に連続なバッファに並べ替える//
    int my_line_yz = ddm_ex.ForwardExchangeX(Nx, local_size_y * m_HR_ratio_y * local_size_z * m_HR_ratio_z, &fft_buffer[0], &fft_buffer[0], &trans_buffer[0], m_HR_ratio_x);//2 means complex
    watch.Record(53);

    double w_x = 1.0 / (double)m_HR_ratio_x;
    const double invNx_w = 1.0 / (double)(Nx* m_HR_ratio_x);
    //such as fft////////////
    for (int l = 0; l < my_line_yz; ++l) {
        auto* hr_line = &fft_buffer[l * hr_Nx];
        auto* line = &modified_buffer[l * Nx];
        {
            for (int i = 0; i < Nx; ++i) {
                line[i] = 0.0;
            }
        }
        for (int di = 0; di < m_HR_ratio_x; ++di) {
            for (int i = 0; i < Nx; ++i) {
                buf1d[i].r = hr_line[i * m_HR_ratio_x + di];
                buf1d[i].i = 0.0;
            }
            m_fft_x.ForwardDirect(&buf1d[0]);
            if (di > 0) {
                ShiftInKspace_1d(buf1d.get(), hr_dx * (double)di, Nx, m_dx);
            }
            if ((Nx & 0x1) == 0) {
                buf1d[Nx / 2].r *= 2.0;
                buf1d[Nx / 2].i = 0.0;
            }
            m_fft_x.BackwardDirect(&buf1d[0]);
            for (int i = 0; i < Nx; ++i) {
                line[i] += buf1d[i].r * invNx_w;
            }
        }
    }
    watch.Record(58);
    ddm_ex.ScaleDown(m_HR_ratio_x);
    ddm_ex.BackwardExchangeX(Nx, local_size_y * m_HR_ratio_y * local_size_z * m_HR_ratio_z, &modified_buffer[0], &modified_buffer[0], &trans_buffer[0], 1);//2 means complex
    watch.Record(51);

    //転置 x,y,z to y,z,x//
    Transpose3<double>(local_size_x, local_size_y * m_HR_ratio_y, local_size_z * m_HR_ratio_z, (double*)&fft_buffer[0], (double*)&modified_buffer[0]);
    watch.Record(52);


    //y exchange////////////////////////////////////////////////////////
    // y方向に領域分割されたバッファをExchangeして、y方向に連続なバッファに並べ替える//    
    int my_line_xz = ddm_ex.ForwardExchangeY(Ny, local_size_x * local_size_z * m_HR_ratio_z, &fft_buffer[0], &fft_buffer[0], &trans_buffer[0], m_HR_ratio_y);//2 means complex
    //int hr_Ny = Ny * m_HR_ratio_y;
    watch.Record(53);

    //such as fft////////////
    double w_y = 1.0 / (double)m_HR_ratio_y;
    const double invNy_w = 1.0 / (double)(Ny * m_HR_ratio_y);
    for (int l = 0; l < my_line_xz; ++l) {
        auto* hr_line = (double*)&fft_buffer[l * hr_Ny];
        auto* line = &modified_buffer[l * Ny];
        {
            for (int i = 0; i < Ny; ++i) {
                line[i] = 0.0;
            }
        }
        for (int di = 0; di < m_HR_ratio_y; ++di) {
            for (int i = 0; i < Ny; ++i) {
                buf1d[i].r = hr_line[i * m_HR_ratio_y + di];
                buf1d[i].i = 0.0;
            }
            m_fft_y.ForwardDirect(&buf1d[0]);
            if (di > 0) {
                ShiftInKspace_1d(buf1d.get(), hr_dy * (double)di, Ny, m_dy);
            }
            if ((Ny & 0x1) == 0) {
                buf1d[Ny / 2].r *= 2.0;
                buf1d[Ny / 2].i = 0.0;
            }
            m_fft_y.BackwardDirect(&buf1d[0]);
            for (int i = 0; i < Ny; ++i) {
                line[i] += buf1d[i].r * invNy_w;
            }
        }        
    }
    watch.Record(58);
    ddm_ex.ScaleDown(m_HR_ratio_y);
    ddm_ex.BackwardExchangeY(Ny, local_size_x * local_size_z * m_HR_ratio_z, &modified_buffer[0], &modified_buffer[0], &trans_buffer[0], 1);//2 means complex
    watch.Record(51);

    //転置 y,z,x to z,x,y//
    Transpose3<double>(local_size_y, local_size_z * m_HR_ratio_z, local_size_x, (double*)&fft_buffer[0], (double*)&modified_buffer[0]);
    watch.Record(52);



    //z exchange////////////////////////////////////////////////////////
    // z方向に領域分割されたバッファをExchangeして、z方向に連続なバッファに並べ替える//    
    int my_line_xy = ddm_ex.ForwardExchangeZ(Nz, local_size_x * local_size_y, &fft_buffer[0], &fft_buffer[0], &trans_buffer[0], m_HR_ratio_z);//2 means complex

    watch.Record(53);

    //such as fft////////////
    double w_z = 1.0 / (double)m_HR_ratio_z;
    const double invNz_w = 1.0 / (double)(Nz * m_HR_ratio_z);
    for (int l = 0; l < my_line_xy; ++l) {
        auto* hr_line = (double*)&fft_buffer[l * hr_Nz];
        auto* line = &modified_buffer[l * Nz];
        {
            for (int i = 0; i < Nz; ++i) {
                line[i] = 0.0;
            }
        }
        for (int di = 0; di < m_HR_ratio_z; ++di) {
            for (int i = 0; i < Nz; ++i) {
                buf1d[i].r = hr_line[i * m_HR_ratio_z + di];
                buf1d[i].i = 0.0;
            }
            m_fft_z.ForwardDirect(&buf1d[0]);
            if (di > 0) {
                ShiftInKspace_1d(buf1d.get(), hr_dz * (double)di, Nz, m_dz);
            }
            if ((Nz & 0x1) == 0) {
                buf1d[Nz / 2].r *= 2.0;
                buf1d[Nz / 2].i = 0.0;
            }
            m_fft_z.BackwardDirect(&buf1d[0]);
            for (int i = 0; i < Nz; ++i) {
                line[i] += buf1d[i].r * invNz_w;
            }
        }
    }
    watch.Record(58);
    ddm_ex.ScaleDown(m_HR_ratio_z);
    ddm_ex.BackwardExchangeZ(Nz, local_size_x * local_size_y, &modified_buffer[0], &modified_buffer[0], &trans_buffer[0], 1);//2 means complex
    watch.Record(51);
    
    //転置 z,x,y to x,y,z//
    Transpose3<double>(local_size_z, local_size_x, local_size_y, (double*)&fft_buffer[0], (double*)&modified_buffer[0]);
    watch.Record(52);


    for (int64_t i = 0; i < local_size; ++i) {
        l_V[i] = fft_buffer[i];
    }
    watch.Record(57);


}


#endif

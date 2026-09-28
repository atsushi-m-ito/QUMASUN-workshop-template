#pragma once
#include "qumasun_base1.h"

#ifdef DDM_FFT



/*
* DDM分割されたHRグリッドの場l_hr_Vを受けとり、
* DownConvertして、通常グリッドのポテンシャルl_Vを返す.
* (3次元を全て波数空間にしてから、1次元ずつDownConvertする場合)
*/
inline
void QUMASUN_BASE1::mDDMFFT_Downconvert(const double* l_hr_V, double* l_V) {
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

    auto& ddm_ex = *m_ddm_exchanger;

    const int64_t local_size = local_size_x * local_size_y * local_size_z;
    const int64_t enough_size = std::max(ddm_ex.GetExcangeBufferSize(Nx, Ny, Nz), local_size) * m_HR_ratio_x * m_HR_ratio_y * m_HR_ratio_z;

    double* modified_buffer = m_work;
    double* fft_buffer = m_work + enough_size * 2;
    double* trans_buffer = m_work + enough_size * 2 * 2;
    

    //initial buffer set//
    const int64_t hr_local_size = local_size * m_HR_ratio_x * m_HR_ratio_y * m_HR_ratio_z;
    const size_t hr_size_3d = m_size_3d * m_HR_ratio_x * m_HR_ratio_y * m_HR_ratio_z;
    const double invN = 1.0 / (double)(m_size_3d);
    const double invN_HR = 1.0 / (double)(hr_size_3d);

    for (int64_t i = 0; i < hr_local_size; ++i) {
        fft_buffer[i * 2] = l_hr_V[i];
        fft_buffer[i * 2 + 1] = 0.0;
    }
    watch.Record(57);

    //x exchange////////////////////////////////////////////////////////
    // x方向に領域分割されたバッファをExchangeして、x方向に連続なバッファに並べ替える//
    int my_line_yz = ddm_ex.ForwardExchangeX(Nx, local_size_y * m_HR_ratio_y * local_size_z * m_HR_ratio_z, &fft_buffer[0], &fft_buffer[0], &trans_buffer[0], 2 * m_HR_ratio_x);//2 means complex
    watch.Record(53);

    //such as fft////////////

#ifdef USE_FFT_MANY
    m_HR_fft_many_x.ForwardDirect((OneComplex*)&fft_buffer[0]);
#else
    for (int l = 0; l < my_line_yz; ++l) {
        auto* hr_line = (OneComplex*)&fft_buffer[l * hr_Nx * 2];
        m_HR_fft_x.ForwardDirect(hr_line);
    }
#endif
    watch.Record(58);

    ddm_ex.BackwardExchangeX(Nx, local_size_y * m_HR_ratio_y * local_size_z * m_HR_ratio_z, &modified_buffer[0], &fft_buffer[0], &trans_buffer[0], 2 * m_HR_ratio_x);//2 means complex
    watch.Record(51);

    //転置 x,y,z to y,z,x//
    Transpose3<OneComplex>(local_size_x * m_HR_ratio_x, local_size_y * m_HR_ratio_y, local_size_z * m_HR_ratio_z, (OneComplex*)&fft_buffer[0], (OneComplex*)&modified_buffer[0]);
    watch.Record(52);


    //y exchange////////////////////////////////////////////////////////
    // y方向に領域分割されたバッファをExchangeして、y方向に連続なバッファに並べ替える//    
    int my_line_xz = ddm_ex.ForwardExchangeY(Ny, local_size_x * m_HR_ratio_x * local_size_z * m_HR_ratio_z, &fft_buffer[0], &fft_buffer[0], &trans_buffer[0], 2 * m_HR_ratio_y);//2 means complex
    //int hr_Ny = Ny * m_HR_ratio_y;
    watch.Record(53);
    //such as fft////////////

#ifdef USE_FFT_MANY
    m_HR_fft_many_y.ForwardDirect((OneComplex*)&fft_buffer[0]);
#else
    for (int l = 0; l < my_line_xz; ++l) {
        auto* hr_line = (OneComplex*)&fft_buffer[l * hr_Ny * 2];
        m_HR_fft_y.ForwardDirect(hr_line);
    }
#endif
    watch.Record(58);

    ddm_ex.BackwardExchangeY(Ny, local_size_x * m_HR_ratio_x * local_size_z * m_HR_ratio_z, &modified_buffer[0], &fft_buffer[0], &trans_buffer[0], 2 * m_HR_ratio_y);//2 means complex
    watch.Record(51);

    //転置 y,z,x to z,x,y//
    Transpose3<OneComplex>(local_size_y * m_HR_ratio_y, local_size_z * m_HR_ratio_z, local_size_x * m_HR_ratio_x, (OneComplex*)&fft_buffer[0], (OneComplex*)&modified_buffer[0]);
    watch.Record(52);



    //z exchange////////////////////////////////////////////////////////
    // z方向に領域分割されたバッファをExchangeして、z方向に連続なバッファに並べ替える//    
    int my_line_xy = ddm_ex.ForwardExchangeZ(Nz, local_size_x * m_HR_ratio_x * local_size_y * m_HR_ratio_y, &fft_buffer[0], &fft_buffer[0], &trans_buffer[0], 2 * m_HR_ratio_z);//2 means complex
    //int my_line_begin_xy = DDM::GridBegin(local_size_x * local_size_y, ddm_ex.GetProccessID_Z(), split_z);

    watch.Record(53);

    //such as fft////////////

#ifdef USE_FFT_MANY
    m_HR_fft_many_z.ForwardDirect((OneComplex*)&fft_buffer[0]);
    for (int l = 0; l < my_line_xy; ++l) {
        auto* hr_line = (OneComplex*)&fft_buffer[l * hr_Nz * 2];
        auto* line = (OneComplex*)&modified_buffer[l * Nz * 2];        
        DownConvert_Kspace_1d(hr_line, Nz, line, m_HR_ratio_z);
    }
    m_fft_many_z.BackwardDirect((OneComplex*)&modified_buffer[0]);
#else
    for (int l = 0; l < my_line_xy; ++l) {
        auto* hr_line = (OneComplex*)&fft_buffer[l * hr_Nz * 2];
        auto* line = (OneComplex*)&modified_buffer[l * Nz * 2];
        m_HR_fft_z.ForwardDirect(hr_line);
        DownConvert_Kspace_1d(hr_line, Nz, line, m_HR_ratio_z);
        m_fft_z.BackwardDirect(line);
    }
#endif
    watch.Record(58);

    ddm_ex.ScaleDown(m_HR_ratio_z);
    ddm_ex.BackwardExchangeZ(Nz, local_size_x * m_HR_ratio_x * local_size_y * m_HR_ratio_y, &modified_buffer[0], &modified_buffer[0], &trans_buffer[0], 2);//2 means complex
    watch.Record(51);
    
    //転置 z,x,y to y,z,x//
    Transpose3_back<OneComplex>(local_size_z, local_size_x * m_HR_ratio_x, local_size_y * m_HR_ratio_y, (OneComplex*)&fft_buffer[0], (OneComplex*)&modified_buffer[0]);
    watch.Record(52);


    //y exchange////////////////////////////////////////////////////////
    // y方向に領域分割されたバッファをExchangeして、y方向に連続なバッファに並べ替える//    
    my_line_xz = ddm_ex.ForwardExchangeY(Ny, local_size_x * m_HR_ratio_x * local_size_z, &fft_buffer[0], &fft_buffer[0], &trans_buffer[0], 2 * m_HR_ratio_y);//2 means complex
    watch.Record(53);
    //such as fft////////////
#ifdef USE_FFT_MANY    
    for (int l = 0; l < my_line_xz; ++l) {
        auto* hr_line = (OneComplex*)&fft_buffer[l * hr_Ny * 2];
        auto* line = (OneComplex*)&modified_buffer[l * Ny * 2];
        DownConvert_Kspace_1d(hr_line, Ny, line, m_HR_ratio_y);        
    }
    m_fft_many_y.BackwardDirect((OneComplex*)&modified_buffer[0]);
#else
    for (int l = 0; l < my_line_xz; ++l) {
        auto* hr_line = (OneComplex*)&fft_buffer[l * hr_Ny * 2];
        auto* line = (OneComplex*)&modified_buffer[l * Ny * 2];
        DownConvert_Kspace_1d(hr_line, Ny, line, m_HR_ratio_y);
        m_fft_y.BackwardDirect(line);
    }
#endif
    watch.Record(58);

    ddm_ex.ScaleDown(m_HR_ratio_y);
    ddm_ex.BackwardExchangeY(Ny, local_size_x * m_HR_ratio_x * local_size_z, &modified_buffer[0], &modified_buffer[0], &trans_buffer[0], 2);//2 means complex
    watch.Record(51);

    //転置 y,z,x to x,y,z//
    Transpose3_back<OneComplex>(local_size_y, local_size_z, local_size_x * m_HR_ratio_x, (OneComplex*)&fft_buffer[0], (OneComplex*)&modified_buffer[0]);
    watch.Record(52);

   

    //x exchange////////////////////////////////////////////////////////
    // x方向に領域分割されたバッファをExchangeして、x方向に連続なバッファに並べ替える//
    my_line_yz = ddm_ex.ForwardExchangeX(Nx, local_size_y * local_size_z, &fft_buffer[0], &fft_buffer[0], &trans_buffer[0], 2 * m_HR_ratio_x);//2 means complex
    watch.Record(53);

    //such as fft////////////
#ifdef USE_FFT_MANY
    for (int l = 0; l < my_line_yz; ++l) {
        auto* hr_line = (OneComplex*)&fft_buffer[l * hr_Nx * 2];
        auto* line = (OneComplex*)&modified_buffer[l * Nx * 2];
        DownConvert_Kspace_1d(hr_line, Nx, line, m_HR_ratio_x);       
    }
    m_fft_many_x.BackwardDirect((OneComplex*)&modified_buffer[0]);
#else
    for (int l = 0; l < my_line_yz; ++l) {
        auto* hr_line = (OneComplex*)&fft_buffer[l * hr_Nx * 2];
        auto* line = (OneComplex*)&modified_buffer[l * Nx * 2];
        DownConvert_Kspace_1d(hr_line, Nx, line, m_HR_ratio_x);
        m_fft_x.BackwardDirect(line);

    }
#endif
    watch.Record(58);

    ddm_ex.ScaleDown(m_HR_ratio_x);
    ddm_ex.BackwardExchangeX(Nx, local_size_y* local_size_z, &modified_buffer[0], &modified_buffer[0], &trans_buffer[0], 2);//2 means complex
    watch.Record(51);


    for (int64_t i = 0; i < local_size; ++i) {
        l_V[i] = modified_buffer[i * 2] * invN_HR;
    }
    watch.Record(57);


}


#endif

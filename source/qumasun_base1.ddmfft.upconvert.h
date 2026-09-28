#pragma once
#include "qumasun_base1.h"

#ifdef DDM_FFT



/*
* DDM分割された通常グリッド(non-HRグリッド)の電子密度l_hr_rhoを受けとり、
* Poisson方程式をDDM-FFTで解き、
* 同じくDDM分割されたポテンシャルl_Vを返す.
* また、Poisson方程式を解いたところでUpconvertして、HRグリッドのポテンシャルl_hr_Vを返す.
* nullptr以外の値がセットされた場合は、l_hr_dVdx等にHRグリッドのポテンシャルのgradientを返す
*/
inline
void QUMASUN_BASE1::mDDMFFT_Upconvert(const double* l_rho, double* l_hr_rho) {
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
    const int& hr_Nx = m_size_x * m_HR_ratio_x;
    const int& hr_Ny = m_size_y * m_HR_ratio_y;
    const int& hr_Nz = m_size_z * m_HR_ratio_z;

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

    for (int64_t i = 0; i < local_size; ++i) {
        modified_buffer[i * 2] = l_rho[i];
        modified_buffer[i * 2 + 1] = 0.0;
    }
    watch.Record(57);

    //x exchange////////////////////////////////////////////////////////
    // x方向に領域分割されたバッファをExchangeして、x方向に連続なバッファに並べ替える//
    int my_line_yz = ddm_ex.ForwardExchangeX(Nx, local_size_y * local_size_z, &modified_buffer[0], &modified_buffer[0], &trans_buffer[0], 2);//2 means complex
    //const int hr_Nx = Nx * m_HR_ratio_x;
    watch.Record(53);

    //such as fft////////////
#ifdef USE_FFT_MANY    
    m_fft_many_x.ForwardDirect((OneComplex*)&modified_buffer[0]);
#else
    for (int l = 0; l < my_line_yz; ++l) {
        auto* line = (OneComplex*)&modified_buffer[l * Nx * 2];;
        m_fft_x.ForwardDirect(line);
    }
#endif
    watch.Record(58);

    
    ddm_ex.BackwardExchangeX(Nx, local_size_y * local_size_z, &fft_buffer[0], &modified_buffer[0], &trans_buffer[0], 2 );//2 means complex
    watch.Record(51);


    //転置 x,y,z to y,z,x//
    Transpose3<OneComplex>(local_size_x, local_size_y, local_size_z, (OneComplex*)&modified_buffer[0], (OneComplex*)&fft_buffer[0]);
    watch.Record(52);

#if 0//def _DEBUG
    printf("[%d]after Transpose3-1: %d, %d\n", GetProcessID(m_ddm_comm));
    fflush(stdout);
    MPI_Barrier(m_ddm_comm);
#endif

    //y exchange////////////////////////////////////////////////////////
    // y方向に領域分割されたバッファをExchangeして、y方向に連続なバッファに並べ替える//    
    int my_line_xz = ddm_ex.ForwardExchangeY(Ny, local_size_x * local_size_z, &modified_buffer[0], &modified_buffer[0], &trans_buffer[0], 2);//2 means complex
    //int hr_Ny = Ny * m_HR_ratio_y;
    watch.Record(53);
    //such as fft////////////
#ifdef USE_FFT_MANY    
    m_fft_many_y.ForwardDirect((OneComplex*)&modified_buffer[0]);
#else
    for (int l = 0; l < my_line_xz; ++l) {
        auto* line = (OneComplex*)&modified_buffer[l * Ny * 2];
        m_fft_y.ForwardDirect(line);
    }
#endif
    watch.Record(58);


    ddm_ex.BackwardExchangeY(Ny, local_size_x * local_size_z, &fft_buffer[0], &modified_buffer[0], &trans_buffer[0], 2 );//2 means complex
    watch.Record(51);

    //転置 y,z,x to z,x,y//
    Transpose3<OneComplex>(local_size_y, local_size_z, local_size_x, (OneComplex*)&modified_buffer[0], (OneComplex*)&fft_buffer[0]);
    watch.Record(52);

#if 0//def _DEBUG
    printf("[%d]after Transpose3-2: %d, %d\n", GetProcessID(m_ddm_comm));
    fflush(stdout);
    MPI_Barrier(m_ddm_comm);
#endif

    //z exchange////////////////////////////////////////////////////////
    // z方向に領域分割されたバッファをExchangeして、z方向に連続なバッファに並べ替える//    
    int my_line_xy = ddm_ex.ForwardExchangeZ(Nz, local_size_x * local_size_y, &modified_buffer[0], &modified_buffer[0], &trans_buffer[0], 2);//2 means complex
    //int my_line_begin_xy = DDM::GridBegin(local_size_x * local_size_y, ddm_ex.GetProccessID_Z(), split_z);
    watch.Record(53);

    //such as fft////////////

#ifdef USE_FFT_MANY
    m_fft_many_z.ForwardDirect((OneComplex*)&modified_buffer[0]);
    for (int l = 0; l < my_line_xy; ++l) {
        auto* line = (OneComplex*)&modified_buffer[l * Nz * 2];
        auto* line_HR = (OneComplex*)&fft_buffer[l * hr_Nz * 2];
        UpConvert_Kspace_1d(line, Nz, line_HR, m_HR_ratio_z);        
    }
    m_HR_fft_many_z.BackwardDirect((OneComplex*)&fft_buffer[0]);
#else
    for (int l = 0; l < my_line_xy; ++l) {
        auto* line = (OneComplex*)&modified_buffer[l * Nz * 2];
        auto* line_HR = (OneComplex*)&fft_buffer[l * hr_Nz * 2];
        m_fft_z.ForwardDirect(line);
        UpConvert_Kspace_1d(line, Nz, line_HR, m_HR_ratio_z);
        m_HR_fft_z.BackwardDirect(line_HR);
    }
#endif
    watch.Record(58);


    ddm_ex.ScaleUp(m_HR_ratio_z);
    ddm_ex.BackwardExchangeZ(Nz, local_size_x * local_size_y, &fft_buffer[0], &fft_buffer[0], &trans_buffer[0], 2 * m_HR_ratio_z);//2 means complex
    watch.Record(51);

    //転置 z,x,y to y,z,x//
    Transpose3_back<OneComplex>(local_size_z * m_HR_ratio_z, local_size_x, local_size_y, (OneComplex*)&modified_buffer[0], (OneComplex*)&fft_buffer[0]);
    watch.Record(52);


    //y exchange////////////////////////////////////////////////////////
    // y方向に領域分割されたバッファをExchangeして、y方向に連続なバッファに並べ替える//    
    my_line_xz = ddm_ex.ForwardExchangeY(Ny, local_size_x * local_size_z * m_HR_ratio_z, &modified_buffer[0], &modified_buffer[0], &trans_buffer[0], 2);//2 means complex
    //int hr_Ny = Ny * m_HR_ratio_y;
    watch.Record(53);
    //such as fft////////////

#ifdef USE_FFT_MANY
    m_HR_fft_many_y.BackwardDirect((OneComplex*)&fft_buffer[0]);
#else
    for (int l = 0; l < my_line_xz; ++l) {
        auto* line = (OneComplex*)&modified_buffer[l * Ny * 2];
        auto* line_HR = (OneComplex*)&fft_buffer[l * hr_Ny * 2];
        UpConvert_Kspace_1d(line, Ny, line_HR, m_HR_ratio_y);
        m_HR_fft_y.BackwardDirect(line_HR);
    }
#endif
    watch.Record(58);


    ddm_ex.ScaleUp(m_HR_ratio_y);
    ddm_ex.BackwardExchangeY(Ny, local_size_x * local_size_z * m_HR_ratio_z, &fft_buffer[0], &fft_buffer[0], &trans_buffer[0], 2 * m_HR_ratio_y);//2 means complex
    watch.Record(51);

    //転置 y,z,x to x,y,z//
    Transpose3_back<OneComplex>(local_size_y * m_HR_ratio_y, local_size_z * m_HR_ratio_z, local_size_x, (OneComplex*)&modified_buffer[0], (OneComplex*)&fft_buffer[0]);
    watch.Record(52);



    //x exchange////////////////////////////////////////////////////////
    // x方向に領域分割されたバッファをExchangeして、x方向に連続なバッファに並べ替える//
    my_line_yz = ddm_ex.ForwardExchangeX(Nx, local_size_y* m_HR_ratio_y * local_size_z * m_HR_ratio_z, &modified_buffer[0], &modified_buffer[0], &trans_buffer[0], 2);//2 means complex
    //const int hr_Nx = Nx * m_HR_ratio_x;
    watch.Record(53);

    //such as fft////////////

#ifdef USE_FFT_MANY
    for (int l = 0; l < my_line_yz; ++l) {
        auto* line = (OneComplex*)&modified_buffer[l * Nx * 2];
        auto* line_HR = (OneComplex*)&fft_buffer[l * hr_Nx * 2];
        UpConvert_Kspace_1d(line, Nx, line_HR, m_HR_ratio_x);        
    }
    m_HR_fft_many_x.BackwardDirect((OneComplex*)&fft_buffer[0]);
#else
    for (int l = 0; l < my_line_yz; ++l) {
        auto* line = (OneComplex*)&modified_buffer[l * Nx * 2];
        auto* line_HR = (OneComplex*)&fft_buffer[l * hr_Nx * 2];        
        UpConvert_Kspace_1d(line, Nx, line_HR, m_HR_ratio_x);
        m_HR_fft_x.BackwardDirect(line_HR);
    }
#endif
    watch.Record(58);

    ddm_ex.ScaleUp(m_HR_ratio_x);
    ddm_ex.BackwardExchangeX(Nx, local_size_y * m_HR_ratio_y * local_size_z * m_HR_ratio_z, &fft_buffer[0], &fft_buffer[0], &trans_buffer[0], 2 * m_HR_ratio_x);//2 means complex
    watch.Record(51);

    for (int64_t i = 0; i < hr_local_size; ++i) {
        l_hr_rho[i] = fft_buffer[i * 2] * invN;
    }
    watch.Record(57);


}


#endif

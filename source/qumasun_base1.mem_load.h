#pragma once
#include "qumasun_base1.h"

inline void QUMASUN_BASE1::LoadStateFromMemory(const QUMASUN_BASE1::StateMemInfo& info, double* load_buffer,
        const double* eigen_values, const double* occupancy)
{
    
    bool allow_different_grid = true;

    const bool is_root_global = IsRoot(m_mpi_comm);
    const bool is_root_each_ddm = IsRoot(m_ddm_comm);

    

    const double dV = m_dx * m_dy * m_dz;

    const int TAG = 30000;
    const int TAGO = 310000;


    const size_t one_load_size = 2 * info.local_size * num_solution;
    size_t offset = 0;
    size_t off2 = 0;

    auto IsSameGrid = [](const GridRange& a, const GridRange& b) {
        return (a.begin_x == b.begin_x) && (a.begin_y == b.begin_y) && (a.begin_z == b.begin_z)
            && (a.end_x == b.end_x) && (a.end_y == b.end_y) && (a.end_z == b.end_z);
        };

    if (IsSameGrid(ml_grid, info.grid_mem)) {
        //grid sizeが同じなので、単にコピーすればOK//
        for (int sk = 0; sk < m_all_kinds_spin_kpoint; ++sk) {
            if ((m_having_spin_kpoint_begin <= sk) && (sk < m_having_spin_kpoint_begin + m_num_having_spin_kpoint)) {
                const auto& ws = ml_wave_set[sk - m_having_spin_kpoint_begin];

                gy::CopyMemory(ws.l_psi_set[0].re, load_buffer + offset, sizeof(double) * one_load_size);
                gy::CopyMemory(ws.eigen_values, eigen_values + off2, sizeof(double) * num_solution);
                gy::CopyMemory(ws.occupancy, occupancy + off2, sizeof(double) * num_solution);

                
                offset += one_load_size;
                off2 += num_solution;
            }
        }
    }else{

        FFTW_ExecutorMany3D fft3d_many_small;
        size_t work_size = 0;
        double* gather_buf = nullptr;

        for (int sk = 0; sk < m_all_kinds_spin_kpoint; ++sk) {
            if ((m_having_spin_kpoint_begin <= sk) && (sk < m_having_spin_kpoint_begin + m_num_having_spin_kpoint)) {
                const auto& ws = ml_wave_set[sk - m_having_spin_kpoint_begin];


                GridRangeMPI grid_mem = ml_grid;//sizeを変えて合成//
                grid_mem.begin_x = info.grid_mem.begin_x;
                grid_mem.begin_y = info.grid_mem.begin_y;
                grid_mem.begin_z = info.grid_mem.begin_z;
                grid_mem.end_x = info.grid_mem.end_x;
                grid_mem.end_y = info.grid_mem.end_y;
                grid_mem.end_z = info.grid_mem.end_z;
                GridRange global_mem = m_global_grid;//sizeを変えて合成//
                global_mem.end_x = info.global_size_x;
                global_mem.end_y = info.global_size_y;
                global_mem.end_z = info.global_size_z;

;

                //sizeが異なるので、ロード後にFFTでUpconvertしてから戻す//
                //分散したグリッド情報を、連続したグリッド情報に集約.
                //ただし、プロセスごとに状態数n=range.num_bundle ~ N/Pとなるように状態方向に分散して保持する//
                //つまり、全ての状態をDDMで持っている状態から、状態方向分散に変換//
                DDMGatherScatter exchanger;
                const auto range = exchanger.Estimate(global_mem, grid_mem, num_solution);
                if (work_size < 2 * m_size_3d * range.num_bundle) {
                    work_size = 2 * m_size_3d * range.num_bundle;
                    if (gather_buf) gy::AlignedFree(gather_buf);
                    gather_buf = gy::AlignedAlloc<double>(work_size);
                }

                size_t small_size = info.global_size_x * info.global_size_y * info.global_size_z;
                OneComplex* src_head = (OneComplex*)m_work;
                OneComplex* dest_head = src_head + small_size * range.num_bundle;

                exchanger.GatherExchangeD2Z(range, global_mem, (double*)src_head, grid_mem, load_buffer + offset, num_solution, gather_buf);

                //FFTしてupscaling//
                if (fft3d_many_small.GetNumBundle() != range.num_bundle) {
                    fft3d_many_small.Initialize(info.global_size_x, info.global_size_y, info.global_size_z, range.num_bundle, src_head);
                }

                fft3d_many_small.ForwardDirect(src_head);

                

                for (int n = 0; n < range.num_bundle; ++n) {
                    OneComplex* dest = dest_head + n * m_size_3d;
                    /*
                    UpDownConvert_Kspace_3d_any<OneComplex>(src_head + n * small_size, info.global_size_x, info.global_size_y, info.global_size_z,
                        dest, m_size_x, m_size_y, m_size_z);
                    */
                    UpConvert_Kspace_3d_any<OneComplex>(src_head + n * small_size, info.global_size_x, info.global_size_y, info.global_size_z,
                        dest, m_size_x, m_size_y, m_size_z);
                    
                }
                gy::Synchronize();

                if (m_fft_many_3d.GetNumBundle() != range.num_bundle) {
                    m_fft_many_3d.Initialize(m_size_x, m_size_y, m_size_z, range.num_bundle, dest_head);
                }
                gy::Synchronize();

                m_fft_many_3d.BackwardDirect(dest_head);
                for (int n = 0; n < range.num_bundle; ++n) {
                    OneComplex* dest = dest_head + n * m_size_3d;
                    double norm = 0.0;
                    for (int64_t i = 0; i < m_size_3d; ++i) {
                        norm += dest[i].r * dest[i].r + dest[i].i * dest[i].i;
                    }
                    //printf("load-norm[%d] = %f\n", n, norm* dV/ (double)m_size_3d);
                    const double inorm = 1.0 / (sqrt(norm * dV));
                    for (int64_t i = 0; i < m_size_3d; ++i) {
                        dest[i].r *= inorm;
                        dest[i].i *= inorm;
                    }
                    /*
                    norm = 0.0;
                    for (int64_t i = 0; i < m_size_3d; ++i) {
                        norm += dest[i].r * dest[i].r + dest[i].i * dest[i].i;
                    }
                    printf("load-re-norm[%d] = %f\n", n, norm* dV);
                    */
                }
                


                DDMGatherScatter exchanger2;
                const auto range2 = exchanger2.Estimate(m_global_grid, ml_grid, num_solution);
                exchanger2.ScatterExchangeZ2D(range2, ml_grid, ws.l_psi_set[0].re, m_global_grid, (double*)dest_head, num_solution, gather_buf);

                gy::CopyMemory(ws.eigen_values, eigen_values + off2, sizeof(double) * num_solution);
                gy::CopyMemory(ws.occupancy, occupancy + off2, sizeof(double) * num_solution);

                /*{
                    for (int n = 0; n < num_solution; ++n) {
                        double norm = 0.0;
                        for (int64_t i = 0; i < m_size_3d; ++i) {
                            norm += ws.l_psi_set[n].re[i] * ws.l_psi_set[n].re[i] + ws.l_psi_set[n].im[i] * ws.l_psi_set[n].im[i];
                        }
                        printf("psi-norm[%d] = %f\n", n, norm * dV);
                    }
                }*/

                offset += one_load_size;
                off2 += num_solution;
            }
        }

        if (gather_buf) gy::AlignedFree(gather_buf);
    }

    m_is_state_loaded = true;


}


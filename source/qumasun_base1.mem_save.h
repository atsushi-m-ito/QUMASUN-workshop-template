#pragma once
#include "qumasun_base1.h"

//計算結果をメモリ上で引き継ぐための機能//

/*
* 実際にコピーしたサイズを返す//
*/ 
size_t QUMASUN_BASE1::SaveStateToMemory(double* state_buffer, double* eigen_values, double* occupancy) {
    if (state_buffer == nullptr) {
        return 0;
    }

    auto local_size = ml_grid.Size3D();

    size_t offset = 0;
    for (int sk = m_having_spin_kpoint_begin; sk < m_num_having_spin_kpoint; ++sk) {
        size_t bundle_size = 2 * local_size * num_solution;
        memcpy(state_buffer + offset, ml_wave_set[sk - m_having_spin_kpoint_begin].l_psi_set[0].re, sizeof(double) * bundle_size);
        offset += bundle_size;
    }

    size_t off2 = 0;
    for (int sk = m_having_spin_kpoint_begin; sk < m_num_having_spin_kpoint; ++sk) {        
        memcpy(eigen_values + off2, ml_wave_set[sk - m_having_spin_kpoint_begin].eigen_values, sizeof(double) * num_solution);
        memcpy(occupancy + off2, ml_wave_set[sk - m_having_spin_kpoint_begin].occupancy, sizeof(double) * num_solution);

        
        off2 += num_solution;
    }
    return offset * sizeof(double);
}

QUMASUN_BASE1::StateMemInfo QUMASUN_BASE1::GetStateMemInfo() {

    StateMemInfo info;
    info.total_state = m_total_state;
    info.begin_state = m_begin_state;
    info.num_solution = num_solution;
    info.is_spin_on = is_spin_on;
    info.having_spin_kpoint_begin = m_having_spin_kpoint_begin;
    info.num_having_spin_kpoint = m_num_having_spin_kpoint;
    auto local_size = ml_grid.Size3D();
    info.local_size = local_size;
    info.required_byte_size = sizeof(double) * 2 * local_size * num_solution * m_num_having_spin_kpoint;
    info.grid_mem = ml_grid;

    info.global_size_x = m_size_x;
    info.global_size_y = m_size_y;
    info.global_size_z = m_size_z;

    return info;

}


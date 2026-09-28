#pragma once
//#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "qumasun_base1.h"
#include "symmetrize_density.h"
#include "qumasun_SetDensityOne.h"



//mpi supported//
/*
* 各プロセスの持つ波動関数から、電子密度を計算する。
* また、電子密度は領域分割されたまま計算されるが、
* m_same_ddm_place_commのrootへのreductionまでは行われる。 
*/
inline
void QUMASUN_BASE1::mSetDensityByPsi() {

    const bool is_root_global = IsRoot(m_mpi_comm);


    const size_t local_size = ml_grid.Size3D();


    if (is_root_global) {
        gy::CopyMemory(&m_rho_prev[0], &m_rho[0], sizeof(double)*m_size_3d);
    }

    gy::ZeroClear<double>(&ml_rho[0], local_size);
    if (is_spin_on) {
        gy::ZeroClear<double>(&ml_rho_diff[0], local_size);
    }

    for (int sk = 0; sk < m_num_having_spin_kpoint; ++sk) {
        if (ml_wave_set[sk].spin == SPIN::UP) {
            QUMASUN::Basic::SetDensityOne(ml_grid, &ml_rho[0], ml_wave_set[sk].l_psi_set, ml_wave_set[sk].occupancy, num_solution);
        } else {
            QUMASUN::Basic::SetDensityOne(ml_grid, &ml_rho_diff[0], ml_wave_set[sk].l_psi_set, ml_wave_set[sk].occupancy, num_solution);
        }
    }
    gy::Synchronize();
    
    //gather to root of same_ddm_place, after that only the gamma point has correct density//
    mReduceFieldInSamePlace(ml_rho);
    if (is_spin_on) {
        mReduceFieldInSamePlace(ml_rho_diff);
    }


    //以降は電子密度(up,down)から和と差を求める//
    //spin==offの場合は積分を出力するためだけのもの//
    if (IsRoot(m_same_ddm_place_comm)) {
        if (is_spin_on) { //spin polarization//


            double sum_rho_up_dn[2]{ 0.0,0.0 };
            //double total_rho_dn = 0.0;
            for (size_t i = 0; i < local_size; ++i) {
                sum_rho_up_dn[0] += ml_rho[i];
                sum_rho_up_dn[1] += ml_rho_diff[i];

                const double plus_rho = ml_rho[i] + ml_rho_diff[i];
                const double diff_rho = ml_rho[i] - ml_rho_diff[i];
                ml_rho[i] = plus_rho;
                ml_rho_diff[i] = diff_rho;
            }

            double total_rho_up_dn[2];
            MPI_Reduce(&sum_rho_up_dn[0], &total_rho_up_dn[0], 2, MPI_DOUBLE, MPI_SUM, 0, m_ddm_comm);
            if (IsRoot(m_ddm_comm)) {
                total_rho_up_dn[0] *= m_dx * m_dy * m_dz;
                total_rho_up_dn[1] *= m_dx * m_dy * m_dz;
                printf("total rho = %f + %f = %.15f\n", total_rho_up_dn[0], total_rho_up_dn[1], total_rho_up_dn[0] + total_rho_up_dn[1]);
            }
        } else {
            //check/////		
            double sum_rho = 0.0;
            for (size_t i = 0; i < local_size; ++i) {
                sum_rho += ml_rho[i];
            }
            double total_rho=0.0;
            MPI_Reduce(&sum_rho, &total_rho, 1, MPI_DOUBLE, MPI_SUM, 0, m_ddm_comm);
            if (IsRoot(m_ddm_comm)) {
                total_rho *= m_dx * m_dy * m_dz;
                printf("total rho = %.15f\n", total_rho);
            }
        }
    }
}


//mpi supported//
inline
void QUMASUN_BASE1::mSymmetrizeDensity() {

    if (kpoint_symmetry == QUMASUN::KPOINT_SYMMETRY::NONE) return;

    if (IsRoot(m_same_ddm_place_comm)) {
        mGatherField(&m_rho[0], ml_rho);
        if (is_spin_on) {
            mGatherField(&m_rho_diff[0], ml_rho_diff);
        }
        


        if (IsRoot(m_mpi_comm)) {

            SymmetrizeDensity(&m_rho[0], m_global_grid.SizeX(), m_global_grid.SizeY(), m_global_grid.SizeZ(), kpoint_symmetry);
            if (is_spin_on) {
                SymmetrizeDensity(&m_rho_diff[0], m_global_grid.SizeX(), m_global_grid.SizeY(), m_global_grid.SizeZ(), kpoint_symmetry);
            }
        }

        
        mScatterField(ml_rho, &m_rho[0]);
        if (is_spin_on) {
            mScatterField(ml_rho_diff, &m_rho_diff[0]);
        }
    }


}

#pragma once
//#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <complex>
#include "qumasun_kpoint.h"
#include "lobpcg_z_multi_mpi.h"
#include "diis_z_mpi.h"



//mpi supported//
inline
void QUMASUN_KPOINT::mSolveLOBPCG(int steps, int SCF_current_step) {
	const int proc_id = GetProcessID(m_mpi_comm);
	const bool is_ddm_root = IsRoot(m_ddm_comm);
	const size_t local_size = ml_grid.Size3D();

#ifdef USE_SCALAPACK
    
    //固有値ソルバーのプロセス数が過剰な場合に抑える//
      

    //Blacsの初期化//
    if (m_blacs_grid == nullptr) {
        m_blacs_grid = new BlacsGridInfo;

        //1次元当たりの最小プロセス数の計算//
        constexpr int EIGEN_LINES_PER_PROC = 2;
        const int num_procs = GetNumProcess(m_ddm_comm);
        int min_procs1 = (num_solution + EIGEN_LINES_PER_PROC - 1) / EIGEN_LINES_PER_PROC;
        int ww = (num_solution + min_procs1 - 1) / min_procs1;
        while (min_procs1 * ww - num_solution >= ww) {
            min_procs1--;
            ww = (num_solution + min_procs1 - 1) / min_procs1;            
        }

        //全プロセス使った場合の行・列方向のプロセス数//
        int np_rows = (int)(sqrt((float)num_procs));
        do {
            if ((num_procs % np_rows) == 0) break;
            np_rows--;
        } while (np_rows >= 2);
        int np_cols = num_procs / np_rows;//必ずnp_cols >= np_rowsとなる//


        if (min_procs1 < np_cols) {
            while (min_procs1 * min_procs1 > num_procs) {
                min_procs1--;
            }
            ww = (num_solution + min_procs1 - 1) / min_procs1;
            while (min_procs1 * ww - num_solution >= ww) {
                min_procs1--;
                ww = (num_solution + min_procs1 - 1) / min_procs1;
            }

            const int id_in_ddm = GetProcessID(m_ddm_comm);
            MPI_Comm blacs_comm;
            MPI_Comm_split(m_ddm_comm, (id_in_ddm < min_procs1 * min_procs1) ? 1 : MPI_UNDEFINED, id_in_ddm, &blacs_comm);
            
            *m_blacs_grid = BeginBLACSFromMPIComm(blacs_comm, min_procs1, min_procs1);
                        
        } else {
            const int mpi_split_color = proc_id / GetNumProcess(m_ddm_comm);
            *m_blacs_grid = BeginBLACS(m_ddm_comm, mpi_split_color);
            
        }
        
        if (is_ddm_root) {
            printf("[%d]Initialize BLACS: used_num_procs = %d x %d / %d\n", proc_id, m_blacs_grid->np_rows, m_blacs_grid->np_cols, num_procs);
            fflush(stdout);
        }
    }
    
#endif

	for(int sk = 0; sk < m_num_having_spin_kpoint;++sk){
		const int kpoint_x = ml_wave_set[sk].kpoint_x;
		const int kpoint_y = ml_wave_set[sk].kpoint_y;
		const int kpoint_z = ml_wave_set[sk].kpoint_z;
		auto& V_tot = (ml_wave_set[sk].spin == SPIN::UP) ? ml_Vtot : ml_Vtot_down;

		if (is_ddm_root) {
			printf("[%d] sk=%d, kpoint=%d,%d,%d, spin=%s\n", proc_id, sk, ml_wave_set[sk].kpoint_x, ml_wave_set[sk].kpoint_y, ml_wave_set[sk].kpoint_z, ml_wave_set[sk].spin == SPIN::UP ? "up" : "down");
            fflush(stdout);
		}

        LOBPCG::Eigen_z_multi_mpi(ml_grid, m_dx * m_dy * m_dz, num_solution, steps,
            ml_wave_set[sk].eigen_values, ml_wave_set[sk].l_psi_set, ml_keep_lobpcg[sk].keep_p, m_work_lobpcg, m_work,
            &m_num_fix_state_sk[sk], m_residual_convergence_threshold,
            ml_keep_lobpcg[sk].keep_S_matrix, SCF_current_step, sk,
#ifdef USE_SCALAPACK
            *m_blacs_grid,
#endif
#if 1
            [&V_tot, &kpoint_x, &kpoint_y, &kpoint_z, &sk, this](SoAComplex* Ax, const SoAComplex* x, int num_solution) {
                this->mHamiltonianMatrix_ddm_bundle(Ax, V_tot, x, num_solution, kpoint_x, kpoint_y, kpoint_z, sk);
            },
#else
			[&V_tot, &kpoint_x, &kpoint_y, &kpoint_z, &sk, this](SoAComplex& Ax, const SoAComplex& x) {
				this->mHamiltonianMatrix_ddm(Ax, V_tot, x, kpoint_x, kpoint_y, kpoint_z, sk);
			},
#endif
			watch);

        gyCheckError(gyGetLastError(), "DEV-ERROR54");

        if (is_ddm_root) {
            printf("[%d] end-LOBPCG-step\n", proc_id);
            fflush(stdout);
        }
	}
	/*
#ifdef USE_SCALAPACK
	EndBLACS(blacs_grid);
#endif
*/
    MPI_Barrier(m_mpi_comm);
    watch.Record(18);
}



//mpi supported//
inline
void QUMASUN_KPOINT::mSolveDIIS(int steps, int SCF_current_step) {
    const int proc_id = GetProcessID(m_mpi_comm);
    const bool is_ddm_root = IsRoot(m_ddm_comm);
    const size_t local_size = ml_grid.Size3D();

    if (m_diis_executer.empty()) {
        m_diis_executer.resize(m_num_having_spin_kpoint);
        for (int sk = 0; sk < m_num_having_spin_kpoint; ++sk) {
            m_diis_executer[sk].Initialize(m_diis_history_count, num_solution);
        }
    }

/*
#ifdef USE_SCALAPACK
    const int mpi_split_color = proc_id / GetNumProcess(m_ddm_comm);
    BlacsGridInfo blacs_grid = BeginBLACS(m_ddm_comm, mpi_split_color);
#endif
*/
    for (int sk = 0; sk < m_num_having_spin_kpoint; ++sk) {
        const int kpoint_x = ml_wave_set[sk].kpoint_x;
        const int kpoint_y = ml_wave_set[sk].kpoint_y;
        const int kpoint_z = ml_wave_set[sk].kpoint_z;
        auto& V_tot = (ml_wave_set[sk].spin == SPIN::UP) ? ml_Vtot : ml_Vtot_down;

        if (is_ddm_root) {
            printf("[%d] DIIS sk=%d, kpoint=%d,%d,%d, spin=%s\n", proc_id, sk, ml_wave_set[sk].kpoint_x, ml_wave_set[sk].kpoint_y, ml_wave_set[sk].kpoint_z, ml_wave_set[sk].spin == SPIN::UP ? "up" : "down");
            fflush(stdout);
        }

        m_diis_executer[sk].Eigen_z_multi_mpi( ml_grid, m_dx * m_dy * m_dz, num_solution, 
            ml_wave_set[sk].eigen_values, ml_wave_set[sk].l_psi_set, ml_keep_lobpcg[sk].keep_p, 
            SCF_current_step, sk,
            /*
#ifdef USE_SCALAPACK
            blacs_grid,
#endif
*/
#if 1
            [&V_tot, &kpoint_x, &kpoint_y, &kpoint_z, &sk, this](SoAComplex* Ax, const SoAComplex* x, int num_solution) {
                this->mHamiltonianMatrix_ddm_bundle(Ax, V_tot, x, num_solution, kpoint_x, kpoint_y, kpoint_z, sk);
            },
#else
            [&V_tot, &kpoint_x, &kpoint_y, &kpoint_z, &sk, this](SoAComplex& Ax, const SoAComplex& x) {
                this->mHamiltonianMatrix_ddm(Ax, V_tot, x, kpoint_x, kpoint_y, kpoint_z, sk);
            },
#endif
            watch);


        if (is_ddm_root) {
            printf("[%d] end-DIIS-step\n", proc_id);
            fflush(stdout);
        }
    }
/*
#ifdef USE_SCALAPACK
    EndBLACS(blacs_grid);
#endif
*/
    MPI_Barrier(m_mpi_comm);
    watch.Record(18);
}


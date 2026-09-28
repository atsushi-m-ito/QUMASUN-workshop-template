#pragma once
#ifdef USE_MPI
#include "mpi_helper.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include "qumasun_kpoint.h"
#include "vps_loader.h"
#include "pao_loader.h"
#include "cube_reader3.h"
#include "actual_kpoint.h"
#include "lobpcg_z_multi_mpi.h"
#include "qumasun_input.h"



inline
QUMASUN_KPOINT::QUMASUN_KPOINT(const Input& input, const MPI_Comm& mpi_comm_, const int ddm_num[4]) :
    QUMASUN_BASE1(input, mpi_comm_, ddm_num),
    //condition of SCF solver///////////////
    m_solver(SOLVER::LOBPCG),		//LOBPCG, DIIS, or Lanczos//
    //m_solver(SOLVER::DIIS),		
    LIMIT_SCF_STEP(input.scf_step),
    LOBPCG_STEP_PER_SCF(input.eigen_step_per_scf),
    LOBPCG_STEP_FIRST(input.eigen_step_initial),
    m_residual_convergence_threshold(input.scf_residual_convergence_threshold),
    m_threshold_density(input.scf_threshold_density),
    m_threshold_energy(input.scf_threshold_energy),
    m_convergence_mode(Mode(input))
{
    if (m_convergence_mode == 0) {
        m_convergence_mode = CONVERGENCE_MODE::ETOT;
        m_threshold_energy = DEFAULT_THRESHOLD;
    }
	

    size_t work_size = 0;
    
    if (m_solver == SOLVER::LOBPCG) {
        work_size = LOBPCG::WorkSize_z_multi_mpi_temporary(ml_grid.Size3D(), num_solution);
        size_t size = LOBPCG::WorkSize_z_multi_mpi_keep(ml_grid.Size3D(), num_solution);
        if (IsRoot(m_ddm_comm)) printf("NOTE: LOBPCG memory size = %zd byte\n", size * sizeof(double)); fflush(stdout);

        m_work_lobpcg = gy::AlignedAlloc<double>(size);
        m_num_fix_state_sk = std::make_unique<int[]>(m_num_having_spin_kpoint);
        ml_keep_lobpcg = new KeepLOBPCGSet[m_num_having_spin_kpoint];
        const size_t local_size = ml_grid.Size3D();
        for (int sk = 0; sk < m_num_having_spin_kpoint; ++sk) {
            ml_keep_lobpcg[sk].Allocate(local_size, num_solution);
            gy::ZeroClear<double>(ml_keep_lobpcg[sk].keep_p, local_size * num_solution * 2);
            m_num_fix_state_sk[sk] = 0;
        }
    }else if(m_solver==SOLVER::DIIS){
        ml_keep_lobpcg = new KeepLOBPCGSet[m_num_having_spin_kpoint];
        const size_t local_size = ml_grid.Size3D();
        for (int sk = 0; sk < m_num_having_spin_kpoint; ++sk) {
            ml_keep_lobpcg[sk].Allocate(local_size * m_diis_history_count * 2, num_solution);
        }
    }


    
    work_size = std::max<size_t>(work_size, DDMGatherScatter::GetWorkSize(m_global_grid.Size3D(), num_solution, m_ddm_comm));
    


#ifndef FORCE_DIFF_PROJ
    constexpr int bundle_width = NL_FORCE_BANDLE_WIDTH;
    work_size = std::max<size_t>(work_size, ml_grid.Size3D() * bundle_width * 2 * 3);//buffer nonlocal force with bundle
#endif
    work_size = std::max<size_t>(work_size, ml_grid.Size3D() * (4 * m_HR_ratio_x * m_HR_ratio_y * m_HR_ratio_z));
    work_size = std::max<size_t>(work_size, ml_grid.Size3D() * 12);//for GGA

    if (IsRoot(m_ddm_comm)) {       
        work_size = std::max<size_t>(work_size, m_size_3d * (6 + 5 * m_HR_ratio_x * m_HR_ratio_y * m_HR_ratio_z));
	}
    //printf("[%d]POS5\n", GetProcessID(m_mpi_comm)); fflush(stdout);
#ifdef DDM_FFT
    if (IsRoot(m_same_ddm_place_comm)) {
        constexpr int required_num_buf = 3 * 2;
        work_size = std::max<size_t>(work_size,
            required_num_buf * std::max<size_t>(m_ddm_exchanger->GetExcangeBufferSize(m_size_x, m_size_y, m_size_z), ml_grid.Size3D()) * m_HR_ratio_x * m_HR_ratio_y * m_HR_ratio_z);
    }
#endif


    work_size = std::max<size_t>(work_size, (m_pp_SvF.GetWorkSize(num_solution, m_num_nuclei) + sizeof(double))/ sizeof(double));

    mInitializeWorkBuffer(work_size);

    //mixing///////////////////////////////////////
    m_mixing_max_params0 = input.scf_mixing_ratio;
    if (IsRoot(m_same_ddm_place_comm)) {
        
        switch (input.scf_mixing_mode) {
        case QUMASUN::MixingMode::Simple:
        {
            m_mixing_params[0] = input.scf_mixing_ratio;
            m_mixer = new MixingSimple(ml_grid.Size3D(), m_ddm_comm);
            break;
        }
        case QUMASUN::MixingMode::LBFGS:
        {
            m_mixing_params[0] = input.scf_mixing_ratio;
            m_mixer = new MixingLBFGS(ml_grid.Size3D(), m_ddm_comm, input.scf_mixing_num_history, m_dx, m_dy, m_dz, nullptr);
            break;
        }
        case QUMASUN::MixingMode::SimpleKerker:
        {
            m_mixing_params[0] = input.scf_mixing_ratio;
            m_mixing_params[1] = input.scf_mixing_kerker_factor;
            m_mixer = new MixingKerker(ml_grid.Size3D(), is_spin_on?1:0, m_ddm_comm, m_global_grid, ml_grid, m_dx, m_dy, m_dz);
            break;
        }
        case QUMASUN::MixingMode::LBFGSKerker:
        {
            m_mixing_params[0] = input.scf_mixing_ratio;
            m_mixing_params[1] = input.scf_mixing_kerker_factor;
            m_mixer = new MixingLBFGS(ml_grid.Size3D(), m_ddm_comm, input.scf_mixing_num_history, m_dx, m_dy, m_dz,
                new TransKerker(m_global_grid, ml_grid, m_dx, m_dy, m_dz));            
            break;
        }
        case QUMASUN::MixingMode::DIIS:
        {
            m_mixing_params[0] = input.scf_mixing_ratio;
            m_mixing_params[1] = input.scf_mixing_kerker_factor;
            m_mixer = new MixingDIIS(ml_grid.Size3D(), is_spin_on ? 1 : 0, m_ddm_comm, input.scf_mixing_num_history);
            break;
        }
        }
    }


    if (m_fftw == nullptr) {//symplecticではroot以外も運動エネルギー演算子においてFFTを担当する//
        m_fftw = new FFTW_Executor;
        m_fftw->Initialize(m_size_x, m_size_y, m_size_z, FFTW_ESTIMATE);
    }

	is_construction_successful = true;

    //printf("[%d]POS6\n", GetProcessID(m_mpi_comm)); fflush(stdout);
};

inline
QUMASUN_KPOINT::~QUMASUN_KPOINT() {

    delete[] ml_keep_lobpcg;
    delete m_mixer;

    gy::AlignedFree(m_work_lobpcg);

#ifdef USE_SCALAPACK
    if (m_blacs_grid != nullptr) {
        if (m_blacs_grid->mpi_comm != MPI_COMM_NULL) {

            EndBLACS(*m_blacs_grid);

            if (m_blacs_grid->mpi_comm != m_ddm_comm) {
                MPI_Comm_free(&(m_blacs_grid->mpi_comm));
            }
        }
    }
#endif
};

#endif

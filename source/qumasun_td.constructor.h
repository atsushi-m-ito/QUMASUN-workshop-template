#pragma once
#ifdef USE_MPI
#include "mpi_helper.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include "qumasun_td.h"
#include "vps_loader.h"
#include "pao_loader.h"
#include "cube_reader3.h"
#include "actual_kpoint.h"



inline
QUMASUN_TD::QUMASUN_TD(const Input& input, const MPI_Comm& mpi_comm_, const int ddm_num[4]) :
    QUMASUN_BASE1(input, mpi_comm_, ddm_num)

{



    size_t work_size = mWorkSizeTimeEvoH4(ml_grid.Size3D(), num_solution);
    constexpr int bundle_width = NL_FORCE_BANDLE_WIDTH;
    work_size = std::max<size_t>(work_size, ml_grid.Size3D() * bundle_width * 2 * 3);//buffer nonlocal force with bundle
    work_size = std::max<size_t>(work_size, ml_grid.Size3D() * (4 * m_HR_ratio_x * m_HR_ratio_y * m_HR_ratio_z));
	if (IsRoot(m_ddm_comm)) {
        work_size = std::max<size_t>(work_size, m_size_3d * (6 + 5 * m_HR_ratio_x * m_HR_ratio_y * m_HR_ratio_z));
	}
    work_size = std::max<size_t>(work_size, DDMGatherScatter::GetWorkSize(m_global_grid.Size3D(), num_solution, m_ddm_comm));
#ifdef DDM_FFT
    if (IsRoot(m_same_ddm_place_comm)) {
        constexpr int required_num_buf = 3 * 2;
        work_size = std::max<size_t>(work_size,
            required_num_buf * std::max<size_t>(m_ddm_exchanger->GetExcangeBufferSize(m_size_x, m_size_y, m_size_z), ml_grid.Size3D()) * m_HR_ratio_x * m_HR_ratio_y * m_HR_ratio_z);
    }
#endif

    mInitializeWorkBuffer(work_size);
	
    if (IsRoot(m_ddm_comm)) {
        const int proc_id = GetProcessID(m_mpi_comm);
        printf("[%d] work_size: %zd, %p\n", proc_id, work_size * 8, m_work);
    }


	is_construction_successful = true;
};

inline
QUMASUN_TD::~QUMASUN_TD() {

};


#endif

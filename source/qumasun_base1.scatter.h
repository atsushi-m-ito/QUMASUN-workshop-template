#pragma once
#include "qumasun_base1.h"
#include "GridRange.h"
#include "GridScatterGather.h"

inline
void QUMASUN_BASE1::mScatterField(double* local_dest, const double* global_src) {
	ScatterGrid(ml_grid, local_dest, m_global_grid, global_src, m_root_id);
}

inline
void QUMASUN_BASE1::mGatherField(double* global_dest, const double* local_src) {	
	GatherGrid(m_global_grid, global_dest, ml_grid, local_src, m_root_id);
}


inline
void QUMASUN_BASE1::mScatterField_HR(double* local_dest, const double* global_src) {
    ScatterGrid(ml_grid, local_dest, m_global_grid, global_src, m_root_id, m_HR_ratio_x, m_HR_ratio_y, m_HR_ratio_z);
}

inline
void QUMASUN_BASE1::mGatherField_HR(double* global_dest, const double* local_src) {
    GatherGrid(m_global_grid, global_dest, ml_grid, local_src, m_root_id, m_HR_ratio_x, m_HR_ratio_y, m_HR_ratio_z);
}



/*
* kpoint, spinは異なるがDDMの担当領域は同じプロセスで先に電子密度などを足し合わせる
* 次に、kpoint=0,0,0, spin=UPのマスタープロセス(場所の異なるものが複数ある)でDDM通信でgather
*/
inline
void QUMASUN_BASE1::mHierarchyGatherField(double* global_dest, const double* local_src) {
	const int num_procs_same_place = GetNumProcess(m_same_ddm_place_comm);
	if (num_procs_same_place > 1) {
		const int local_size = ml_grid.Size3D();
		const int proc_id_same_place = GetProcessID(m_same_ddm_place_comm);
		if (proc_id_same_place == 0) {
			double* red_local = new double[local_size];
			MPI_Reduce(local_src, red_local, local_size, MPI_DOUBLE, MPI_SUM, 0, m_same_ddm_place_comm);
			GatherGrid(m_global_grid, global_dest, ml_grid, red_local, m_root_id);
			delete[] red_local;
		} else {
			MPI_Reduce(local_src, nullptr, local_size, MPI_DOUBLE, MPI_SUM, 0, m_same_ddm_place_comm);
		}
		
	} else {
		GatherGrid(m_global_grid, global_dest, ml_grid, local_src, m_root_id);
	}
}


/*
* kpoint=0,0,0, spin=UPのマスタープロセス(場所の異なるものが複数ある)でDDM通信でscatter
* 次に、kpoint, spinは異なるがDDMの担当領域は同じプロセスへとコピー.
*/
inline
void QUMASUN_BASE1::mHierarchyScatterField(double* local_dest, const double* global_src) {
	
	const int num_procs_same_place = GetNumProcess(m_same_ddm_place_comm);
	if (num_procs_same_place > 1) {
		const int local_size = ml_grid.Size3D();
		const int proc_id_same_place = GetProcessID(m_same_ddm_place_comm);
		if (proc_id_same_place == 0) {
			ScatterGrid(ml_grid, local_dest, m_global_grid, global_src, m_root_id);
		}
		MPI_Bcast(local_dest, local_size, MPI_DOUBLE, 0, m_same_ddm_place_comm);

	} else {
		ScatterGrid(ml_grid, local_dest, m_global_grid, global_src, m_root_id);
	}
}

/*
* kpoint, spinは異なるがDDMの担当領域は同じプロセスで先に電子密度などを足し合わせる
* ここまで止めておき、m_mpi_commのrootへは集約しない。
*/
inline
void QUMASUN_BASE1::mReduceFieldInSamePlace(double* local_src_dest) {
    const int num_procs_same_place = GetNumProcess(m_same_ddm_place_comm);
    if (num_procs_same_place <= 1) {
        return; //nothing to do, because kpoint,spin,or state decomposition is not used.//
    }

    const int local_size = ml_grid.Size3D();
    const int proc_id_same_place = GetProcessID(m_same_ddm_place_comm);
    if (proc_id_same_place == 0) {
        //double* red_local = new double[local_size];
        MPI_Reduce(MPI_IN_PLACE, local_src_dest, local_size, MPI_DOUBLE, MPI_SUM, 0, m_same_ddm_place_comm);
        //GatherGrid(m_global_grid, global_dest, ml_grid, red_local, m_root_id);
        //delete[] red_local;
    } else {
        MPI_Reduce(local_src_dest, nullptr, local_size, MPI_DOUBLE, MPI_SUM, 0, m_same_ddm_place_comm);
    }

}



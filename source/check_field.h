#pragma once
#include "mpi_helper.h"
#include "GridRange.h"

inline 
void CheckMinMax(const GridRangeMPI& l_grid, const double* V, const char* str) {
    {
        auto m_ddm_comm = l_grid.mpi_comm;
        const int local_size = l_grid.Size3D();
        double max_V = -DBL_MAX;
        double min_V = DBL_MAX;
        for (int i = 0; i < local_size; ++i) {
            if (max_V < V[i]) {
                max_V = V[i];
            }
            if (min_V > V[i]) {
                min_V = V[i];
            }
        }
        double global_max, global_min;
        MPI_Reduce(&max_V, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, m_ddm_comm);
        MPI_Reduce(&min_V, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, m_ddm_comm);
        if (IsRoot(m_ddm_comm)) {
            printf("(%s)min,max = %.10f, %.10f\n", str, global_min, global_max);
        }
    }
}

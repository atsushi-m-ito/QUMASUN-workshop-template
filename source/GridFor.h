#pragma once
#ifdef USE_MPI
#include <mpi.h>
#include "mpi_helper.h"
#endif
#include "GridRange.h"

//3次元デカルト座標グリッドの計算のためのループを司る//
//MPI並列時にはリダクションも提供する



template <class FUNC>
void For(const GridRange& grid, FUNC&& func) {
	const size_t size_xyz = grid.Size3D();
	for (size_t i = 0; i < size_xyz; ++i) {
		func(i);
	}
}

template <class T, class FUNC>
void ForReduce(const GridRange& grid, T* result, FUNC&& func) {
	const size_t size_xyz = grid.Size3D();
	T sum{ 0 };
	for (size_t i = 0; i < size_xyz; ++i) {
		sum += func(i);
	}
	*result = sum;
}


template <class T, class FUNC>
auto ForReduce2(const GridRange& grid, T init, FUNC&& func) {
	const size_t size_xyz = grid.Size3D();
	T sum = init;
	for (size_t i = 0; i < size_xyz; ++i) {
		sum += func(i);
	}
	return sum;
}

template <class FUNC>
void ForXYZ(const GridRange& grid, FUNC&& func) {
	const int64_t size_x = (grid.end_x - grid.begin_x);
	const int64_t size_y = (grid.end_y - grid.begin_y);
	const int64_t size_z = (grid.end_z - grid.begin_z);

	for (int64_t iz = 0; iz < size_z; ++iz) {
		const int64_t giz = iz + grid.begin_z;
		for (int64_t iy = 0; iy < size_y; ++iy) {
			const int64_t giy = iy + grid.begin_y;
			for (int64_t ix = 0; ix < size_x; ++ix) {
				const int64_t gix = ix + grid.begin_x;
				const int64_t i = ix + size_x * (iy + size_y * iz);
				func(i, gix, giy, giz);
			}
		}
	}
}


#ifdef USE_MPI

template <class FUNC>
void For(const GridRangeMPI& grid, FUNC&& func) {
	For( *(GridRange*)&grid, func);
}

template <class T>
void ArrayReduce(const GridRangeMPI& grid, T* arrays, int N) {
	constexpr int root_id = 0;
	const bool is_root = IsRoot(grid.mpi_comm);
	T* results = (is_root) ? new T[N] : nullptr;
	MPI_Reduce(arrays, results, N, GetDatatype<T>(), MPI_SUM, root_id, grid.mpi_comm);
	if (is_root) {
		memcpy(arrays, results, sizeof(T) * N);
		delete[] results;
	}
}

template <class T>
T ValueReduce(const GridRangeMPI& grid, const T& value) {
	constexpr int root_id = 0;
	T result;
	
	MPI_Reduce(&value, &result, 1, GetDatatype<T>(), MPI_SUM, root_id, grid.mpi_comm);

	/*{
		int proc_id;
		MPI_Comm_rank(grid.mpi_comm, &proc_id);
		printf("[%d]reduce: %f to %f\n", proc_id, value, result); fflush(stdout);
	}*/
	return result;
}

template <class T, class FUNC>
void ForReduce(const GridRangeMPI& grid, T* result, FUNC&& func) {
	T local_sum;
	ForReduce(*(GridRange*)&grid, &local_sum, func);
	*result = ValueReduce(grid, local_sum);
}

template <class T, class FUNC>
auto ForReduce2(const GridRangeMPI& grid, T init, FUNC&& func) {
	T local_sum = ForReduce2(*(GridRange*)&grid, init, func);
	return ValueReduce(grid, local_sum);
}

#endif



/*
* 原子核が原点(0,0,0)にあるとしてデカルトグリッドの離散場にCoreChargeを貼り付け
*/
template<class FUNC>
void tForCutoffAtOrigin(
    int grid_x, int grid_y, int grid_z,
    double dx, double dy, double dz,
    double cutoff_length, FUNC func) {


    //const double rr_min = (r_minimum) * (r_minimum);
    const double rr_max = (cutoff_length) * (cutoff_length);
    constexpr int margin = 0;
    const int half_x = margin + (int)ceil(cutoff_length / dx);
    const int half_y = margin + (int)ceil(cutoff_length / dy);
    const int half_z = margin + (int)ceil(cutoff_length / dz);

    const int size_x = (2 * half_x + 1);
    const int size_y = (2 * half_y + 1);
    const int size_z = (2 * half_z + 1);



    for (int iz = -half_z; iz <= half_z; ++iz) {
        const double z = dz * (double)iz;
        const double zz = z * z;
        for (int iy = -half_y; iy <= half_x; ++iy) {
            const double y = dy * (double)iy;
            const double yy_zz = y * y + zz;
            for (int ix = -half_x; ix <= half_x; ++ix) {
                const double x = dx * (double)ix;
                const double rr = x * x + yy_zz;
                const int i = (ix + half_x) + size_x * ((iy + half_y) + size_y * (iz + half_z));
                const int id = ((ix >= 0) ? ix : ix + grid_x) + grid_x * (((iy >= 0) ? iy : iy + grid_y) + grid_y * ((iz >= 0) ? iz : iz + grid_z));

                //if (rr <= rr_max) {
                func(rr, id, i);
                //}
                /*if (rr_min > rr) {
                    cartesian_dest[i].r = radial_src[0];
                    cartesian_dest[i].r = radial_src[0];
                } else {
                    cartesian_dest[i] = RadialGrid2::GetValueBySquare(rr, pp->V_local, pp->num_radial_grids, pp->xi_min, pp->xi_delta);
                }*/

            }
        }

    }
}

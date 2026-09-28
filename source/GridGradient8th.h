#pragma once
#ifdef USE_MPI
#include <mpi.h>
#include "mpi_helper.h"
#endif
#include "GridRange.h"
#include <cstdio>


#include "GridDifference8th.h"




template <class T>
void Gradient8th_x_halo_f(const GridRange& grid, T* out, const T* p, double coef, double dx, double dy, double dz) {
	
	const int Nx = grid.SizeX();
	const int Ny = grid.SizeY();
	const int Nz = grid.SizeZ();
	constexpr int MW = 4;

	const double c4x = -3.0 * coef / (840.0 * dx);
	const double c3x = 32.0*coef / (840.0 * dx);
	const double c2x = -168.0 * coef / (840.0 * dx);
	const double c1x = 672.0 * coef / (840.0 * dx);
	
	const int size_x = grid.SizeX() + MW*2;
	const int size_y = grid.SizeY() + MW * 2;
	const int size_z = grid.SizeZ() + MW * 2;

	for (int iz = MW; iz < size_z - MW; ++iz) {
		const int idz_m4 = -4 * size_x * size_y;
		const int idz_m3 = -3 * size_x * size_y;
		const int idz_m2 = -2 * size_x * size_y;
		const int idz_m = -1 * size_x * size_y;
		const int idz_p = 1 * size_x * size_y;
		const int idz_p2 = 2 * size_x * size_y;
		const int idz_p3 = 3 * size_x * size_y;
		const int idz_p4 = 4 * size_x * size_y;

		for (int iy = MW; iy < size_y - MW; ++iy) {
			const int idy_m4 = -4 * size_x;
			const int idy_m3 = -3 * size_x;
			const int idy_m2 = -2 * size_x;
			const int idy_m = -1 * size_x;
			const int idy_p = 1 * size_x;
			const int idy_p2 = 2 * size_x;
			const int idy_p3 = 3 * size_x;
			const int idy_p4 = 4 * size_x;
#pragma ivdep
			for (int ix = MW; ix < size_x - MW; ++ix) {
				const int idx_m4 = -4;
				const int idx_m3 = -3;
				const int idx_m2 = -2;
				const int idx_m = -1;
				const int idx_p = 1;
				const int idx_p2 = 2;
				const int idx_p3 = 3;
				const int idx_p4 = 4;
				const int64_t i = ix + size_x * (iy + (size_y * iz));
				const int64_t io = (ix - MW) + Nx * (iy - MW + (Ny * (iz - MW)));


				auto dpsi_dx = c4x * (p[i + idx_p4] - p[i + idx_m4]) + c3x * (p[i + idx_p3] - p[i + idx_m3]) + c2x * (p[i + idx_p2] - p[i + idx_m2]) + c1x * (p[i + idx_p] - p[i + idx_m]);
				out[io] = dpsi_dx;
			}
		}
	}
}


template <class T>
void Gradient8th_y_halo_f(const GridRange& grid, T* out, const T* p, double coef, double dx, double dy, double dz) {

	const int Nx = grid.SizeX();
	const int Ny = grid.SizeY();
	const int Nz = grid.SizeZ();
	constexpr int MW = 4;

	const double c4y = -3.0 * coef / (840.0 * dy);
	const double c3y = 32.0 * coef / (840.0 * dy);
	const double c2y = -168.0 * coef / (840.0 * dy);
	const double c1y = 672.0 * coef / (840.0 * dy);

	const int size_x = grid.SizeX() + MW * 2;
	const int size_y = grid.SizeY() + MW * 2;
	const int size_z = grid.SizeZ() + MW * 2;

	for (int iz = MW; iz < size_z - MW; ++iz) {
		const int idz_m4 = -4 * size_x * size_y;
		const int idz_m3 = -3 * size_x * size_y;
		const int idz_m2 = -2 * size_x * size_y;
		const int idz_m = -1 * size_x * size_y;
		const int idz_p = 1 * size_x * size_y;
		const int idz_p2 = 2 * size_x * size_y;
		const int idz_p3 = 3 * size_x * size_y;
		const int idz_p4 = 4 * size_x * size_y;

		for (int iy = MW; iy < size_y - MW; ++iy) {
			const int idy_m4 = -4 * size_x;
			const int idy_m3 = -3 * size_x;
			const int idy_m2 = -2 * size_x;
			const int idy_m = -1 * size_x;
			const int idy_p = 1 * size_x;
			const int idy_p2 = 2 * size_x;
			const int idy_p3 = 3 * size_x;
			const int idy_p4 = 4 * size_x;
#pragma ivdep
			for (int ix = MW; ix < size_x - MW; ++ix) {
				const int idx_m4 = -4;
				const int idx_m3 = -3;
				const int idx_m2 = -2;
				const int idx_m = -1;
				const int idx_p = 1;
				const int idx_p2 = 2;
				const int idx_p3 = 3;
				const int idx_p4 = 4;
				const int64_t i = ix + size_x * (iy + (size_y * iz));
				const int64_t io = (ix - MW) + Nx * (iy - MW + (Ny * (iz - MW)));


				auto dpsi_dy = c4y * (p[i + idy_p4] - p[i + idy_m4]) + c3y * (p[i + idy_p3] - p[i + idy_m3]) + c2y * (p[i + idy_p2] - p[i + idy_m2]) + c1y * (p[i + idy_p] - p[i + idy_m]);
				out[io] = dpsi_dy;
			}
		}
	}
}


template <class T>
void Gradient8th_z_halo_f(const GridRange& grid, T* out, const T* p, double coef, double dx, double dy, double dz) {

	const int Nx = grid.SizeX();
	const int Ny = grid.SizeY();
	const int Nz = grid.SizeZ();
	constexpr int MW = 4;

	const double c4z = -3.0 * coef / (840.0 * dz);
	const double c3z = 32.0 * coef / (840.0 * dz);
	const double c2z = -168.0 * coef / (840.0 * dz);
	const double c1z = 672.0 * coef / (840.0 * dz);

	const int size_x = grid.SizeX() + MW * 2;
	const int size_y = grid.SizeY() + MW * 2;
	const int size_z = grid.SizeZ() + MW * 2;

	for (int iz = MW; iz < size_z - MW; ++iz) {
		const int idz_m4 = -4 * size_x * size_y;
		const int idz_m3 = -3 * size_x * size_y;
		const int idz_m2 = -2 * size_x * size_y;
		const int idz_m = -1 * size_x * size_y;
		const int idz_p = 1 * size_x * size_y;
		const int idz_p2 = 2 * size_x * size_y;
		const int idz_p3 = 3 * size_x * size_y;
		const int idz_p4 = 4 * size_x * size_y;

		for (int iy = MW; iy < size_y - MW; ++iy) {
			const int idy_m4 = -4 * size_x;
			const int idy_m3 = -3 * size_x;
			const int idy_m2 = -2 * size_x;
			const int idy_m = -1 * size_x;
			const int idy_p = 1 * size_x;
			const int idy_p2 = 2 * size_x;
			const int idy_p3 = 3 * size_x;
			const int idy_p4 = 4 * size_x;
#pragma ivdep
			for (int ix = MW; ix < size_x - MW; ++ix) {
				const int idx_m4 = -4;
				const int idx_m3 = -3;
				const int idx_m2 = -2;
				const int idx_m = -1;
				const int idx_p = 1;
				const int idx_p2 = 2;
				const int idx_p3 = 3;
				const int idx_p4 = 4;
				const int64_t i = ix + size_x * (iy + (size_y * iz));
				const int64_t io = (ix - MW) + Nx * (iy - MW + (Ny * (iz - MW)));

				auto dpsi_dz = c4z * (p[i + idz_p4] - p[i + idz_m4]) + c3z * (p[i + idz_p3] - p[i + idz_m3]) + c2z * (p[i + idz_p2] - p[i + idz_m2]) + c1z * (p[i + idz_p] - p[i + idz_m]);
				out[io] = dpsi_dz;
			}
		}
	}
}


template <class T>
void Gradient8th_halo(const GridRange& grid, T* out_x, T* out_y, T* out_z, const T* src, T* buf_with_halo, const double coef, const double dx, const double dy, const double dz) {
	const int size_x = grid.SizeX();
	const int size_y = grid.SizeY();
	const int size_z = grid.SizeZ();

	PasteHalo_f4(grid, buf_with_halo,src,
		[&](int ix, int iy, int iz) { return src[(ix + size_x * (iy + 1 + size_y * iz))]; },
		[&](int ix, int iy, int iz) { return src[(ix + size_x * (iy - 1 + size_y * iz))]; },
		[&](int ix, int iy, int iz) { return src[(ix + size_x * (iy + size_y * (iz + 1)))]; },
		[&](int ix, int iy, int iz) { return src[(ix + size_x * (iy + size_y * (iz - 1)))]; },
		[&](int ix, int iy, int iz) { return src[(ix + size_x * (iy + size_y * (iz + size_z)))]; },
		[&](int ix, int iy, int iz) { return src[(ix + size_x * (iy + size_y * (iz - size_z)))]; }	);

	const size_t local_size = grid.Size3D();
	Gradient8th_x_halo_f(grid, out_x, buf_with_halo, coef, dx, dy, dz);
	Gradient8th_y_halo_f(grid, out_y, buf_with_halo, coef, dx, dy, dz);
	Gradient8th_z_halo_f(grid, out_z, buf_with_halo, coef, dx, dy, dz);


}


/*
* wrapper of the fastest algorithm of 8th order difference equation
* 
*/
template <class T>
void Gradient8th_zero(const GridRange& grid, T* out_x, T* out_y, T* out_z, const T* src, double coef, double dx, double dy, double dz) {
	
	const int margin_width = 4;
	const int over_size = (grid.SizeX() + 2 * margin_width) * (grid.SizeY() + 2 * margin_width) * (grid.SizeZ() + 2 * margin_width);
	auto over_data = std::make_unique<T[]>(over_size);//margin data//

	const int size_x = grid.SizeX();
    const int size_y = grid.SizeY();
    const int size_z = grid.SizeZ();

    PasteHalo_f4(grid, &over_data[0], src,
        [&](int ix, int iy, int iz) { return 0.0; },
        [&](int ix, int iy, int iz) { return 0.0; },
        [&](int ix, int iy, int iz) { return 0.0; },
        [&](int ix, int iy, int iz) { return 0.0; },
        [&](int ix, int iy, int iz) { return 0.0; },
        [&](int ix, int iy, int iz) { return 0.0; });

    const size_t local_size = grid.Size3D();
    Gradient8th_x_halo_f(grid, out_x, &over_data[0], coef, dx, dy, dz);
    Gradient8th_y_halo_f(grid, out_y, &over_data[0], coef, dx, dy, dz);
    Gradient8th_z_halo_f(grid, out_z, &over_data[0], coef, dx, dy, dz);

}


/*
* wrapper of the fastest algorithm of 8th order difference equation
*
*/
template <class T>
void Gradient8th(const GridRange& grid, T* out_x, T* out_y, T* out_z, const T* p, double coef, double dx, double dy, double dz) {

    const int margin_width = 4;
    const int over_size = (grid.SizeX() + 2 * margin_width) * (grid.SizeY() + 2 * margin_width) * (grid.SizeZ() + 2 * margin_width);
    auto over_data = std::make_unique<T[]>(over_size);//margin data//

    Gradient8th_halo(grid, out_x, out_y, out_z, p, &over_data[0], coef, dx, dy, dz);
}



#ifdef USE_MPI


/*
* 1次元領域分割の場合のGradient
* 分割方向はz方向
*/
template <class T>
void Gradient8th_MPI_1D_overspace(const GridRangeMPI& grid, T* out_x, T* out_y, T* out_z, const T* src, T* buf_with_halo, double coef, double dx, double dy, double dz) {

	MakeOverspace_1D_mpi(grid, buf_with_halo, src);

	Gradient8th_x_halo_f(grid, out_x, buf_with_halo, coef, dx, dy, dz);
	Gradient8th_y_halo_f(grid, out_y, buf_with_halo, coef, dx, dy, dz);
	Gradient8th_z_halo_f(grid, out_z, buf_with_halo, coef, dx, dy, dz);

}


/*
* 3次元領域分割の場合のGradient
*/
template <class T>
void Gradient8th_MPI_3D_overspace(const GridRangeMPI& grid, T* out_x, T* out_y, T* out_z, const T* src, T* buf_with_halo, double coef, double dx, double dy, double dz) {

   	MakeOverspace8th_3D_mpi(grid, buf_with_halo, src);

	Gradient8th_x_halo_f(grid, out_x, buf_with_halo, coef, dx, dy, dz);
	Gradient8th_y_halo_f(grid, out_y, buf_with_halo, coef, dx, dy, dz);
	Gradient8th_z_halo_f(grid, out_z, buf_with_halo, coef, dx, dy, dz);
}


template <class T>
void Gradient8th_ddm(const GridRangeMPI& grid, T* out_x, T* out_y, T* out_z, const T* src, double coef, double dx, double dy, double dz) {

	const int margin_width = 4;
	const int over_size = (grid.SizeX() + 2 * margin_width) * (grid.SizeY() + 2 * margin_width) * (grid.SizeZ() + 2 * margin_width);
	auto buf_with_halo = std::make_unique<T[]>(over_size);//margin data//

#ifdef TEST_PRINT
    printf("buf_with_halo = %zd\n", buf_with_halo.size()); fflush(stdout);
#endif
	switch (grid.split_dimension) {
	case 1:
		Gradient8th_MPI_1D_overspace(grid, out_x, out_y, out_z, src, &buf_with_halo[0], coef, dx, dy, dz);
		break;
	case 2:
		Gradient8th_MPI_3D_overspace(grid, out_x, out_y, out_z, src, &buf_with_halo[0], coef, dx, dy, dz);
		break;
	case 3:		
		Gradient8th_MPI_3D_overspace(grid, out_x, out_y, out_z, src, &buf_with_halo[0], coef, dx, dy, dz);
		break;
	default:
		Gradient8th_halo(grid, out_x, out_y, out_z, src, &buf_with_halo[0], coef, dx, dy, dz);
		break;
	}
}



template <class T>
void GradientX_8th_ddm(const GridRangeMPI& grid, T* out_x, const T* src, double coef, double dx) {

    const int margin_width = 4;
    const int over_size = (grid.SizeX() + 2 * margin_width) * (grid.SizeY() + 2 * margin_width) * (grid.SizeZ() + 2 * margin_width);
    auto buf_with_halo = std::make_unique<T[]>(over_size);//margin data//

#ifdef TEST_PRINT
    printf("buf_with_halo = %zd\n", buf_with_halo.size()); fflush(stdout);
#endif
    
    MakeOverspace8th_3D_mpi(grid, &buf_with_halo[0], src);

    Gradient8th_x_halo_f(grid, out_x, &buf_with_halo[0], coef, dx, dx, dx);
}

template <class T>
void GradientY_8th_ddm(const GridRangeMPI& grid, T* out_y, const T* src, double coef, double dy) {

    const int margin_width = 4;
    const int over_size = (grid.SizeX() + 2 * margin_width) * (grid.SizeY() + 2 * margin_width) * (grid.SizeZ() + 2 * margin_width);
    auto buf_with_halo = std::make_unique<T[]>(over_size);//margin data//

#ifdef TEST_PRINT
    printf("buf_with_halo = %zd\n", buf_with_halo.size()); fflush(stdout);
#endif

    MakeOverspace8th_3D_mpi(grid, &buf_with_halo[0], src);

    Gradient8th_y_halo_f(grid, out_y, &buf_with_halo[0], coef, dy, dy, dy);
}

template <class T>
void GradientZ_8th_ddm(const GridRangeMPI& grid, T* out_z, const T* src, double coef, double dz) {

    const int margin_width = 4;
    const int over_size = (grid.SizeX() + 2 * margin_width) * (grid.SizeY() + 2 * margin_width) * (grid.SizeZ() + 2 * margin_width);
    auto buf_with_halo = std::make_unique<T[]>(over_size);//margin data//

#ifdef TEST_PRINT
    printf("buf_with_halo = %zd\n", buf_with_halo.size()); fflush(stdout);
#endif

    MakeOverspace8th_3D_mpi(grid, &buf_with_halo[0], src);

    Gradient8th_z_halo_f(grid, out_z, &buf_with_halo[0], coef, dz, dz, dz);
}




template <class T>
void SecondDifferential8th_halo_f(const GridRange& grid, T* out_xx, T* out_yy, T* out_zz, const T* p, double coef, double dx, double dy, double dz) {

    const int Nx = grid.SizeX();
    const int Ny = grid.SizeY();
    const int Nz = grid.SizeZ();
    constexpr int MW = 4;

    //const double c0 = -14350.0 / 5040.0 * coef * (1.0 / (dx * dx) + 1.0 / (dy * dy) + 1.0 / (dz * dz));
    const double c0x = -14350.0 / 5040.0 * coef * (1.0 / (dx * dx) );
    const double c0y = -14350.0 / 5040.0 * coef * (1.0 / (dy * dy) );
    const double c0z = -14350.0 / 5040.0 * coef * (1.0 / (dz * dz) );

    const double c4x = -9.0 * coef / (5040.0 * dx * dx);
    const double c3x = 128.0 * coef / (5040.0 * dx * dx);
    const double c2x = -1008.0 * coef / (5040.0 * dx * dx);
    const double c1x = 8064.0 * coef / (5040.0 * dx * dx);
    const double c4y = -9.0 * coef / (5040.0 * dy * dy);
    const double c3y = 128.0 * coef / (5040.0 * dy * dy);
    const double c2y = -1008.0 * coef / (5040.0 * dy * dy);
    const double c1y = 8064.0 * coef / (5040.0 * dy * dy);
    const double c4z = -9.0 * coef / (5040.0 * dz * dz);
    const double c3z = 128.0 * coef / (5040.0 * dz * dz);
    const double c2z = -1008.0 * coef / (5040.0 * dz * dz);
    const double c1z = 8064.0 * coef / (5040.0 * dz * dz);

    const int size_x = grid.SizeX() + MW * 2;
    const int size_y = grid.SizeY() + MW * 2;
    const int size_z = grid.SizeZ() + MW * 2;

    for (int iz = MW; iz < size_z - MW; ++iz) {
        const int idz_m4 = -4 * size_x * size_y;
        const int idz_m3 = -3 * size_x * size_y;
        const int idz_m2 = -2 * size_x * size_y;
        const int idz_m = -1 * size_x * size_y;
        const int idz_p = 1 * size_x * size_y;
        const int idz_p2 = 2 * size_x * size_y;
        const int idz_p3 = 3 * size_x * size_y;
        const int idz_p4 = 4 * size_x * size_y;

        for (int iy = MW; iy < size_y - MW; ++iy) {
            const int idy_m4 = -4 * size_x;
            const int idy_m3 = -3 * size_x;
            const int idy_m2 = -2 * size_x;
            const int idy_m = -1 * size_x;
            const int idy_p = 1 * size_x;
            const int idy_p2 = 2 * size_x;
            const int idy_p3 = 3 * size_x;
            const int idy_p4 = 4 * size_x;
#pragma ivdep
            for (int ix = MW; ix < size_x - MW; ++ix) {
                const int idx_m4 = -4;
                const int idx_m3 = -3;
                const int idx_m2 = -2;
                const int idx_m = -1;
                const int idx_p = 1;
                const int idx_p2 = 2;
                const int idx_p3 = 3;
                const int idx_p4 = 4;
                const int64_t i = ix + size_x * (iy + (size_y * iz));
                const int64_t io = (ix - MW) + Nx * (iy - MW + (Ny * (iz - MW)));


                out_xx[io] = c0x * p[i] + c4x * (p[i + idx_p4] + p[i + idx_m4]) + c3x * (p[i + idx_p3] + p[i + idx_m3]) + c2x * (p[i + idx_p2] + p[i + idx_m2]) + c1x * (p[i + idx_p] + p[i + idx_m]);
                out_yy[io] = c0y * p[i] + c4y * (p[i + idy_p4] + p[i + idy_m4]) + c3y * (p[i + idy_p3] + p[i + idy_m3]) + c2y * (p[i + idy_p2] + p[i + idy_m2]) + c1y * (p[i + idy_p] + p[i + idy_m]);
                out_zz[io] = c0z * p[i] + c4z * (p[i + idz_p4] + p[i + idz_m4]) + c3z * (p[i + idz_p3] + p[i + idz_m3]) + c2z * (p[i + idz_p2] + p[i + idz_m2]) + c1z * (p[i + idz_p] + p[i + idz_m]);
            }
        }
    }
}



template <class T>
void NineDifferentials8th_ddm(const GridRangeMPI& grid, 
    T* out_x, T* out_y, T* out_z, T* out_xx, T* out_yy, T* out_zz, T* out_xz, T* out_yz, T* out_xy, 
    const T* src, double coef, double dx, double dy, double dz) {

    const int64_t margin_width = 4;
    const int64_t over_size = (grid.SizeX() + 2 * margin_width) * (grid.SizeY() + 2 * margin_width) * (grid.SizeZ() + 2 * margin_width);
    const int64_t local_size = grid.Size3D();
    auto buf_with_halo = std::make_unique<T[]>(over_size);//margin data//

    
    //out_x,y,zを求める//
    switch (grid.split_dimension) {
    case 1:

        Gradient8th_MPI_1D_overspace(grid, out_x, out_y, out_z, src, &buf_with_halo[0], coef, dx, dy, dz);
        SecondDifferential8th_halo_f(grid, out_xx, out_yy, out_zz, &buf_with_halo[0], coef, dx, dy, dz);

        break;
    case 2:
        Gradient8th_MPI_3D_overspace(grid, out_x, out_y, out_z, src, &buf_with_halo[0], coef, dx, dy, dz);
        SecondDifferential8th_halo_f(grid, out_xx, out_yy, out_zz, &buf_with_halo[0], coef, dx, dy, dz);
        break;
    case 3:
        Gradient8th_MPI_3D_overspace(grid, out_x, out_y, out_z, src, &buf_with_halo[0], coef, dx, dy, dz);
        SecondDifferential8th_halo_f(grid, out_xx, out_yy, out_zz, &buf_with_halo[0], coef, dx, dy, dz);
        break;
    default:
        Gradient8th_halo(grid, out_x, out_y, out_z, src, &buf_with_halo[0], coef, dx, dy, dz);
        SecondDifferential8th_halo_f(grid, out_xx, out_yy, out_zz, &buf_with_halo[0], coef, dx, dy, dz);
        break;
    }


    //out_zx//
    MakeOverspace8th_3D_mpi(grid, &buf_with_halo[0], out_z);
    Gradient8th_x_halo_f(grid, out_xz, &buf_with_halo[0], coef, dx, dy, dz);
    Gradient8th_y_halo_f(grid, out_yz, &buf_with_halo[0], coef, dx, dy, dz);

    //out_xy//
    MakeOverspace8th_3D_mpi(grid, &buf_with_halo[0], out_y);
    Gradient8th_x_halo_f(grid, out_xy, &buf_with_halo[0], coef, dx, dy, dz);

    
}

/*
* input(src)から次の量を計算する 
* out_sq_nabla: (\nabla src)*(\nabla src)
* out_laplace: (\nabla * \nabla src) = (\Delta src)
* out_nabla_vmv: \sum_{a,b} (\nabla_a src)*(\nabla_a \nabla_b src)*(\nabla_b src) 
*/
template <class T>
void SquareGradient8th_ddm(const GridRangeMPI& grid, T* out_sq_nabla, T* out_laplace, T* out_nabla_vmv, const T* src, double coef, double dx, double dy, double dz) {

    const int64_t local_size = grid.Size3D();
    
    double* out_x = new T[local_size * 9];
    double* out_y = out_x + local_size;
    double* out_z = out_x + local_size*2;
    double* out_xx = out_x + local_size * 3;
    double* out_yy = out_x + local_size * 4;
    double* out_zz = out_x + local_size * 5;
    double* out_xz = out_x + local_size * 6;
    double* out_yz = out_x + local_size * 7;
    double* out_xy = out_x + local_size * 8;

    NineDifferentials8th_ddm(grid, out_x, out_y, out_z, out_xx, out_yy, out_zz, out_xz, out_yz, out_xy, src, coef, dx, dy, dz);

    for (int64_t i = 0; i < local_size; ++i) {
        out_sq_nabla[i] = out_x[i] * out_x[i] + out_y[i] * out_y[i] + out_z[i] * out_z[i];
        out_laplace[i] = out_xx[i] + out_yy[i] + out_zz[i];
        out_nabla_vmv[i] = out_x[i] * (out_xx[i] * out_x[i] + out_xy[i] * out_y[i] + out_xz[i] * out_z[i])
            + out_y[i] * (out_xy[i] * out_x[i] + out_yy[i] * out_y[i] + out_yz[i] * out_z[i])
            + out_z[i] * (out_xz[i] * out_x[i] + out_yz[i] * out_y[i] + out_zz[i] * out_z[i]);
    }

    delete[] out_x;
}



#endif

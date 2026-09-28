#pragma once
#include "GridFor.h"
#include "GridSubgrid.h"
//#include "GridSubgrid_arithmetic.h"
#include "GridVoxelSphere.h"
#include "RealSphericalHarmonics.h"


template<class YLM>
void SetBlockYlmAroundAny(YLM Y, double value_org, double* buf, const std::vector<GridRange>& range_blocks,
	const std::vector<ShiftedGrid>& shifted_grid,
	double Rx, double Ry, double Rz,
	double dx, double dy, double dz)
{

	size_t num_blocks = range_blocks.size();
	constexpr double rr_min = 1.0e-14;
	int offset_i = 0;
	for (size_t ib = 0; ib < num_blocks; ++ib) {
		const int ix_begin = range_blocks[ib].begin_x- shifted_grid[ib].x;
		const int iy_begin = range_blocks[ib].begin_y- shifted_grid[ib].y;
		const int iz_begin = range_blocks[ib].begin_z- shifted_grid[ib].z;
		const int ix_end = range_blocks[ib].end_x- shifted_grid[ib].x;
		const int iy_end = range_blocks[ib].end_y- shifted_grid[ib].y;
		const int iz_end = range_blocks[ib].end_z- shifted_grid[ib].z;

		const int size_x = ix_end - ix_begin;
		const int size_y = iy_end - iy_begin;

		for (int iz = iz_begin; iz < iz_end; ++iz) {
			const double z = dz * (double)(iz) - Rz;
			const double zz = z * z;
			for (int iy = iy_begin; iy < iy_end; ++iy) {
				const double y = dy * (double)(iy) - Ry;
				const double yy_zz = y * y + zz;
				for (int ix = ix_begin; ix < ix_end; ++ix) {
					const double x = dx * (double)(ix) - Rx;
					const double rr = x * x + yy_zz;
					const int i = offset_i + (ix - ix_begin) + size_x * ((iy - iy_begin) + size_y * (iz - iz_begin));

					if (rr_min > rr) {
						buf[i] = value_org;
					} else {
						const double r = sqrt(rr);
						buf[i] = Y(x / r, y / r, z / r);
					}
				}
			}
		}
		offset_i += range_blocks[ib].Size3D();
	}

}

inline
void SetBlockYlmAround(int L, int M, double* buf, const std::vector<GridRange>& range_blocks,
	const std::vector<ShiftedGrid>& shifted_grid,
	double Rx, double Ry, double Rz,
	double dx, double dy, double dz)
{
	const double Y00 = 1.0 / sqrt(4.0 * M_PI);
	switch (L * L + L + M) {
	case 0:
	{
		Ylm<0, 0> Y;
		SetBlockYlmAroundAny(Y, Y00, buf, range_blocks, shifted_grid, Rx, Ry, Rz, dx, dy, dz);
		break;
	}
	case 1:
	{
		Ylm<1, -1> Y;
		SetBlockYlmAroundAny(Y, 0.0, buf, range_blocks, shifted_grid, Rx, Ry, Rz, dx, dy, dz);
		break;
	}
	case 2:
	{
		Ylm<1, 0> Y;
		SetBlockYlmAroundAny(Y, 0.0, buf, range_blocks, shifted_grid, Rx, Ry, Rz, dx, dy, dz);
		break;
	}
	case 3:
	{
		Ylm<1, 1> Y;
		SetBlockYlmAroundAny(Y, 0.0, buf, range_blocks, shifted_grid, Rx, Ry, Rz, dx, dy, dz);
		break;
	}
	case 4:
	{
		Ylm<2, -2> Y;
		SetBlockYlmAroundAny(Y, 0.0, buf, range_blocks, shifted_grid, Rx, Ry, Rz, dx, dy, dz);
		break;
	}
	case 5:
	{
		Ylm<2, -1> Y;
		SetBlockYlmAroundAny(Y, 0.0, buf, range_blocks, shifted_grid, Rx, Ry, Rz, dx, dy, dz);
		break;
	}
	case 6:
	{
		Ylm<2, 0> Y;
		SetBlockYlmAroundAny(Y, 0.0, buf, range_blocks, shifted_grid, Rx, Ry, Rz, dx, dy, dz);
		break;
	}
	case 7:
	{
		Ylm<2, 1> Y;
		SetBlockYlmAroundAny(Y, 0.0, buf, range_blocks, shifted_grid, Rx, Ry, Rz, dx, dy, dz);
		break;
	}
	case 8:
	{
		Ylm<2, 2> Y;
		SetBlockYlmAroundAny(Y, 0.0, buf, range_blocks, shifted_grid, Rx, Ry, Rz, dx, dy, dz);
		break;
	}
	case 9:
	{
		Ylm<3, -3> Y;
		SetBlockYlmAroundAny(Y, 0.0, buf, range_blocks, shifted_grid, Rx, Ry, Rz, dx, dy, dz);
		break;
	}
	case 10:
	{
		Ylm<3, -2> Y;
		SetBlockYlmAroundAny(Y, 0.0, buf, range_blocks, shifted_grid, Rx, Ry, Rz, dx, dy, dz);
		break;
	}
	case 11:
	{
		Ylm<3, -1> Y;
		SetBlockYlmAroundAny(Y, 0.0, buf, range_blocks, shifted_grid, Rx, Ry, Rz, dx, dy, dz);
		break;
	}
	case 12:
	{
		Ylm<3, 0> Y;
		SetBlockYlmAroundAny(Y, 0.0, buf, range_blocks, shifted_grid, Rx, Ry, Rz, dx, dy, dz);
		break;
	}
	case 13:
	{
		Ylm<3, 1> Y;
		SetBlockYlmAroundAny(Y, 0.0, buf, range_blocks, shifted_grid, Rx, Ry, Rz, dx, dy, dz);
		break;
	}
	case 14:
	{
		Ylm<3, 2> Y;
		SetBlockYlmAroundAny(Y, 0.0, buf, range_blocks, shifted_grid, Rx, Ry, Rz, dx, dy, dz);
		break;
	}
	case 15:
	{
		Ylm<3, 3> Y;
		SetBlockYlmAroundAny(Y, 0.0, buf, range_blocks, shifted_grid, Rx, Ry, Rz, dx, dy, dz);
		break;
	}
	default:
	{
		YlmSimple Y(L, M);
		SetBlockYlmAroundAny(Y, 0.0, buf, range_blocks, shifted_grid, Rx, Ry, Rz, dx, dy, dz);
		break;
	}
	}
}

template<class YLM>
void SetVsYlmAroundAny(YLM& Y, double value_org, double* buf, const std::vector<int>& voxel_lines,
	const std::vector<ShiftedPos>& shifted_pos,
	double dx, double dy, double dz)
{


	constexpr double rr_min = 1.0e-14;

	int total_lines = 0;
	int total_i = 0;
	for (const auto& pos : shifted_pos) {
		for (int il = 0; il < pos.num_lines; ++il) {
			const int index = (il + total_lines) * 4;
			const int iz = voxel_lines[index];
			const int iy = voxel_lines[index + 1];
			const int ix_begin = voxel_lines[index + 2];
			const int ix_end = ix_begin + voxel_lines[index + 3];

			const double z = dz * (double)iz - pos.z;
			const double y = dy * (double)iy - pos.y;
			const double yy_zz = y * y + z * z;
			for (int ix = ix_begin; ix < ix_end; ++ix) {
				const double x = dx * (double)ix - pos.x;
				const double rr = x * x + yy_zz;

				if (rr_min > rr) {
					buf[total_i + ix - ix_begin] = value_org;
				} else {
					const double r = sqrt(rr);
					buf[total_i + ix - ix_begin] = Y(x / r, y / r, z / r);
				}
			}
			total_i += voxel_lines[index + 3];
		}
		total_lines += pos.num_lines;
	}

}

inline
void SetVsYlmAround(int L, int M, double* buf, const std::vector<int>& voxel_lines,
	const std::vector<ShiftedPos>& shifted_pos,
	double dx, double dy, double dz)
{
	const double Y00 = 1.0 / sqrt(4.0 * M_PI);
	switch (L * L + L + M) {
	case 0:
	{
		Ylm<0, 0> Y;
		SetVsYlmAroundAny(Y, Y00, buf, voxel_lines, shifted_pos, dx, dy, dz);
		break;
	}
	case 1:
	{
		Ylm<1, -1> Y;
		SetVsYlmAroundAny(Y, 0.0, buf, voxel_lines, shifted_pos, dx, dy, dz);
		break;
	}
	case 2:
	{
		Ylm<1, 0> Y;
		SetVsYlmAroundAny(Y, 0.0, buf, voxel_lines, shifted_pos, dx, dy, dz);
		break;
	}
	case 3:
	{
		Ylm<1, 1> Y;
		SetVsYlmAroundAny(Y, 0.0, buf, voxel_lines, shifted_pos, dx, dy, dz);
		break;
	}
	case 4:
	{
		Ylm<2, -2> Y;
		SetVsYlmAroundAny(Y, 0.0, buf, voxel_lines, shifted_pos, dx, dy, dz);
		break;
	}
	case 5:
	{
		Ylm<2, -1> Y;
		SetVsYlmAroundAny(Y, 0.0, buf, voxel_lines, shifted_pos, dx, dy, dz);
		break;
	}
	case 6:
	{
		Ylm<2, 0> Y;
		SetVsYlmAroundAny(Y, 0.0, buf, voxel_lines, shifted_pos, dx, dy, dz);
		break;
	}
	case 7:
	{
		Ylm<2, 1> Y;
		SetVsYlmAroundAny(Y, 0.0, buf, voxel_lines, shifted_pos, dx, dy, dz);
		break;
	}
	case 8:
	{
		Ylm<2, 2> Y;
		SetVsYlmAroundAny(Y, 0.0, buf, voxel_lines, shifted_pos, dx, dy, dz);
		break;
	}
	case 9:
	{
		Ylm<3, -3> Y;
		SetVsYlmAroundAny(Y, 0.0, buf, voxel_lines, shifted_pos, dx, dy, dz);
		break;
	}
	case 10:
	{
		Ylm<3, -2> Y;
		SetVsYlmAroundAny(Y, 0.0, buf, voxel_lines, shifted_pos, dx, dy, dz);
		break;
	}
	case 11:
	{
		Ylm<3, -1> Y;
		SetVsYlmAroundAny(Y, 0.0, buf, voxel_lines, shifted_pos, dx, dy, dz);
		break;
	}
	case 12:
	{
		Ylm<3, 0> Y;
		SetVsYlmAroundAny(Y, 0.0, buf, voxel_lines, shifted_pos, dx, dy, dz);
		break;
	}
	case 13:
	{
		Ylm<3, 1> Y;
		SetVsYlmAroundAny(Y, 0.0, buf, voxel_lines, shifted_pos, dx, dy, dz);
		break;
	}
	case 14:
	{
		Ylm<3, 2> Y;
		SetVsYlmAroundAny(Y, 0.0, buf, voxel_lines, shifted_pos, dx, dy, dz);
		break;
	}
	case 15:
	{
		Ylm<3, 3> Y;
		SetVsYlmAroundAny(Y, 0.0, buf, voxel_lines, shifted_pos, dx, dy, dz);
		break;
	}
	default:
	{
		YlmSimple Y(L, M);
		SetVsYlmAroundAny(Y, 0.0, buf, voxel_lines, shifted_pos, dx, dy, dz);
		break;
	}
	}
}




template<class YLM>
void SetYlmAnyAtCenter(YLM Y, double Y_at_origin, double* dest, int size_x, int size_y, int size_z,
    double dx, double dy, double dz, double cutoff_length)
{
    const double rr_max = cutoff_length * cutoff_length;
    const int half_x = size_x / 2;
    const int half_y = size_y / 2;
    const int half_z = size_z / 2;

    for (int iz = 0; iz < size_z; ++iz) {
        const double z = dz * (double)((iz >= half_z) ? iz - size_z : iz);
        const double zz = z * z;
        for (int iy = 0; iy < size_y; ++iy) {
            const double y = dy * (double)((iy >= half_y) ? iy - size_y : iy);
            const double yy_zz = y * y + zz;
            for (int ix = 0; ix < size_x; ++ix) {
                const double x = dx * (double)((ix >= half_x) ? ix - size_x : ix);
                const double rr = x * x + yy_zz;
                const int i = ix + size_x * (iy + size_y * iz);

                if (i == 0) {
                    dest[i] = Y_at_origin;
                } else {
                    const double r = sqrt(rr);
                    dest[i] = Y(x / r, y / r, z / r);
                }
                /*
                if (rr > rr_max) {
                    dest[i] *= exp(-(rr - rr_max));
                }
                */
            }
        }
    }


}

inline
void SetYlmAtCenter(int L, int M, double* dest,
    int size_x, int size_y, int size_z, double dx, double dy, double dz, double cutoff_length)
{
    const double Y00 = 1.0 / sqrt(4.0 * M_PI);
    switch (L * L + L + M) {
    case 0:
    {
        Ylm<0, 0> Y;
        SetYlmAnyAtCenter(Y, Y00, dest, size_x, size_y, size_z, dx, dy, dz, cutoff_length);
        break;
    }
    case 1:
    {
        Ylm<1, -1> Y;
        SetYlmAnyAtCenter(Y, 0.0, dest, size_x, size_y, size_z, dx, dy, dz, cutoff_length);
        break;
    }
    case 2:
    {
        Ylm<1, 0> Y;
        SetYlmAnyAtCenter(Y, 0.0, dest, size_x, size_y, size_z, dx, dy, dz, cutoff_length);
        break;
    }
    case 3:
    {
        Ylm<1, 1> Y;
        SetYlmAnyAtCenter(Y, 0.0, dest, size_x, size_y, size_z, dx, dy, dz, cutoff_length);
        break;
    }
    case 4:
    {
        Ylm<2, -2> Y;
        SetYlmAnyAtCenter(Y, 0.0, dest, size_x, size_y, size_z, dx, dy, dz, cutoff_length);
        break;
    }
    case 5:
    {
        Ylm<2, -1> Y;
        SetYlmAnyAtCenter(Y, 0.0, dest, size_x, size_y, size_z, dx, dy, dz, cutoff_length);
        break;
    }
    case 6:
    {
        Ylm<2, 0> Y;
        SetYlmAnyAtCenter(Y, 0.0, dest, size_x, size_y, size_z, dx, dy, dz, cutoff_length);
        break;
    }
    case 7:
    {
        Ylm<2, 1> Y;
        SetYlmAnyAtCenter(Y, 0.0, dest, size_x, size_y, size_z, dx, dy, dz, cutoff_length);
        break;
    }
    case 8:
    {
        Ylm<2, 2> Y;
        SetYlmAnyAtCenter(Y, 0.0, dest, size_x, size_y, size_z, dx, dy, dz, cutoff_length);
        break;
    }
    case 9:
    {
        Ylm<3, -3> Y;
        SetYlmAnyAtCenter(Y, 0.0, dest, size_x, size_y, size_z, dx, dy, dz, cutoff_length);
        break;
    }
    case 10:
    {
        Ylm<3, -2> Y;
        SetYlmAnyAtCenter(Y, 0.0, dest, size_x, size_y, size_z, dx, dy, dz, cutoff_length);
        break;
    }
    case 11:
    {
        Ylm<3, -1> Y;
        SetYlmAnyAtCenter(Y, 0.0, dest, size_x, size_y, size_z, dx, dy, dz, cutoff_length);
        break;
    }
    case 12:
    {
        Ylm<3, 0> Y;
        SetYlmAnyAtCenter(Y, 0.0, dest, size_x, size_y, size_z, dx, dy, dz, cutoff_length);
        break;
    }
    case 13:
    {
        Ylm<3, 1> Y;
        SetYlmAnyAtCenter(Y, 0.0, dest, size_x, size_y, size_z, dx, dy, dz, cutoff_length);
        break;
    }
    case 14:
    {
        Ylm<3, 2> Y;
        SetYlmAnyAtCenter(Y, 0.0, dest, size_x, size_y, size_z, dx, dy, dz, cutoff_length);
        break;
    }
    case 15:
    {
        Ylm<3, 3> Y;
        SetYlmAnyAtCenter(Y, 0.0, dest, size_x, size_y, size_z, dx, dy, dz, cutoff_length);
        break;
    }
    default:
    {
        YlmSimple Y(L, M);
        SetYlmAnyAtCenter(Y, 0.0, dest, size_x, size_y, size_z, dx, dy, dz, cutoff_length);
        break;
    }
    }
}



template<class YLM>
void SetYlmAnyRange(YLM Y, double Y_at_origin, double* dest, const GridRange& grid,
    double dx, double dy, double dz)
{
    const int ix_begin = grid.begin_x;
    const int iy_begin = grid.begin_y;
    const int iz_begin = grid.begin_z;
    const int ix_end = grid.end_x;
    const int iy_end = grid.end_y;
    const int iz_end = grid.end_z;
    const int size_x = ix_end - ix_begin;
    const int size_y = iy_end - iy_begin;

    for (int iz = iz_begin; iz < iz_end; ++iz) {
        const double z = dz * (double)iz;
        const double zz = z * z;
        for (int iy = iy_begin; iy < iy_end; ++iy) {
            const double y = dy * (double)iy;
            const double yy_zz = y * y + zz;
            for (int ix = ix_begin; ix < ix_end; ++ix) {
                const double x = dx * (double)ix;
                const double rr = x * x + yy_zz;
                const int i = (ix - ix_begin) + size_x * ((iy - iy_begin) + size_y * (iz - iz_begin));

                if ((ix == 0) && (iy == 0) && (iz == 0)) {
                    dest[i] = Y_at_origin;
                } else {
                    const double r = sqrt(rr);
                    dest[i] = Y(x / r, y / r, z / r);
                }
            }
        }
    }


}


template<class YLM>
void SetYlmAnyRangeFromOrigin(YLM Y, double Y_at_origin, double* dest, const GridRange& grid,
    double dx, double dy, double dz, double org_x, double org_y, double org_z)
{
    const int ix_begin = grid.begin_x;
    const int iy_begin = grid.begin_y;
    const int iz_begin = grid.begin_z;
    const int ix_end = grid.end_x;
    const int iy_end = grid.end_y;
    const int iz_end = grid.end_z;
    const int size_x = ix_end - ix_begin;
    const int size_y = iy_end - iy_begin;

    for (int iz = iz_begin; iz < iz_end; ++iz) {
        const double z = dz * ((double)iz - org_z);
        const double zz = z * z;
        for (int iy = iy_begin; iy < iy_end; ++iy) {
            const double y = dy * ((double)iy -org_y);
            const double yy_zz = y * y + zz;
            for (int ix = ix_begin; ix < ix_end; ++ix) {
                const double x = dx * ((double)ix - org_x);
                const double rr = x * x + yy_zz;
                const int i = (ix - ix_begin) + size_x * ((iy - iy_begin) + size_y * (iz - iz_begin));

                if (rr < 1.0e-14) {
                    dest[i] = Y_at_origin;
                } else {
                    const double r = sqrt(rr);
                    dest[i] = Y(x / r, y / r, z / r);
                }
            }
        }
    }


}

inline
void SetYlmAnyRange(int L, int M, double* dest,
    const GridRange& grid, double dx, double dy, double dz)
{
    const double Y00 = 1.0 / sqrt(4.0 * M_PI);
    switch (L * L + L + M) {
    case 0:
    {
        Ylm<0, 0> Y;
        SetYlmAnyRange(Y, Y00, dest, grid, dx, dy, dz);
        break;
    }
    case 1:
    {
        Ylm<1, -1> Y;
        SetYlmAnyRange(Y, 0.0, dest, grid, dx, dy, dz);
        break;
    }
    case 2:
    {
        Ylm<1, 0> Y;
        SetYlmAnyRange(Y, 0.0, dest, grid, dx, dy, dz);
        break;
    }
    case 3:
    {
        Ylm<1, 1> Y;
        SetYlmAnyRange(Y, 0.0, dest, grid, dx, dy, dz);
        break;
    }
    case 4:
    {
        Ylm<2, -2> Y;
        SetYlmAnyRange(Y, 0.0, dest, grid, dx, dy, dz);
        break;
    }
    case 5:
    {
        Ylm<2, -1> Y;
        SetYlmAnyRange(Y, 0.0, dest, grid, dx, dy, dz);
        break;
    }
    case 6:
    {
        Ylm<2, 0> Y;
        SetYlmAnyRange(Y, 0.0, dest, grid, dx, dy, dz);
        break;
    }
    case 7:
    {
        Ylm<2, 1> Y;
        SetYlmAnyRange(Y, 0.0, dest, grid, dx, dy, dz);
        break;
    }
    case 8:
    {
        Ylm<2, 2> Y;
        SetYlmAnyRange(Y, 0.0, dest, grid, dx, dy, dz);
        break;
    }
    case 9:
    {
        Ylm<3, -3> Y;
        SetYlmAnyRange(Y, 0.0, dest, grid, dx, dy, dz);
        break;
    }
    case 10:
    {
        Ylm<3, -2> Y;
        SetYlmAnyRange(Y, 0.0, dest, grid, dx, dy, dz);
        break;
    }
    case 11:
    {
        Ylm<3, -1> Y;
        SetYlmAnyRange(Y, 0.0, dest, grid, dx, dy, dz);
        break;
    }
    case 12:
    {
        Ylm<3, 0> Y;
        SetYlmAnyRange(Y, 0.0, dest, grid, dx, dy, dz);
        break;
    }
    case 13:
    {
        Ylm<3, 1> Y;
        SetYlmAnyRange(Y, 0.0, dest, grid, dx, dy, dz);
        break;
    }
    case 14:
    {
        Ylm<3, 2> Y;
        SetYlmAnyRange(Y, 0.0, dest, grid, dx, dy, dz);
        break;
    }
    case 15:
    {
        Ylm<3, 3> Y;
        SetYlmAnyRange(Y, 0.0, dest, grid, dx, dy, dz);
        break;
    }
    default:
    {
        YlmSimple Y(L, M);
        SetYlmAnyRange(Y, 0.0, dest, grid, dx, dy, dz);
        break;
    }
    }
}


/*
* unit of origin is grid.
* That is, if org_x = 1.0, then x = dx * (ix - org_x)
*/
inline
void SetYlmAnyRangeFromOrigin(int L, int M, double* dest,
    const GridRange& grid, double dx, double dy, double dz, double org_x, double org_y, double org_z)
{
    const double Y00 = 1.0 / sqrt(4.0 * M_PI);
    switch (L * L + L + M) {
    case 0:
    {
        Ylm<0, 0> Y;
        SetYlmAnyRangeFromOrigin(Y, Y00, dest, grid, dx, dy, dz, org_x, org_y, org_z);
        break;
    }
    case 1:
    {
        Ylm<1, -1> Y;
        SetYlmAnyRangeFromOrigin(Y, 0.0, dest, grid, dx, dy, dz, org_x, org_y, org_z);
        break;
    }
    case 2:
    {
        Ylm<1, 0> Y;
        SetYlmAnyRangeFromOrigin(Y, 0.0, dest, grid, dx, dy, dz, org_x, org_y, org_z);
        break;
    }
    case 3:
    {
        Ylm<1, 1> Y;
        SetYlmAnyRangeFromOrigin(Y, 0.0, dest, grid, dx, dy, dz, org_x, org_y, org_z);
        break;
    }
    case 4:
    {
        Ylm<2, -2> Y;
        SetYlmAnyRangeFromOrigin(Y, 0.0, dest, grid, dx, dy, dz, org_x, org_y, org_z);
        break;
    }
    case 5:
    {
        Ylm<2, -1> Y;
        SetYlmAnyRangeFromOrigin(Y, 0.0, dest, grid, dx, dy, dz, org_x, org_y, org_z);
        break;
    }
    case 6:
    {
        Ylm<2, 0> Y;
        SetYlmAnyRangeFromOrigin(Y, 0.0, dest, grid, dx, dy, dz, org_x, org_y, org_z);
        break;
    }
    case 7:
    {
        Ylm<2, 1> Y;
        SetYlmAnyRangeFromOrigin(Y, 0.0, dest, grid, dx, dy, dz, org_x, org_y, org_z);
        break;
    }
    case 8:
    {
        Ylm<2, 2> Y;
        SetYlmAnyRangeFromOrigin(Y, 0.0, dest, grid, dx, dy, dz, org_x, org_y, org_z);
        break;
    }
    case 9:
    {
        Ylm<3, -3> Y;
        SetYlmAnyRangeFromOrigin(Y, 0.0, dest, grid, dx, dy, dz, org_x, org_y, org_z);
        break;
    }
    case 10:
    {
        Ylm<3, -2> Y;
        SetYlmAnyRangeFromOrigin(Y, 0.0, dest, grid, dx, dy, dz, org_x, org_y, org_z);
        break;
    }
    case 11:
    {
        Ylm<3, -1> Y;
        SetYlmAnyRangeFromOrigin(Y, 0.0, dest, grid, dx, dy, dz, org_x, org_y, org_z);
        break;
    }
    case 12:
    {
        Ylm<3, 0> Y;
        SetYlmAnyRangeFromOrigin(Y, 0.0, dest, grid, dx, dy, dz, org_x, org_y, org_z);
        break;
    }
    case 13:
    {
        Ylm<3, 1> Y;
        SetYlmAnyRangeFromOrigin(Y, 0.0, dest, grid, dx, dy, dz, org_x, org_y, org_z);
        break;
    }
    case 14:
    {
        Ylm<3, 2> Y;
        SetYlmAnyRangeFromOrigin(Y, 0.0, dest, grid, dx, dy, dz, org_x, org_y, org_z);
        break;
    }
    case 15:
    {
        Ylm<3, 3> Y;
        SetYlmAnyRangeFromOrigin(Y, 0.0, dest, grid, dx, dy, dz, org_x, org_y, org_z);
        break;
    }
    default:
    {
        YlmSimple Y(L, M);
        SetYlmAnyRangeFromOrigin(Y, 0.0, dest, grid, dx, dy, dz, org_x, org_y, org_z);
        break;
    }
    }
}

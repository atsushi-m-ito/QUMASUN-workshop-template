#pragma once
#include "nucleus.h"
#include "GridSubgrid.h"

//グローバルなグリッド空間に対して、各原子を中心に指定の半径内を1とするフィルターを作用してマスクを生成
template<class FILTER>
void MakeAtomicMask(int global_x, int global_y, int global_z, 
    double dx, double dy, double dz,
    const Nucleus* nuclei, int num_nuclei, double cutoff_length, FILTER filter)
{
    const int margin_grid = 1;
    auto subgrid_org0 = SubgridFromCutoff(cutoff_length, dx, dy, dz, margin_grid);//原点中心のサブグリッド//
    GridRange global_grid{ 0,0,0, global_x, global_y, global_z };

    for (int ni = 0; ni < num_nuclei; ++ni) {

        const double Rx = nuclei[ni].Rx;
        const double Ry = nuclei[ni].Ry;
        const double Rz = nuclei[ni].Rz;

        const int irx = GridPosForSvF(Rx, dx);
        const int iry = GridPosForSvF(Ry, dy);
        const int irz = GridPosForSvF(Rz, dz);
        GridRange subgrid_actual = subgrid_org0;
        subgrid_actual.begin_x += irx;
        subgrid_actual.end_x += irx;
        subgrid_actual.begin_y += iry;
        subgrid_actual.end_y += iry;
        subgrid_actual.begin_z += irz;
        subgrid_actual.end_z += irz;
        
        std::vector<GridRange> range_blocks;
        std::vector<ShiftedGrid> shifted_grids;
        const int total_block_size = GetOverlapPeriodic(global_grid, subgrid_actual, global_x, global_y, global_z, range_blocks, shifted_grids);

        ForBlocks_in_global(range_blocks, shifted_grids, [&](int ix, int iy, int iz, int igx, int igy, int igz) {
            const double delta_x = (double)igx * dx - Rx;
            const double delta_y = (double)igy * dy - Ry;
            const double delta_z = (double)igz * dz - Rz;

            const double r2 = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
            const int64_t i = ix + global_x * (iy + (int64_t)global_y * iz);
            filter(i, r2);

            });


    }//ni//

}

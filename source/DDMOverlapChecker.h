#pragma once
#include <vector>
#include "GridRange.h"
//#include "GridSubgrid.h"

/*
* グリッドサイズがgrid_size_xの周期境界系において、split_xプロセスで分割されていたとする
* このときに、[subgrid_x_begin,subgrid_x_end) で指定されたSubgridが跨いでいるプロセス番号と、
* そのオーバーラップする各グリッドごとのグリッド領域を求めるクラス
* 
* 
* 
*/
class DDMOverlapChecker {
public:
    struct Info {
        int proc_id;
        int range_begin;
        int range_end;
        int periodic_num_laps;  //周期境界を何度跨いでいるか//
    };
    
    static int GetPeriodicNumLaps(int x, int N) {
        return (x < 0) ? -((-x + N - 1) / N) : x / N;
    }

    //プロセスごとの開始点//
    static int BeginPoint(int proc_id, int N, int num_procs) {
        return (N * proc_id) / num_procs;
    }

    static int GetProcID_no_periodic(int x, int N, int num_procs) {
        for (int p = num_procs-1; p >=0; --p) {
            if (BeginPoint(p, N, num_procs) <= x) {
                return p;  //担当プロセスを返す//
            }
        }
        return -1;//エラー//
    }

    /*
    * 1次元の領域分割に対して、サブグリッドのまだぐプロセスの範囲を返す。周期境界を跨いでいてもよい
    * 
    */
    static 
    std::vector<Info> GetOverlapProcList(int subgrid_x_begin, int subgrid_x_end, int grid_size_x, int num_procs) {
        const int num_laps_begin = GetPeriodicNumLaps(subgrid_x_begin, grid_size_x);
        const int num_laps_end = GetPeriodicNumLaps(subgrid_x_end-1, grid_size_x);

        int x0 = subgrid_x_begin - num_laps_begin * grid_size_x;
        int x1 = (subgrid_x_end-1) - num_laps_end * grid_size_x;

        int pid0 = GetProcID_no_periodic(x0, grid_size_x, num_procs);
        int pid1 = GetProcID_no_periodic(x1, grid_size_x, num_procs);

        std::vector<Info> list;

        if (num_laps_begin == num_laps_end) {
            if (pid0 == pid1) {
                list.emplace_back(Info{ pid0, x0, x1 + 1,num_laps_begin });
            } else {
                int next_x = x0;
                for (int p = pid0; p < pid1; ++p) {
                    int x_back = BeginPoint(p + 1, grid_size_x, num_procs);
                    list.emplace_back(Info{ p, next_x, x_back,num_laps_begin });
                    next_x = x_back;
                }
                list.emplace_back(Info{ pid1, next_x, x1 + 1,num_laps_end });
            }
        } else {

            {
                int lap = num_laps_end;
                int next_x = 0;
                for (int p = 0; p < pid1; ++p) {
                    int x_back = BeginPoint(p + 1, grid_size_x, num_procs);
                    list.emplace_back(Info{ p, next_x, x_back,lap });
                    next_x = x_back;
                }
                list.emplace_back(Info{ pid1, next_x, x1 + 1,num_laps_end });
            }
            for (int lap = num_laps_end - 1; lap > num_laps_begin; --lap) {
                int next_x = 0;
                for (int p = 0; p < num_procs; ++p) {
                    int x_back = BeginPoint(p + 1, grid_size_x, num_procs);
                    list.emplace_back(Info{ p, next_x, x_back,lap });
                    next_x = x_back;
                }
            }
            {
                int lap = num_laps_begin;
                int next_x = x0;
                for (int p = pid0; p < num_procs; ++p) {
                    int x_back = BeginPoint(p + 1, grid_size_x, num_procs);
                    list.emplace_back(Info{ p, next_x, x_back,lap });
                    next_x = x_back;
                }
            }
        }
        return list;
    }

    struct Info3D {
        int proc_id;
        GridRange range;            //周期境界範囲内に折りたたまれた座標値//
        ShiftedGrid shifted_grid;   //これをrangeから引くことで、折りたたまれていない座標になる//
        //int periodic_num_laps_x;  //周期境界を何度跨いでいるか//
        //int periodic_num_laps_y;  //周期境界を何度跨いでいるか//
        //int periodic_num_laps_z;  //周期境界を何度跨いでいるか//
    };

    /*
    global空間中の指定したsubgridが、領域分割で各プロセスが担当している分割領域とオーバーラップしているかどうかを判定
    オーバーラップしているプロセスとそのグリッド範囲の一覧をリストとして返す
    */
    static
    std::vector<Info3D> GetOverlapProc3D(const GridRange& subgrid, const GridRange& global_grid, int num_procs_x, int num_procs_y, int num_procs_z) {
        const int global_size_x = global_grid.SizeX();
        const int global_size_y = global_grid.SizeY();
        const int global_size_z = global_grid.SizeZ();
        auto list_x = GetOverlapProcList(subgrid.begin_x, subgrid.end_x, global_size_x, num_procs_x);
        auto list_y = GetOverlapProcList(subgrid.begin_y, subgrid.end_y, global_size_y, num_procs_y);
        auto list_z = GetOverlapProcList(subgrid.begin_z, subgrid.end_z, global_size_z, num_procs_z);

        std::vector< Info3D> list3d;
        for (const auto& lz : list_z) {
            for (const auto& ly : list_y) {
                for (const auto& lx : list_x) {
                    list3d.emplace_back(Info3D{ lx.proc_id + num_procs_x * (ly.proc_id + num_procs_y * lz.proc_id),
                        GridRange{lx.range_begin, ly.range_begin, lz.range_begin, lx.range_end, ly.range_end, lz.range_end},
                        ShiftedGrid {-global_size_x * lx.periodic_num_laps,
                        -global_size_y * ly.periodic_num_laps,
                        -global_size_z * lz.periodic_num_laps } });
                }
            }
        }
        return list3d;
    }


    static
    void Sort(std::vector<Info3D>& list) {        
        std::stable_sort(list.begin(), list.end(), [](const Info3D& a, const Info3D& b) {
            return a.proc_id < b.proc_id;
            });
    }


};


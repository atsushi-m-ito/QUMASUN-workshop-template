#pragma once
#ifdef USE_MPI

#pragma once
#include <mpi.h>
#include <vector>
#include "gyield/gy_memory.h"
#include "mpi_helper.h"
#include "GridRange.h"
#include "GridSubgrid.h"
#include "StopWatch.h"


/*
* Number of processes is P. 
* When N sets wave functions are divided into P processes by DDM parallelization,
* this class exchange the data of wave functions.
* By ExchangeGather() function, 
* each process has as N/P wave functions which is sequential data. 
* That is, ExchangeGather() converts from the DDM parallelization to state parallelization.
* 
* And, ExchangeScatter converts from state parallelization to DDM parallelization.
*/

class DDMGatherScatter {
private:
    StopWatch<5, true> watch;

    BlockCutInfo2* m_block_info = nullptr;

    
    int* m_send_recv_sizes = nullptr;

public:


    ~DDMGatherScatter() {
        if (m_block_info) {
            gy::AlignedFree(m_block_info);
            
            delete[] m_send_recv_sizes;
        }
//#define DEBUG_PRINT_TIME
#ifdef DEBUG_PRINT_TIME
        if (IsRoot(MPI_COMM_WORLD)) {
            watch.Print("DDMGatherScatter.Estimate :", 0);
            watch.Print("DDMGatherScatter.AlltoAllv:", 1);
            watch.Print("DDMGatherScatter.PasteSubg:", 2);
            watch.Print("DDMGatherScatter.CutSubgri:", 3);
            watch.Print("DDMGatherScatter.AlltoAllv:", 4);

        }
#endif

    }
    
    

    /*
    * プロセスごとに担当する状態数を返す
    */
    static
    int GetNumDistributed(int num_solution, MPI_Comm& ddm_comm) {
        const int num_procs = GetNumProcess(ddm_comm);
        const int proc_id = GetProcessID(ddm_comm);
        const int one_size = (num_solution + num_procs - 1) / num_procs;
        int remainder = num_solution - one_size * proc_id;
        if (remainder <= 0) return 0;
        if (remainder < one_size) return remainder;
        return one_size;        
    }


    static 
    size_t GetWorkSize(int global_grid_size, int num_solution, MPI_Comm& ddm_comm) {
        const int num_procs = GetNumProcess(ddm_comm);
        const int one_size = (num_solution + num_procs - 1) / num_procs;
        return global_grid_size * one_size * 4;
    }



    /*
    src_bundleはddm分割されているが、バッファとしては連続していることが前提
    戻り値として受信した波動関数の数を返す//
    */
    struct ExhangeRange {
        int offset;
        int num_bundle;
    };

    ExhangeRange EstimateRecieveRegion(const GridRange& global_grid, const GridRangeMPI& l_grid, int num_solution, int* send_heads) {

        const int num_procs = GetNumProcess(l_grid.mpi_comm);
        const int proc_id = GetProcessID(l_grid.mpi_comm);

        //auto send_heads = std::make_unique<int[]>(num_procs * 4);
        int* send_sizes = &send_heads[num_procs];
        int* recv_heads = &send_heads[num_procs * 2];
        int* recv_sizes = &send_heads[num_procs * 3];

        const int local_size = l_grid.Size3D() * 2; //2 means complex
        const int one_size = (num_solution + num_procs - 1) / num_procs;

        //send size////////////////////////
        {
            int num = 0;
            for (int i = 0; i < num_procs; ++i) {
                send_heads[i] = num * local_size;
                if (num_solution >= num + one_size) {
                    send_sizes[i] = one_size * local_size;
                    num += one_size;
                } else {
                    send_sizes[i] = (num_solution - num) * local_size;
                    num += num_solution - num;
                }
            }
        }


        //recv size///////////////////////////
        int num_recv_bundle = send_sizes[proc_id] / local_size; //zeroもあり得る//


        const int Nx = global_grid.SizeX();
        const int Ny = global_grid.SizeY();
        const int Nz = global_grid.SizeZ();
        const size_t global_size = (int64_t)Nx * (int64_t)Ny * (int64_t)Nz;

        const int num_divided = l_grid.num_split_x * l_grid.num_split_y * l_grid.num_split_z;
        if (m_block_info == nullptr) {
            m_block_info = gy::AlignedAlloc<BlockCutInfo2>(num_divided);            
        }

        int head = 0;        
        for (int pz = 0; pz < l_grid.num_split_z; ++pz) {
            for (int py = 0; py < l_grid.num_split_y; ++py) {
                for (int px = 0; px < l_grid.num_split_x; ++px) {

                    int pid = px + l_grid.num_split_x * (py + l_grid.num_split_y * pz);

                    GridRange grid_p;
                    grid_p.begin_z = ((Nz * pz) / l_grid.num_split_z);
                    grid_p.end_z = ((Nz * (pz + 1)) / l_grid.num_split_z);
                    grid_p.begin_y = ((Ny * py) / l_grid.num_split_y);
                    grid_p.end_y = ((Ny * (py + 1)) / l_grid.num_split_y);
                    grid_p.begin_x = ((Nx * px) / l_grid.num_split_x);
                    grid_p.end_x = ((Nx * (px + 1)) / l_grid.num_split_x);

                    //AlltoAllvの通信サイズ用
                    recv_heads[pid] = head;
                    recv_sizes[pid] = grid_p.Size3D() * num_recv_bundle * 2;//2 means complex//
                    head += recv_sizes[pid];

                    //Block-copy用の部分領域情報
                    m_block_info[pid].ix_end = grid_p.SizeX();
                    m_block_info[pid].iy_end = grid_p.SizeY();
                    m_block_info[pid].iz_end = grid_p.SizeZ();
                    m_block_info[pid].ix_offset = grid_p.begin_x - global_grid.begin_x;
                    m_block_info[pid].iy_offset = grid_p.begin_y - global_grid.begin_y;
                    m_block_info[pid].iz_offset = grid_p.begin_z - global_grid.begin_z;
                    m_block_info[pid].small_offset = recv_heads[pid];                    

                }
            }
        }

        return ExhangeRange{ send_heads[proc_id] / local_size,num_recv_bundle };
    }

    /*
    * 領域分割された各プロセスから集めた、subgridの塊(しかもbubdle)であるgather_bufを、
    * globalなグリッドに焼き直す
    */
#ifdef GY_WITH_CUDA_OR_HIP
    void PasteSubgridToGlobal_bundle(const GridRange& global_grid, const GridRangeMPI& l_grid, int num_recv_bundle, double* dest_Z_bundle, const double* gather_buf)
    {
        const size_t global_size = global_grid.Size3D();

        const int num_divided = l_grid.num_split_x * l_grid.num_split_y * l_grid.num_split_z;


        dim3 cu_threads;
        cu_threads.x = GY_MAX_THREADS_PER_BLOCK;
        cu_threads.y = 1;
        cu_threads.z = 1;
        dim3 cu_blocks;
        cu_blocks.x = num_recv_bundle;
        cu_blocks.y = num_divided;
        cu_blocks.z = 1;

        gy::kernel_PasteSubgridD2Z_2 <<<cu_blocks, cu_threads>>> (m_block_info, (double2*)&dest_Z_bundle[0], global_grid.SizeX(), global_grid.SizeX() * global_grid.SizeY(), global_size, gather_buf);
        gy::Synchronize();
    }

#else

    void PasteSubgridToGlobal_bundle(const GridRange& global_grid, const GridRangeMPI& l_grid, int num_recv_bundle, double* dest_Z_bundle, const double* gather_buf)
    {
        

        const size_t global_size = global_grid.Size3D();
        const int num_divided = l_grid.num_split_x * l_grid.num_split_y * l_grid.num_split_z;

        for (int n = 0; n < num_recv_bundle; ++n) {
            for (int ib = 0; ib < num_divided; ++ib) {
                const int grid_p_size = m_block_info[ib].ix_end * m_block_info[ib].iy_end * m_block_info[ib].iz_end;
                PasteSubgridD2Z(m_block_info[ib], dest_Z_bundle + global_size * n * 2,  global_grid.SizeX(), global_grid.SizeX() * global_grid.SizeY(), gather_buf + (2 * n) * grid_p_size, gather_buf + (2 * n + 1) * grid_p_size);
            }
        }
        
    }

#endif


    ExhangeRange Estimate(const GridRange& global_grid, const GridRangeMPI& l_grid, int num_solution) {
        const int num_procs = GetNumProcess(l_grid.mpi_comm);
        if (m_send_recv_sizes == nullptr) {
            m_send_recv_sizes = new int[num_procs * 4];
        }
        ExhangeRange range = EstimateRecieveRegion(global_grid, l_grid, num_solution, &m_send_recv_sizes[0]);
        return range;
    }

    /*
    * SoAComplexではなく、
    * std::complex<double>相当の配列として出力するモード.
    */
    void GatherExchangeD2Z(const ExhangeRange& range, const GridRange& global_grid, double* dest_Z_bundle, const GridRangeMPI& l_grid, const double* src_bundle, int num_solution, double* work) {
        watch.Restart();
        if (m_send_recv_sizes == nullptr) {
            return ;
        }

        const int num_procs = GetNumProcess(l_grid.mpi_comm);
        const int proc_id = GetProcessID(l_grid.mpi_comm);

        
        int* send_heads = m_send_recv_sizes; 
        int* send_sizes = &send_heads[num_procs];
        int* recv_heads = &send_heads[num_procs * 2];
        int* recv_sizes = &send_heads[num_procs * 3];

        

        //このプロセスが担当するbundle数が返る//
        //ExhangeRange range = EstimateRecieveRegion(global_grid, l_grid, num_solution, &send_heads[0]);


        const int num_recv_bundle = range.num_bundle;
        size_t global_size = global_grid.Size3D();
        size_t recv_offset = global_size * 2 * num_recv_bundle;
        auto* gather_buf = work;

        watch.Record(0);

        MPI_Alltoallv(src_bundle, &send_sizes[0], &send_heads[0], MPI_DOUBLE,
            gather_buf, &recv_sizes[0], &recv_heads[0], MPI_DOUBLE, l_grid.mpi_comm);

        watch.Record(1);
        

        PasteSubgridToGlobal_bundle(global_grid, l_grid, num_recv_bundle, dest_Z_bundle, gather_buf);

        watch.Record(2);

    }




#ifdef GY_WITH_CUDA_OR_HIP
    void CutSubgridFromGlobal_bundle(const GridRangeMPI& l_grid, double* dest_buf, const GridRange& global_grid, double* src_Z_bundle, int num_hold)
    {
        const size_t global_size = global_grid.Size3D();

        const int num_divided = l_grid.num_split_x * l_grid.num_split_y * l_grid.num_split_z;


        dim3 cu_threads;
        cu_threads.x = GY_MAX_THREADS_PER_BLOCK;
        cu_threads.y = 1;
        cu_threads.z = 1;
        dim3 cu_blocks;
        cu_blocks.x = num_hold;
        cu_blocks.y = num_divided;
        cu_blocks.z = 1;

        gy::kernel_CutSubgridZ2D_2 <<<cu_blocks, cu_threads >>> (m_block_info, dest_buf, (double2*)&src_Z_bundle[0], global_grid.SizeX(), global_grid.SizeX() * global_grid.SizeY(), global_size);
        gy::Synchronize();
    }

#else


    void CutSubgridFromGlobal_bundle(const GridRangeMPI& l_grid, double* dest_buf, const GridRange& global_grid, double* src_Z_bundle, int num_hold)
    {
        const size_t global_size = global_grid.Size3D();
        const int num_divided = l_grid.num_split_x * l_grid.num_split_y * l_grid.num_split_z;

        for (int pid = 0; pid < num_divided; ++pid) {
            const size_t grid_p_size = m_block_info[pid].ix_end * m_block_info[pid].iy_end * m_block_info[pid].iz_end;
            for (int b = 0; b < num_hold; ++b) {        
                CutSubgridZ2D(m_block_info[pid], dest_buf + 2 * b * grid_p_size, dest_buf + (2 * b + 1) * grid_p_size, src_Z_bundle + global_size * b * 2, global_grid.SizeX(), global_grid.SizeX() * global_grid.SizeY());
            }
        }
    }
#endif

    /*
    src_bundleはddm分割されているが、バッファとしては連続していることが前提
    戻り値として受信した波動関数の数を返す//
    入力配列がSoAComplexではなく、std::complex<double>を想定しているモード
    */
    void ScatterExchangeZ2D(const ExhangeRange& range, const GridRangeMPI& l_grid, double* dest_bundle, const GridRange& global_grid, double* src_Z_bundle, int total_bundle, double* work) {
        watch.Restart();
        const int num_procs = GetNumProcess(l_grid.mpi_comm);
        const int proc_id = GetProcessID(l_grid.mpi_comm);

        //printf("[%d]TEST:ScatterExchange: num_hold = %d\n", GetProcessID(l_grid.mpi_comm), num_hold); fflush(stdout);
        //MPI_Barrier(l_grid.mpi_comm);
        const int num_hold = range.num_bundle;

        const int* send_heads = &m_send_recv_sizes[num_procs * 2];
        const int* send_sizes = &m_send_recv_sizes[num_procs * 3];
        const int* recv_heads = m_send_recv_sizes;
        const int* recv_sizes = &m_send_recv_sizes[num_procs * 1];

        //send info///////////////////////////////////////
        const size_t global_size = global_grid.Size3D();
        const size_t send_offset = global_size * 2 * num_hold;
        auto* gather_buf = work;

        CutSubgridFromGlobal_bundle(l_grid, gather_buf, global_grid, src_Z_bundle, num_hold); 

        watch.Record(3);

        MPI_Alltoallv(gather_buf, &send_sizes[0], &send_heads[0], MPI_DOUBLE,
            dest_bundle, &recv_sizes[0], &recv_heads[0], MPI_DOUBLE, l_grid.mpi_comm);

        watch.Record(4);
        
    }

    template< int NUM_SECTOR, bool IS_MEASURE>
    void MergeTimer(int index, StopWatch<NUM_SECTOR, IS_MEASURE>& dest){
        dest.Merge(index, watch);
    }
};


#endif


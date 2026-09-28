#pragma once
#include <mpi.h>
#include <vector>

//#define DEBUG_DDMEXCHANGE_1


namespace DDM {
    inline
    int GridBegin(int Nx, int pid_x, int num_procs_x) {
        return (Nx * pid_x) / num_procs_x;
    }
};

/*
* 3次元領域分割されたグリッドデータをFFTするためのヘルパークラス
* ExchangeX()によってDDM並列されたデータを、X方向にそれぞれ連続な並びのデータにする。
* ただし、X方向に連続としたとき、Y,Z方向のグリッドデータ(line)は各プロセスが分担して保持する。
* つまり、X方向のDDM並列から、X方向には連続だがY-Z方向に分割されたデータとする
* 
* ExchangeY,Zを呼ぶためには、メモリがY,Z方向に連続である必要がある。
* そのために、事前に転置Transpose3()関数を呼ぶこと
* 
* Alltoallよ呼ぶ実装と、Alltoallvを呼ぶ実装が考えられる
* AlltoallvはCUDAでmanaged memoryをバッファに指定した時にバグが出た
*/
#define DDMEXCHANGE_ALLTOALL_V

class DDMExchange{
private:
    MPI_Comm mpi_comm;
    MPI_Comm ddm_comm_x, ddm_comm_y, ddm_comm_z;
    
    std::vector<int> send_heads;
    
public:
    DDMExchange(MPI_Comm& mpi_comm_, int split_x_, int split_y_, int split_z_) :
        mpi_comm(mpi_comm_) 
    {

        int proc_id = GetProcessID(mpi_comm);
        int num_procs = GetNumProcess(mpi_comm);
        int num_procs_x = split_x_;
        int num_procs_y = split_y_;
        int num_procs_z = split_z_;
        
        int pid_x = proc_id % num_procs_x;
        int pid_y = (proc_id / num_procs_x) % num_procs_y;
        int pid_z = proc_id / (num_procs_x * num_procs_y);

        //make ddm comm for each x, y, z direction.//////////////////////////////////////        
        MPI_Comm_split(mpi_comm, pid_y + num_procs_y * pid_z, pid_x, &ddm_comm_x);
        MPI_Comm_split(mpi_comm, pid_x + num_procs_x * pid_z, pid_y, &ddm_comm_y);
        MPI_Comm_split(mpi_comm, pid_x + num_procs_x * pid_y, pid_z, &ddm_comm_z);


        int max_num_proc = std::max(std::max(num_procs_x, num_procs_y), num_procs_z);
        send_heads.resize(max_num_proc * 4);
    }

    ~DDMExchange() {
        MPI_Comm_free(&ddm_comm_x);
        MPI_Comm_free(&ddm_comm_y);
        MPI_Comm_free(&ddm_comm_z);
    }


    
    GridRange GetLocalRange(int Nx, int Ny, int Nz) {
        const int pid_x = GetProcessID(ddm_comm_x);
        const int num_procs_x = GetNumProcess(ddm_comm_x);

        const int local_begin_x = DDM::GridBegin(Nx, pid_x, num_procs_x);
        const int local_end_x = DDM::GridBegin(Nx, pid_x + 1, num_procs_x);

        const int pid_y = GetProcessID(ddm_comm_y);
        const int num_procs_y = GetNumProcess(ddm_comm_y);

        const int local_begin_y = DDM::GridBegin(Ny, pid_y, num_procs_y);
        const int local_end_y = DDM::GridBegin(Ny, pid_y + 1, num_procs_y);
        
        const int pid_z = GetProcessID(ddm_comm_z);
        const int num_procs_z = GetNumProcess(ddm_comm_z);

        const int local_begin_z = DDM::GridBegin(Nz, pid_z, num_procs_z);
        const int local_end_z = DDM::GridBegin(Nz, pid_z + 1, num_procs_z);
        
        return GridRange{ local_begin_x , local_begin_y, local_begin_z, local_end_x, local_end_y, local_end_z };
    }

    
    int64_t GetExcangeBufferSize(int Nx, int Ny, int Nz) {
#if 1
        auto range = GetLocalRange(Nx, Ny, Nz);
        const int local_size_x = range.end_x - range.begin_x;
        const int local_size_y = range.end_y - range.begin_y;
        const int local_size_z = range.end_z - range.begin_z;

        const int num_procs_x = GetNumProcess(ddm_comm_x);
        const int num_procs_y = GetNumProcess(ddm_comm_y);
        const int num_procs_z = GetNumProcess(ddm_comm_z);
#else
        const int pid_x = GetProcessID(ddm_comm_xyz);
        const int num_procs_x = GetNumProcess(ddm_comm_xyz);

        const int local_begin_x = DDM::GridBegin(Nx, pid_x, num_procs_x);
        const int local_end_x = DDM::GridBegin(Nx, pid_x + 1, num_procs_x);
        const int local_size_x = local_end_x - local_begin_x;

        const int pid_y = GetProcessID(ddm_comm_y);
        const int num_procs_y = GetNumProcess(ddm_comm_y);

        const int local_begin_y = DDM::GridBegin(Ny, pid_y, num_procs_y);
        const int local_end_y = DDM::GridBegin(Ny, pid_y + 1, num_procs_y);
        const int local_size_y = local_end_y - local_begin_y;

        const int pid_z = GetProcessID(ddm_comm_z);
        const int num_procs_z = GetNumProcess(ddm_comm_z);

        const int local_begin_z = DDM::GridBegin(Nz, pid_z, num_procs_z);
        const int local_end_z = DDM::GridBegin(Nz, pid_z + 1, num_procs_z);
        const int local_size_z = local_end_z - local_begin_z;
#endif

#ifndef DDMEXCHANGE_ALLTOALL_V

        const int64_t max_lines_yz = (local_size_y * local_size_z + num_procs_x - 1) / num_procs_x;
        const int64_t max_ddm_width_x = (Nx + num_procs_x - 1) / num_procs_x;
        int64_t size_local_kspace = max_lines_yz * max_ddm_width_x * num_procs_x;

        const int64_t max_lines_zx = (local_size_z * local_size_x + num_procs_y - 1) / num_procs_y;
        const int64_t max_ddm_width_y = (Ny + num_procs_y - 1) / num_procs_y;
        size_local_kspace = std::max(size_local_kspace, max_lines_zx * max_ddm_width_y * num_procs_y);

        const int64_t max_lines_xy = (local_size_x * local_size_y + num_procs_z - 1) / num_procs_z;
        const int64_t max_ddm_width_z = (Nz + num_procs_z - 1) / num_procs_z;
        size_local_kspace = std::max(size_local_kspace, max_lines_xy * max_ddm_width_z * num_procs_z);
        return size_local_kspace;
#else
        int64_t size_local_kspace = (int64_t)Nx * ((local_size_y * local_size_z + num_procs_x - 1) / num_procs_x);
        size_local_kspace = std::max(size_local_kspace, (int64_t)Ny * ((local_size_x * local_size_z + num_procs_y - 1) / num_procs_y));
        size_local_kspace = std::max(size_local_kspace, (int64_t)Nz * ((local_size_x * local_size_y + num_procs_z - 1) / num_procs_z));
        return size_local_kspace;
#endif
    }

    /* x exchange////////////////////////////////////////////////////////
    // x方向に領域分割されたバッファをExchangeして、x方向に連続なバッファに並べ替える//
    // 並べ替えた後に
    // 
    // local_size_yz = local_size_y * local_size_z 
    */

    int GetNumLinesExchangedAny(int total_lines, MPI_Comm& ddm_comm_xyz) {
        const int pid_x = GetProcessID(ddm_comm_xyz);
        const int num_procs_x = GetNumProcess(ddm_comm_xyz);
        int my_lines = DDM::GridBegin(total_lines, (pid_x + 1), num_procs_x) - DDM::GridBegin(total_lines, pid_x, num_procs_x);
        return my_lines;
    }
    int GetNumLinesExchangedX(int total_lines) {
        return GetNumLinesExchangedAny(total_lines, ddm_comm_x);
    }
    int GetNumLinesExchangedY(int total_lines) {
        return GetNumLinesExchangedAny(total_lines, ddm_comm_y);
    }
    int GetNumLinesExchangedZ(int total_lines) {
        return GetNumLinesExchangedAny(total_lines, ddm_comm_z);
    }


#ifndef DDMEXCHANGE_ALLTOALL_V
    int ForwardExchangeAny(int Nx, int total_lines, const double* ddm_local_buffer, double* continuous_buffer, double* work_buffer, MPI_Comm& ddm_comm_xyz, int size_of_double)
    {
        const int pid_x = GetProcessID(ddm_comm_xyz);
        const int num_procs_x = GetNumProcess(ddm_comm_xyz);


        const int local_begin_x = DDM::GridBegin(Nx, pid_x, num_procs_x);
        const int local_end_x = DDM::GridBegin(Nx, pid_x + 1, num_procs_x);
        const int local_size_x = local_end_x - local_begin_x;

        //通信サイズの計算//
        //send_heads.resize(num_procs_x * 4);
        int* send_sizes = &send_heads[num_procs_x];
        int* recv_heads = &send_heads[num_procs_x * 2];
        int* recv_sizes = &send_heads[num_procs_x * 3];
        
        //int my_lines = DDM::GridBegin(total_lines, (pid_x + 1), num_procs_x) - DDM::GridBegin(total_lines, pid_x, num_procs_x);
        const int my_lines = GetNumLinesExchangedAny(total_lines, ddm_comm_xyz);
        const int max_lines = (total_lines + num_procs_x - 1) / num_procs_x;
        const int max_ddm_width = (Nx + num_procs_x - 1) / num_procs_x;
        const int block_size = max_lines * max_ddm_width * size_of_double;
        {
            int r_heads = 0;
            int s_heads = 0;
            for (int p = 0; p < num_procs_x; ++p) {
                recv_heads[p] = r_heads;
                r_heads = ((Nx * (p + 1)) / num_procs_x) * my_lines * size_of_double;
                recv_sizes[p] = r_heads - recv_heads[p];

                send_heads[p] = s_heads;
                //s_heads = ((total_lines * (p + 1)) / num_procs_x) * local_size_x;
                s_heads = DDM::GridBegin(total_lines, (p + 1), num_procs_x) * local_size_x * size_of_double;
                send_sizes[p] = s_heads - send_heads[p];
                
                gy::CopyMemory(&work_buffer[block_size * p], &ddm_local_buffer[send_heads[p]], send_sizes[p] * sizeof(double));
            }


        }

        int total_send = 0;
        int total_recv = 0;
        for(int p = 0; p < num_procs_x;++p){
            total_send += send_sizes[p];
            total_recv += recv_sizes[p];
        }
        double sum2 = 0;
        for (int64_t i = 0; i < total_send; ++i) {
            sum2 += ddm_local_buffer[i];
        }
#if 1
        //順方向Exchange//
        MPI_Alltoall(MPI_IN_PLACE, block_size, MPI_DOUBLE,
            &work_buffer[0], block_size, MPI_DOUBLE, ddm_comm_xyz);
#elif 1
        auto sub_send_buf = std::make_unique<double[]>(total_send);
        auto sub_recv_buf = std::make_unique<double[]>(total_recv);
        gy::CopyMemoryToHost(sub_send_buf.get(), ddm_local_buffer, sizeof(double)* total_send);
        gyCheckError(gyGetLastError(), "DEV-ERROR187");
        gy::Synchronize();
        //順方向Exchange//
        MPI_Alltoallv(&sub_send_buf[0], &send_sizes[0], &send_heads[0], MPI_DOUBLE,
            &sub_recv_buf[0], &recv_sizes[0], &recv_heads[0], MPI_DOUBLE, ddm_comm_xyz);
        
#ifdef GY_WITH_CUDA_OR_HIP
        gyMemcpy(&work_buffer[0], &sub_recv_buf[0], sizeof(double) * total_recv, gyMemcpyHostToDevice);
#else
        std::memcpy(&work_buffer[0], &sub_recv_buf[0], sizeof(double) * total_recv);
#endif

#else
        //順方向Exchange//
        MPI_Alltoallv(&ddm_local_buffer[0], &send_sizes[0], &send_heads[0], MPI_DOUBLE,
            &work_buffer[0], &recv_sizes[0], &recv_heads[0], MPI_DOUBLE, ddm_comm_xyz);
#endif
        gy::Synchronize();
        gyCheckError(gyGetLastError(), "DEV-ERROR193");

        double sum = 0.0;
        int64_t icount = 0;

        //受信データを1列に並べ替え
        for (int l = 0; l < my_lines; ++l) {
            int64_t offset = l * Nx * size_of_double;
            for (int p = 0; p < num_procs_x; ++p) {
                const int64_t width = recv_sizes[p] / my_lines;
                //int64_t r_head = recv_heads[p] + width * l;
                const int64_t r_head = block_size * p + width * l;

                
                for (int64_t i = 0; i < width; ++i) {
                    continuous_buffer[(offset + i)] = work_buffer[r_head + i];

                    sum += work_buffer[r_head + i];

                }
                offset += width;
                icount += width;
            }

            //printf("DEBUG(%d)FEA: %d, %f\n", GetProcessID(MPI_COMM_WORLD), l, sum);
        }


        printf("DEBUG(%d)ForwardExchangeAny:%f, %f, %zd, %d\n", GetProcessID(MPI_COMM_WORLD), sum, sum2, icount, send_sizes[0] + send_sizes[1]);
        if (num_procs_x > 1) {
            printf("DEBUG(%d)ForwardExchangeAny-2: %d, %d, %d, %d\n", GetProcessID(MPI_COMM_WORLD), send_heads[0], send_heads[1], recv_heads[0], recv_heads[1]);
        }
        return my_lines;
    }

    void BackwardExchangeAny(int Nx, int total_lines, double* ddm_local_buffer, const double* continuous_buffer, double* work_buffer, MPI_Comm& ddm_comm_xyz, int size_of_double)
    {
     
        const int pid_x = GetProcessID(ddm_comm_xyz);
        const int num_procs_x = GetNumProcess(ddm_comm_xyz);


        const int local_begin_x = DDM::GridBegin(Nx, pid_x, num_procs_x);
        const int local_end_x = DDM::GridBegin(Nx, pid_x + 1, num_procs_x);
        const int local_size_x = local_end_x - local_begin_x;

        //通信サイズの計算//
        //send_heads.resize(num_procs_x * 4);
        int* send_sizes = &send_heads[num_procs_x];
        int* recv_heads = &send_heads[num_procs_x * 2];
        int* recv_sizes = &send_heads[num_procs_x * 3];
        //int my_lines = DDM::GridBegin(total_lines, (pid_x + 1), num_procs_x) - DDM::GridBegin(total_lines, pid_x, num_procs_x);
        int my_lines = GetNumLinesExchangedAny(total_lines, ddm_comm_xyz);
        const int max_lines = (total_lines + num_procs_x - 1) / num_procs_x;
        const int max_ddm_width = (Nx + num_procs_x - 1) / num_procs_x;
        const int block_size = max_lines * max_ddm_width * size_of_double;


        double sum = 0.0;

        //1列に並んだデータを逆方向通信のデータに並べ替え
        for (int l = 0; l < my_lines; ++l) {
            int64_t offset = l * Nx * size_of_double;
            for (int p = 0; p < num_procs_x; ++p) {
                int64_t width = recv_sizes[p] / my_lines;
                int64_t r_head = block_size * p + width * l;

                
                for (int64_t i = 0; i < width; ++i) {
                    work_buffer[(r_head + i)] = continuous_buffer[(offset + i)];

                    sum += work_buffer[r_head + i];
                }
                offset += width;
            }
        }

        //逆方向Exchange//
        MPI_Alltoall(MPI_IN_PLACE, block_size, MPI_DOUBLE,
            &work_buffer[0], block_size, MPI_DOUBLE, ddm_comm_xyz);


        int r_heads = 0;
        int s_heads = 0;
        for (int p = 0; p < num_procs_x; ++p) {
            /*
            recv_heads[p] = r_heads;
            r_heads = ((Nx * (p + 1)) / num_procs_x) * my_lines * size_of_double;
            recv_sizes[p] = r_heads - recv_heads[p];

            send_heads[p] = s_heads;
            //s_heads = ((total_lines * (p + 1)) / num_procs_x) * local_size_x;
            s_heads = DDM::GridBegin(total_lines, (p + 1), num_procs_x) * local_size_x * size_of_double;
            send_sizes[p] = s_heads - send_heads[p];
            */
            gy::CopyMemory(&ddm_local_buffer[send_heads[p]], &work_buffer[block_size * p], send_sizes[p] * sizeof(double));
        }


        printf("DEBUG(%d)BackwardExchangeAny:%f\n", GetProcessID(MPI_COMM_WORLD), sum);

    }
#else
//use Alltoallv


int ForwardExchangeAny(int Nx, int total_lines, const double* ddm_local_buffer, double* continuous_buffer, double* work_buffer, MPI_Comm& ddm_comm_xyz, int size_of_double)
{
    const int pid_x = GetProcessID(ddm_comm_xyz);
    const int num_procs_x = GetNumProcess(ddm_comm_xyz);


    const int local_begin_x = DDM::GridBegin(Nx, pid_x, num_procs_x);
    const int local_end_x = DDM::GridBegin(Nx, pid_x + 1, num_procs_x);
    const int local_size_x = local_end_x - local_begin_x;

    //通信サイズの計算//
    //send_heads.resize(num_procs_x * 4);
    int* send_sizes = &send_heads[num_procs_x];
    int* recv_heads = &send_heads[num_procs_x * 2];
    int* recv_sizes = &send_heads[num_procs_x * 3];

    //int my_lines = DDM::GridBegin(total_lines, (pid_x + 1), num_procs_x) - DDM::GridBegin(total_lines, pid_x, num_procs_x);
    int my_lines = GetNumLinesExchangedAny(total_lines, ddm_comm_xyz);
    {
        int r_heads = 0;
        int s_heads = 0;
        for (int p = 0; p < num_procs_x; ++p) {
            recv_heads[p] = r_heads;
            r_heads = ((Nx * (p + 1)) / num_procs_x) * my_lines * size_of_double;
            recv_sizes[p] = r_heads - recv_heads[p];

            send_heads[p] = s_heads;
            //s_heads = ((total_lines * (p + 1)) / num_procs_x) * local_size_x;
            s_heads = DDM::GridBegin(total_lines, (p + 1), num_procs_x) * local_size_x * size_of_double;
            send_sizes[p] = s_heads - send_heads[p];
        }


    }

#ifdef DEBUG_DDMEXCHANGE_1
    int total_send = 0;
    int total_recv = 0;
    for (int p = 0; p < num_procs_x; ++p) {
        total_send += send_sizes[p];
        total_recv += recv_sizes[p];
    }
    gy::Synchronize();
    double sum2 = 0;
    for (int64_t i = 0; i < total_send; ++i) {
        sum2 += ddm_local_buffer[i];
    }
#endif

#if 0
    auto sub_send_buf = std::make_unique<double[]>(total_send);
    auto sub_recv_buf = std::make_unique<double[]>(total_recv);
    gy::CopyMemoryToHost(sub_send_buf.get(), ddm_local_buffer, sizeof(double) * total_send);
    gyCheckError(gyGetLastError(), "DEV-ERROR187");
    gy::Synchronize();
    //順方向Exchange//
    MPI_Alltoallv(&sub_send_buf[0], &send_sizes[0], &send_heads[0], MPI_DOUBLE,
        &sub_recv_buf[0], &recv_sizes[0], &recv_heads[0], MPI_DOUBLE, ddm_comm_xyz);

#ifdef GY_WITH_CUDA_OR_HIP
    gyMemcpy(&work_buffer[0], &sub_recv_buf[0], sizeof(double) * total_recv, gyMemcpyHostToDevice);
#else
    std::memcpy(&work_buffer[0], &sub_recv_buf[0], sizeof(double) * total_recv);
#endif

#else
    gy::Synchronize();
    //順方向Exchange//
    MPI_Alltoallv(&ddm_local_buffer[0], &send_sizes[0], &send_heads[0], MPI_DOUBLE,
        &work_buffer[0], &recv_sizes[0], &recv_heads[0], MPI_DOUBLE, ddm_comm_xyz);
#endif
    gy::Synchronize();
    //gyCheckError(gyGetLastError(), "DEV-ERROR193");

#ifdef DEBUG_DDMEXCHANGE_1
    double sum = 0.0;
    int64_t icount = 0;
#endif

    //受信データを1列に並べ替え
    for (int l = 0; l < my_lines; ++l) {
        int64_t offset = l * Nx * size_of_double;
        for (int p = 0; p < num_procs_x; ++p) {
            int64_t width = recv_sizes[p] / my_lines;
            int64_t r_head = recv_heads[p] + width * l;


            for (int64_t i = 0; i < width; ++i) {
                continuous_buffer[(offset + i)] = work_buffer[r_head + i];
#ifdef DEBUG_DDMEXCHANGE_1
                sum += work_buffer[r_head + i];
#endif
            }
            offset += width;
#ifdef DEBUG_DDMEXCHANGE_1
            icount += width;
#endif
        }

        //printf("DEBUG(%d)FEA: %d, %f\n", GetProcessID(MPI_COMM_WORLD), l, sum);
    }

#ifdef DEBUG_DDMEXCHANGE_1
    printf("DEBUG(%d)ForwardExchangeAny:%f, %f, %zd, %d\n", GetProcessID(MPI_COMM_WORLD), sum, sum2, icount, send_sizes[0] + send_sizes[1]);
    if (num_procs_x > 1) {
        printf("DEBUG(%d)ForwardExchangeAny-2: %d, %d, %d, %d\n", GetProcessID(MPI_COMM_WORLD), send_heads[0], send_heads[1], recv_heads[0], recv_heads[1]);
    }
#endif
    return my_lines;
}

void BackwardExchangeAny(int Nx, int total_lines, double* ddm_local_buffer, const double* continuous_buffer, double* work_buffer, MPI_Comm& ddm_comm_xyz, int size_of_double)
{

    const int pid_x = GetProcessID(ddm_comm_xyz);
    const int num_procs_x = GetNumProcess(ddm_comm_xyz);


    const int local_begin_x = DDM::GridBegin(Nx, pid_x, num_procs_x);
    const int local_end_x = DDM::GridBegin(Nx, pid_x + 1, num_procs_x);
    const int local_size_x = local_end_x - local_begin_x;

    //通信サイズの計算//
    //send_heads.resize(num_procs_x * 4);
    int* send_sizes = &send_heads[num_procs_x];
    int* recv_heads = &send_heads[num_procs_x * 2];
    int* recv_sizes = &send_heads[num_procs_x * 3];
    //int my_lines = DDM::GridBegin(total_lines, (pid_x + 1), num_procs_x) - DDM::GridBegin(total_lines, pid_x, num_procs_x);
    int my_lines = GetNumLinesExchangedAny(total_lines, ddm_comm_xyz);

    gy::Synchronize();

#ifdef DEBUG_DDMEXCHANGE_1
    double sum = 0.0;
#endif

    //1列に並んだデータを逆方向通信のデータに並べ替え
    for (int l = 0; l < my_lines; ++l) {
        int64_t offset = l * Nx * size_of_double;
        for (int p = 0; p < num_procs_x; ++p) {
            int64_t width = recv_sizes[p] / my_lines;
            int64_t r_head = recv_heads[p] + width * l;


            for (int64_t i = 0; i < width; ++i) {
                work_buffer[(r_head + i)] = continuous_buffer[(offset + i)];
#ifdef DEBUG_DDMEXCHANGE_1
                sum += work_buffer[r_head + i];
#endif
            }
            offset += width;
        }
    }

    gy::Synchronize();
    //逆方向Exchange//
    MPI_Alltoallv(&work_buffer[0], &recv_sizes[0], &recv_heads[0], MPI_DOUBLE,
        &ddm_local_buffer[0], &send_sizes[0], &send_heads[0], MPI_DOUBLE, ddm_comm_xyz);

    gy::Synchronize();

#ifdef DEBUG_DDMEXCHANGE_1
    double sum2 = 0;
    for (int64_t i = 0; i < send_sizes[0] + send_sizes[1]; ++i) {
        sum2 += ddm_local_buffer[i];
    }

    printf("DEBUG(%d)BackwardExchangeAny:%f, %f\n", GetProcessID(MPI_COMM_WORLD), sum, sum2);
#endif
}
#endif //DDMEXCHANGE_ALLTOALL_V

    //ForwardとBackwardでデータのサイズを変える//
    //Forwardでデータサイズwidthを通信した時、Backwardではデータのサイズwidth*HR_ratioを通信するようにする//
    //FFTを使ったupconvert等をするときに用いる//
    //Scale downはできない//
    void ScaleUp(int HR_ratio) {
        for (auto& s : send_heads) {
            s *= HR_ratio;
        }
    }

    //ForwardとBackwardでデータのサイズを変える//
    //FFTを使ったdown-convertに限り利用可能//
    void ScaleDown(int HR_ratio) {
        for (auto& s : send_heads) {
            s /= HR_ratio;
        }
    }

    int ForwardExchangeX(int Nx, int local_size_yz, const double* ddm_local_xyz_buffer, double* continuous_xyz_buffer, double* work_buffer, int size_of_double = 1) {
        return ForwardExchangeAny(Nx, local_size_yz, ddm_local_xyz_buffer, continuous_xyz_buffer, work_buffer, ddm_comm_x, size_of_double);
    }
    void BackwardExchangeX(int Nx, int local_size_yz, double* ddm_local_xyz_buffer, const double* continuous_xyz_buffer, double* work_buffer, int size_of_double = 1) {
        BackwardExchangeAny(Nx, local_size_yz, ddm_local_xyz_buffer, continuous_xyz_buffer, work_buffer, ddm_comm_x, size_of_double);
    }
    int ForwardExchangeY(int Ny, int local_size_xz, const double* ddm_local_yzx_buffer, double* continuous_yzx_buffer, double* work_buffer, int size_of_double = 1) {
        return ForwardExchangeAny(Ny, local_size_xz, ddm_local_yzx_buffer, continuous_yzx_buffer, work_buffer, ddm_comm_y, size_of_double);
    }
    void BackwardExchangeY(int Ny, int local_size_xz, double* ddm_local_yzx_buffer, const double* continuous_yzx_buffer, double* work_buffer, int size_of_double = 1) {
        BackwardExchangeAny(Ny, local_size_xz, ddm_local_yzx_buffer, continuous_yzx_buffer, work_buffer, ddm_comm_y, size_of_double);
    }
    int ForwardExchangeZ(int Nz, int local_size_xy, const double* ddm_local_zxy_buffer, double* continuous_zxy_buffer, double* work_buffer, int size_of_double = 1) {
        return ForwardExchangeAny(Nz, local_size_xy, ddm_local_zxy_buffer, continuous_zxy_buffer, work_buffer, ddm_comm_z, size_of_double);
    }
    void BackwardExchangeZ(int Nz, int local_size_xy, double* ddm_local_zxy_buffer, const double* continuous_zxy_buffer, double* work_buffer, int size_of_double = 1) {
        BackwardExchangeAny(Nz, local_size_xy, ddm_local_zxy_buffer, continuous_zxy_buffer, work_buffer, ddm_comm_z, size_of_double);
    }

    int GetProccessID_X() {
        return GetProcessID(ddm_comm_x);
    }

    int GetProccessID_Y() {
        return GetProcessID(ddm_comm_y);
    }

    int GetProccessID_Z() {
        return GetProcessID(ddm_comm_z);
    }
};

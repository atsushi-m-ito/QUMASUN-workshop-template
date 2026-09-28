#pragma once
#include <mpi.h>
#include "mpi_helper.h"
//#include "wrap_lapack.h"
#include "w_dgemm.h"
#include "print_matrix.h"

/*
* cblass_zgemmをmpiで1次元並列するための関数.
* 行列行列積をMPIで1次元並列する
* C += A*Bにおいて、B側を縦に分割(N列をN/P列毎に持たせる)
* 結果のCはN/P列だけが信頼できるので、あとからAllgatherで結合する.
* ただし、この関数の内部ではMPI通信は行わず、あくまで各コアの処理を担当.
* Allgatherはzmatrix_gather_mpi1d()関数で行う.
*
* note: 
* - OrderはColumn majorが前提
* - TransBについてはnotransが前提
*/
void cblas_zgemm_mpi1d(int proc_id, int num_procs, char TransA, char TransB,
    const int M, const int N, const int K,
    const void* alpha, const void* A, const int lda, const void* B, const int ldb, const void* beta, void* C, const int ldc)
{
    
    const int begin_n = (N * proc_id) / num_procs;
    const int end_n = (N * (proc_id + 1)) / num_procs;
    
    blas_ZGEMM(TransA, TransB, 
        M, end_n - begin_n, K,
        alpha, A, lda,
        (((const double*)B) + begin_n * ldb * 2), ldb, beta,
        (((double*)C) + begin_n * ldc * 2), ldc);

}

void zmatrix_gather_mpi1d(int proc_id, int num_procs, const MPI_Comm& comm, int ldc, int N, void* C) {
    
    if (num_procs <= 1)return;

    auto heads = std::make_unique<int[]>(num_procs *2);
    int* sizes = &heads[0] + num_procs;
    int offset = 0;
    for (int p = 0; p < num_procs; ++p) {
        const int end_n = ((N * (p + 1)) / num_procs ) * ldc *2;
        heads[p] = offset;
        sizes[p] = end_n - offset;
        offset = end_n;
        //printf("[%d]zmatrix[%d]: [%d, %d], %d, %d\n", proc_id, p, heads[p], sizes[p], num_procs, N);
    }

#if 0
    //substitute for Alltoallv, but hang-up if num processes is large.
    gy::Synchronize();
    MPI_Gatherv(MPI_IN_PLACE, 0, MPI_DOUBLE,
        C, &sizes[0], &heads[0], MPI_DOUBLE, 0, comm);
    MPI_Bcast(C, N*ldc*2, MPI_DOUBLE, 0, comm);
    gy::Synchronize();

#else
    gy::Synchronize();
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE,
        C, &sizes[0], &heads[0], MPI_DOUBLE, comm);
    gy::Synchronize();
#endif 


}


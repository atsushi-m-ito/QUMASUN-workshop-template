#ifdef USE_MPI
#pragma once
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "wrap_scalapack.h"
#include "w_zhegvx.h"
#include "w_zhegvd.h"
#include "w_pdpotrf.h"
#include "w_pzheevd.h"
#include "w_pdtrtrs.h"


/*
ScaLAPACKにはPZHEGVDが存在しないので、
Cholesky分解とPZHEEVDを用いてPZHEGVDを代替する関数

ZTRGSTの代わりににZTRSMを使う.
*/
template<class MYCOMPLEX>
inline
int lapack_PZHEGVD_impl2_2(char UPLO, MYCOMPLEX* A, MYCOMPLEX* B, double* eigen_values, MYCOMPLEX* eigen_vectors, int N, const BlacsGridInfo& blacs_grid) {


    int ITYPE = 1;	// 1 indicate "A v = e B v" type problems//
    char  JOBZ = 'V';//((eigen_vectors) ? 'V' : 'N');

    int n = N;
    int ZERO = 0;

    MPI_Comm mpi_comm = blacs_grid.mpi_comm;
    if (mpi_comm == MPI_COMM_NULL) {
        return 0;
    }

    //printf("ScaLapack: start\n");

    int proc_id, num_procs;
    MPI_Comm_size(mpi_comm, &num_procs);
    MPI_Comm_rank(mpi_comm, &proc_id);


    int np_rows = blacs_grid.np_rows;
    int np_cols = blacs_grid.np_cols;
    int icontxt = blacs_grid.icontxt;


    int nblk;  //width of block cyclic//
    //check for num_proc is much larger than matrix size//
    {

        auto bits_msb = [](unsigned int v)
            {
                v = v | (v >> 1);
                v = v | (v >> 2);
                v = v | (v >> 4);
                v = v | (v >> 8);
                v = v | (v >> 16);
                return v ^ (v >> 1);
            };

        const int max_col_row = std::max(np_cols, np_rows);
        nblk = bits_msb(N / max_col_row);
        nblk = std::min(nblk, 64);

#if 1//def DEBUG_PRINT
        if (proc_id == 0) {
            printf("Block size in ScaLAPACK [%d/%d]: nblk=%d, np_rows=%d, np_cols=%d\n", proc_id, num_procs, nblk, np_rows, np_cols); fflush(stdout);
        }
#endif

        if (nblk <= 0) {  //when matrix size if much smaller than num_procs, call LAPACK on root procs//

            if (proc_id == 0) {
                return lapack_ZHEGVD(A, B, eigen_values, eigen_vectors, N);
            } else {
                return 0;
            }
        }
    }


    int myrow, mycol;
    blacs_gridinfo_(&icontxt, &np_rows, &np_cols, &myrow, &mycol);

#ifdef DEBUG_PRINT
    printf("blacs_gridinfo_ [%d/%d contxt=%d]: myrow=%d, mycol=%d\n", proc_id, num_procs, icontxt, myrow, mycol);
#endif

    int local_row_size = numroc_(&n, &nblk, &myrow, &ZERO, &np_rows);
    int local_col_size = numroc_(&n, &nblk, &mycol, &ZERO, &np_cols);

#ifdef DEBUG_PRINT
    printf("numroc_ [%d/%d contxt=%d]: local_row_size=%d, local_col_size=%d\n", proc_id, num_procs, icontxt, local_row_size, local_col_size);
#endif
    //#undef DEBUG_PRINT    //NG
    struct double2 { double x; double y; };
    double2* Ad = (double2*)A;
    double2* Bd = (double2*)B;
    double2* local_a = new double2[local_row_size * local_col_size * 2];
    double2* local_b = new double2[local_row_size * local_col_size * 2];
    double2* local_z = new double2[local_row_size * local_col_size * 2];

    if (UPLO == 'L') {
        for (int j = 0; j < local_col_size; j++) {
            for (int i = 0; i < local_row_size; i++) {
                int ig = np_rows * nblk * ((i) / nblk) + (i) % nblk + ((np_rows + myrow) % np_rows) * nblk;
                int jg = np_cols * nblk * ((j) / nblk) + (j) % nblk + ((np_cols + mycol) % np_cols) * nblk;
                if (ig >= jg) {
                    local_a[j * local_row_size + i] = Ad[ig + N * jg];
                    local_b[j * local_row_size + i] = Bd[ig + N * jg];
                } else {
                    local_a[j * local_row_size + i].x = Ad[jg + N * ig].x;
                    local_a[j * local_row_size + i].y = -Ad[jg + N * ig].y;
                    local_b[j * local_row_size + i].x = Bd[jg + N * ig].x;
                    local_b[j * local_row_size + i].y = -Bd[jg + N * ig].y;
                }
            }
        }
    } else {
        for (int j = 0; j < local_col_size; j++) {
            for (int i = 0; i < local_row_size; i++) {
                int ig = np_rows * nblk * ((i) / nblk) + (i) % nblk + ((np_rows + myrow) % np_rows) * nblk;
                int jg = np_cols * nblk * ((j) / nblk) + (j) % nblk + ((np_cols + mycol) % np_cols) * nblk;
                if (ig <= jg) {
                    local_a[j * local_row_size + i] = Ad[ig + N * jg];
                    local_b[j * local_row_size + i] = Bd[ig + N * jg];
                } else {
                    local_a[j * local_row_size + i].x = Ad[jg + N * ig].x;
                    local_a[j * local_row_size + i].y = -Ad[jg + N * ig].y;
                    local_b[j * local_row_size + i].x = Bd[jg + N * ig].x;
                    local_b[j * local_row_size + i].y = -Bd[jg + N * ig].y;
                }
            }
        }
    }



    int descA[9] = { 0 };//size is 9//
    int max_a_row = local_row_size;//MAX(1, localA_row);//	
    int info;
    descinit_(descA, &n, &n, &nblk, &nblk, &ZERO, &ZERO, &icontxt, &max_a_row, &info);
    int descZ[9] = { 0 };
    descinit_(descZ, &n, &n, &nblk, &nblk, &ZERO, &ZERO, &icontxt, &max_a_row, &info);

#ifdef DEBUG_PRINT
    printf("descinit_ [%d/%d contxt=%d]: descA=%d, %d, %d, %d, %d, %d, %d, %d, %d\n", proc_id, num_procs, icontxt, descA[0], descA[1], descA[2], descA[3], descA[4], descA[5], descA[6], descA[7], descA[8]);
    fflush(stdout);
#endif

    int ia = 1;  //always 1 is enough//
    int ja = 1;
    int iz = 1;
    int jz = 1;

    int lwork;
    int lrwork;
    {
        int nn = std::max(std::max(n, nblk), 2);
        int NP0 = numroc_(&nn, &nblk, &ZERO, &ZERO, &np_rows);
        int NQ0 = numroc_(&nn, &nblk, &ZERO, &ZERO, &np_cols);
        lwork = n + (NP0 + NQ0 + nblk) * nblk;


        const int rsrc_ = 7;
        const int csrc_ = 8;
        int iarow = indxg2p_(&ia, &nblk, &myrow, &descZ[rsrc_ - 1], &np_rows);
        int iacol = indxg2p_(&ja, &nblk, &mycol, &descZ[csrc_ - 1], &np_cols);
        int NP = numroc_(&n, &nblk, &myrow, &iarow, &np_rows);
        int NQ = numroc_(&n, &nblk, &mycol, &iacol, &np_cols);
        lrwork = 1 + 9 * N + 3 * NP * NQ;

    }
    int liwork;
    {
        liwork = 7 * n + 8 * np_cols + 2;
    }
#ifdef DEBUG_PRINT    //OK
    printf("worksize [%d/%d contxt=%d]: lwork=%d, liwork=%d\n", proc_id, num_procs, icontxt, lwork, liwork);
#endif



    info = 0;

    //PZPOTRF
    pzpotrf_(&UPLO, &n, (lapack_complex_double*)local_b, &ia, &ja, descA, &info);
#ifdef DEBUG_PRINT
    printf("[%d] pdpotrf_ info = %d\n", proc_id, info);
    fflush(stdout);
#endif



    char cU = 'U';
    char cL = 'L';
    char cR = 'R';
    char cC = 'C';
    char cN = 'N';
    lapack_complex_double alpha{ 1.0, 0.0 }; //return value but always 1.0 in the general ScaLAPACK version//

#if 1

    // C = L^ { -1 } A(L ^ t)^ { -1 }
    //here, L == local_b;
    if (UPLO == 'L') {
        pztrsm_(&cL, &UPLO, &cN, &cN, &n, &n, &alpha, (lapack_complex_double*)local_b, &ia, &ja, descA,
            (lapack_complex_double*)local_a, &ia, &ja, descA);
        pztrsm_(&cR, &UPLO, &cC, &cN, &n, &n, &alpha, (lapack_complex_double*)local_b, &ia, &ja, descA,
            (lapack_complex_double*)local_a, &ia, &ja, descA);
    } else {
        pztrsm_(&cL, &UPLO, &cC, &cN, &n, &n, &alpha, (lapack_complex_double*)local_b, &ia, &ja, descA,
            (lapack_complex_double*)local_a, &ia, &ja, descA);
        pztrsm_(&cR, &UPLO, &cN, &cN, &n, &n, &alpha, (lapack_complex_double*)local_b, &ia, &ja, descA,
            (lapack_complex_double*)local_a, &ia, &ja, descA);
    }
#ifdef DEBUG_PRINT
    printf("pztrsm_ [%d/%d contxt=%d]:\n", proc_id, num_procs, icontxt); fflush(stdout);
#endif

#else
    //PZHEGST	
    double scale = 1.0; //return value but always 1.0 in the general ScaLAPACK version//

    pzhegst_(&ITYPE, &UPLO, &n, (lapack_complex_double*)local_a, &ia, &ja, descA,
        (lapack_complex_double*)local_b, &iz, &jz, descZ, &scale, &info);
#ifdef DEBUG_PRINT
    printf("[%d] pdsygst_ info = %d\n", proc_id, info);
    fflush(stdout);
#endif
#endif



    MYCOMPLEX* work = new MYCOMPLEX[lwork];
    double* rwork = new double[lrwork];
    int* iwork = new int[liwork];

    pzheevd_(&JOBZ, &UPLO, &n, (lapack_complex_double*)local_a, &ia, &ja, descA,
        eigen_values, (lapack_complex_double*)local_z, &iz, &jz, descZ,
        (lapack_complex_double*)work, &lwork, rwork, &lrwork, iwork, &liwork, &info);

#ifdef DEBUG_PRINT
    printf("pdsygvx_ [%d/%d contxt=%d]: info=%d\n", proc_id, num_procs, icontxt, info); fflush(stdout);
#endif

    //blacs_gridexit_(&icontxt);

#ifdef DEBUG_PRINT
    printf("Cblacs_gridexit [%d/%d contxt=%d]:\n", proc_id, num_procs, icontxt); fflush(stdout);
#endif
    delete[] work;
    delete[] iwork;
    delete[] rwork;


#if 1
    if (UPLO == 'L') {
        pztrsm_(&cL, &UPLO, &cC, &cN, &n, &n, &alpha, (lapack_complex_double*)local_b, &ia, &ja, descA,
            (lapack_complex_double*)local_z, &iz, &jz, descZ);
    } else {
        pztrsm_(&cL, &UPLO, &cN, &cN, &n, &n, &alpha, (lapack_complex_double*)local_b, &ia, &ja, descA,
            (lapack_complex_double*)local_z, &iz, &jz, descZ);
    }

#else
    //PZTRTRS
    char TRANS = (UPLO == 'U') ? 'N' : 'C';
    char is_unit = 'N';
    pztrtrs_(&UPLO, &TRANS, &is_unit, &n, &n, (lapack_complex_double*)local_b, &ia, &ja, descA,
        (lapack_complex_double*)local_z, &iz, &jz, descZ, &info);
#endif


#if 1

    const int root_id = 0;
    int* xy_inod = nullptr;
    if (proc_id == root_id) {
        xy_inod = new int[4 * num_procs];
    }

    int my_inod[4];
    my_inod[0] = myrow;
    my_inod[1] = mycol;
    my_inod[2] = local_row_size;
    my_inod[3] = local_col_size;

    MPI_Gather(my_inod, 4, MPI_INT, xy_inod, 4, MPI_INT, 0, mpi_comm);

    scalapack_GatherMatrix<double2>(N, nblk, (proc_id == root_id) ? xy_inod : my_inod, local_z, (double2*)eigen_vectors, mpi_comm, root_id, blacs_grid);

    delete[] xy_inod;

#else

    int* xy_inod = nullptr;
    int* local_size_list = nullptr;
    int* displs = nullptr;
    if (proc_id == 0) {
        xy_inod = new int[4 * num_procs];
        local_size_list = new int[num_procs];
        displs = new int[num_procs];
    }


    int my_inod[4];
    my_inod[0] = myrow;
    my_inod[1] = mycol;
    my_inod[2] = local_row_size;
    my_inod[3] = local_col_size;
#ifdef DEBUG_PRINT
    printf("MPI_Gather before[%d/%d contxt=%d]: %d x %d\n", proc_id, num_procs, icontxt, local_row_size, local_col_size); fflush(stdout);
    MPI_Barrier(mpi_comm);
#endif
    MPI_Gather(&my_inod[0], 4, MPI_INT, &xy_inod[0], 4, MPI_INT, 0, mpi_comm);
#ifdef DEBUG_PRINT
    printf("MPI_Gather [%d/%d contxt=%d]:\n", proc_id, num_procs, icontxt); fflush(stdout);
#endif
    int total_grid_size = 0;
    MYCOMPLEX* z_buffers = nullptr;
    if (proc_id == 0) {

        for (int pid = 0; pid < num_procs; ++pid) {
            //int irow = xy_inod[pid*4 + 0];
            //int icol = xy_inod[pid * 4 + 1];
            int LOCr = xy_inod[pid * 4 + 2];
            int LOCc = xy_inod[pid * 4 + 3];
#ifdef DEBUG_PRINT
            printf("[%d/%d contxt=%d]:LOCr = %d, LOCc = %d\n", proc_id, num_procs, icontxt, LOCr, LOCc); fflush(stdout);
#endif
            local_size_list[pid] = LOCr * LOCc;
            displs[pid] = total_grid_size;
            total_grid_size += LOCr * LOCc;
        }
#ifdef DEBUG_PRINT
        printf("total_grid_size [%d/%d contxt=%d]: %d\n", proc_id, num_procs, icontxt, total_grid_size); fflush(stdout);
#endif

        z_buffers = new MYCOMPLEX[total_grid_size];
        for (int i = 0; i < total_grid_size; ++i) {
            z_buffers[i] = { 0.0,0.0 };
        }

        MPI_Gatherv(local_z, local_row_size * local_col_size, MPI_DOUBLE_COMPLEX, z_buffers, local_size_list, displs, MPI_DOUBLE_COMPLEX, 0, mpi_comm);

#ifdef DEBUG_PRINT
        printf("MPI_Gatherv [%d/%d contxt=%d]:\n", proc_id, num_procs, icontxt); fflush(stdout);
#endif

        for (int pid = 0; pid < num_procs; ++pid) {
            int irow = xy_inod[pid * 4 + 0];
            int icol = xy_inod[pid * 4 + 1];
            int LOCr = xy_inod[pid * 4 + 2];
            int LOCc = xy_inod[pid * 4 + 3];
            int offset = displs[pid];

            for (int j = 0; j < LOCc; j++) {
                for (int i = 0; i < LOCr; i++) {
                    int ig = np_rows * nblk * ((i) / nblk) + (i) % nblk + ((np_rows + irow) % np_rows) * nblk;
                    int jg = np_cols * nblk * ((j) / nblk) + (j) % nblk + ((np_cols + icol) % np_cols) * nblk;
                    eigen_vectors[ig + N * jg] = z_buffers[offset + j * LOCr + i];
                }
            }

        }
        delete[] xy_inod;
        delete[] local_size_list;
        delete[] displs;
        delete[] z_buffers;
    } else {
#ifdef DEBUG_PRINT
        printf("MPI_Gatherv before [%d/%d contxt=%d]: %d x %d\n", proc_id, num_procs, icontxt, local_row_size, local_col_size); fflush(stdout);
#endif
        MPI_Gatherv(local_z, local_row_size * local_col_size, MPI_DOUBLE_COMPLEX, nullptr, local_size_list, displs, MPI_DOUBLE_COMPLEX, 0, mpi_comm);

#ifdef DEBUG_PRINT
        printf("MPI_Gatherv [%d/%d contxt=%d]:\n", proc_id, num_procs, icontxt); fflush(stdout);
#endif
    }
#endif

    delete[] local_a;
    delete[] local_b;
    delete[] local_z;

#ifdef DEBUG_PRINT
    printf("pzhegvd_impl2 [%d/%d contxt=%d]: info=%d\n", proc_id, num_procs, icontxt, info); fflush(stdout);
#endif
    return info;

}


#endif

#ifdef USE_MPI
#pragma once
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "wrap_scalapack.h"
#include "w_dtrtrs.h"

//note: meaning of element of descinit_
	//parameter( block_cyclic_2d = 1, dlen_ = 9, dtype_ = 1,
	//           ctxt_ = 2, m_ = 3, n_ = 4, mb_ = 5, nb_ = 6,
	//			 rsrc_ = 7, csrc_ = 8, lld_ = 9 )


inline
int lapack_PDTRTRS(char UPLO, char TRANS, char is_unit, int N, double* L_or_U, double* A, double* RESULT, const MPI_Comm& mpi_comm, const BlacsGridInfo& blacs_grid) {


	//INTEGER ITYPE = 1;	// 1 indicate "A v = e B v" type problems//
	char  JOBZ = 'V';//((eigen_vectors) ? 'V' : 'N');

	int n = N;
	

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

#ifdef DEBUG_PRINT
		if (proc_id == 0) {
			printf("Block size in ScaLAPACK [%d/%d]: nblk=%d, np_rows=%d, np_cols=%d\n", proc_id, num_procs, nblk, np_rows, np_cols);
		}
#endif
		if (nblk <= 0) {  //when matrix size if much smaller than num_procs, call LAPACK on root procs//

			if (proc_id == 0) {
				return lapack_DTRTRS(UPLO, TRANS, is_unit, N, L_or_U, A, RESULT);
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
	const int ZERO = 0;
	int local_row_size = numroc_(&n, &nblk, &myrow, &ZERO, &np_rows);
	int local_col_size = numroc_(&n, &nblk, &mycol, &ZERO, &np_cols);

#ifdef DEBUG_PRINT
	printf("numroc_ [%d/%d contxt=%d]: local_row_size=%d, local_col_size=%d\n", proc_id, num_procs, icontxt, local_row_size, local_col_size);
#endif


	double* local_a = new double[local_row_size * local_col_size * 2];
	double* local_b = new double[local_row_size * local_col_size * 2];


	for (int j = 0; j < local_col_size; j++) {
		for (int i = 0; i < local_row_size; i++) {
			int ig = np_rows * nblk * ((i) / nblk) + (i) % nblk + ((np_rows + myrow) % np_rows) * nblk;
			int jg = np_cols * nblk * ((j) / nblk) + (j) % nblk + ((np_cols + mycol) % np_cols) * nblk;
			local_a[j * local_row_size + i] = A[ig + N * jg];
			local_b[j * local_row_size + i] = L_or_U[ig + N * jg];
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
#endif

	int ia = 1;  //always 1 is enough//
	int ja = 1;
	int iz = 1;
	int jz = 1;

	pdtrtrs_(&UPLO, &TRANS, &is_unit, &n, &n, local_b, &ia, &ja, descA,
		local_a, &iz, &jz, descZ, &info);


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

	MPI_Gather(my_inod, 4, MPI_INT, xy_inod, 4, MPI_INT, 0, mpi_comm);

	int total_grid_size = 0;
	double* z_buffers = nullptr;
	if (proc_id == 0) {

		for (int pid = 0; pid < num_procs; ++pid) {
			//int irow = xy_inod[pid*4 + 0];
			//int icol = xy_inod[pid * 4 + 1];
			int LOCr = xy_inod[pid * 4 + 2];
			int LOCc = xy_inod[pid * 4 + 3];
			local_size_list[pid] = LOCr * LOCc;
			displs[pid] = total_grid_size;
			total_grid_size += LOCr * LOCc;
		}

		z_buffers = new double[total_grid_size];
		MPI_Gatherv(local_a, local_row_size* local_col_size, MPI_DOUBLE, z_buffers, local_size_list, displs, MPI_DOUBLE, 0, mpi_comm);		


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
					RESULT[ig + N * jg] = z_buffers[offset + j * LOCr + i];
				}
			}

		}
		delete[] xy_inod;
		delete[] local_size_list;
		delete[] displs;
		delete[] z_buffers;
	} else {
		MPI_Gatherv(local_a, local_row_size* local_col_size, MPI_DOUBLE, nullptr, local_size_list, displs, MPI_DOUBLE, 0, mpi_comm);
	}


	delete[] local_a;
	delete[] local_b;

	return info;

}


inline
int lapack_PZTRTRS(char UPLO, char TRANS, char is_unit, int N, lapack_complex_double* L_or_U, lapack_complex_double* A, lapack_complex_double* RESULT, const MPI_Comm& mpi_comm, const BlacsGridInfo& blacs_grid) {


	//INTEGER ITYPE = 1;	// 1 indicate "A v = e B v" type problems//
	char  JOBZ = 'V';//((eigen_vectors) ? 'V' : 'N');

	int n = N;


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

#ifdef DEBUG_PRINT
		if (proc_id == 0) {
			printf("Block size in ScaLAPACK [%d/%d]: nblk=%d, np_rows=%d, np_cols=%d\n", proc_id, num_procs, nblk, np_rows, np_cols);
		}
#endif
		if (nblk <= 0) {  //when matrix size if much smaller than num_procs, call LAPACK on root procs//

			if (proc_id == 0) {
				return lapack_ZTRTRS(UPLO, TRANS, is_unit, N, L_or_U, A, RESULT);
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
	const int ZERO = 0;
	int local_row_size = numroc_(&n, &nblk, &myrow, &ZERO, &np_rows);
	int local_col_size = numroc_(&n, &nblk, &mycol, &ZERO, &np_cols);

#ifdef DEBUG_PRINT
	printf("numroc_ [%d/%d contxt=%d]: local_row_size=%d, local_col_size=%d\n", proc_id, num_procs, icontxt, local_row_size, local_col_size);
#endif


	lapack_complex_double* local_a = new lapack_complex_double[local_row_size * local_col_size * 2];
	lapack_complex_double* local_b = new lapack_complex_double[local_row_size * local_col_size * 2];


	for (int j = 0; j < local_col_size; j++) {
		for (int i = 0; i < local_row_size; i++) {
			int ig = np_rows * nblk * ((i) / nblk) + (i) % nblk + ((np_rows + myrow) % np_rows) * nblk;
			int jg = np_cols * nblk * ((j) / nblk) + (j) % nblk + ((np_cols + mycol) % np_cols) * nblk;
			local_a[j * local_row_size + i] = A[ig + N * jg];
			local_b[j * local_row_size + i] = L_or_U[ig + N * jg];
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
#endif

	int ia = 1;  //always 1 is enough//
	int ja = 1;
	int iz = 1;
	int jz = 1;

	pztrtrs_(&UPLO, &TRANS, &is_unit, &n, &n, (lapack_complex_double*)local_b, &ia, &ja, descA,
		(lapack_complex_double*)local_a, &iz, &jz, descZ, &info);


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

	MPI_Gather(my_inod, 4, MPI_INT, xy_inod, 4, MPI_INT, 0, mpi_comm);

	int total_grid_size = 0;
	lapack_complex_double* z_buffers = nullptr;
	if (proc_id == 0) {

		for (int pid = 0; pid < num_procs; ++pid) {
			//int irow = xy_inod[pid*4 + 0];
			//int icol = xy_inod[pid * 4 + 1];
			int LOCr = xy_inod[pid * 4 + 2];
			int LOCc = xy_inod[pid * 4 + 3];
			local_size_list[pid] = LOCr * LOCc;
			displs[pid] = total_grid_size;
			total_grid_size += LOCr * LOCc;
		}

		z_buffers = new lapack_complex_double[total_grid_size];
		MPI_Gatherv(local_a, local_row_size * local_col_size, MPI_DOUBLE_COMPLEX, z_buffers, local_size_list, displs, MPI_DOUBLE_COMPLEX, 0, mpi_comm);


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
					RESULT[ig + N * jg] = z_buffers[offset + j * LOCr + i];
				}
			}

		}
		delete[] xy_inod;
		delete[] local_size_list;
		delete[] displs;
		delete[] z_buffers;
	} else {
		MPI_Gatherv(local_a, local_row_size * local_col_size, MPI_DOUBLE_COMPLEX, nullptr, local_size_list, displs, MPI_DOUBLE_COMPLEX, 0, mpi_comm);
	}


	delete[] local_a;
	delete[] local_b;

	return info;

}

#endif

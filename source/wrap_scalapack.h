#pragma once
#include <mpi.h>

#if defined(__INTEL_LLVM_COMPILER ) || defined(__INTEL_COMPILER )
#include <complex>
#include <mkl.h>
#include <mkl_lapack.h>
#include <mkl_lapacke.h>
#include <mkl_scalapack.h>
#elif defined(_NEC) || defined(__aocc__)
#undef HAVE_LAPACK_CONFIG_H
#include <complex>
#include <cblas.h>
//#define lapack_complex_double std::complex<double>
//#include <lapack.h>
#include <lapacke.h>

#else
//#define HAVE_LAPACK_CONFIG_H
#define LAPACK_COMPLEX_CPP
#include <complex>
//#define lapack_complex_float std::complex<float>
//#define lapack_complex_double std::complex<double>
#define LAPACK_GLOBAL_PATTERN_UC
#define __EMSCRIPTEN__
#include <cblas.h>
//#include <f77blas.h>
#include <lapack.h>
#include <lapacke.h>
//#define MKL_Complex16 std::complex<double>
#endif

using INTEGER = int;
/*
#ifndef ___dcomplex_definition___
using dcomplex = std::complex<double>;
//typedef struct { double r,i; } dcomplex;
#define ___dcomplex_definition___ 
#endif
*/
#ifndef LAPACK_ABSTOL
#define LAPACK_ABSTOL (1.0E-13)
#endif



extern "C" {
    int Csys2blacs_handle(MPI_Comm comm);

    void sl_init_(int* icontext, int* nprow, int* npcolumn);
    void blacs_get_(int* icontext, const int* what, int* val);

    void blacs_gridinit_(int* ConTxt, const char* layout, const int* nprow, const int* npcol);

    void blacs_gridinfo_(int* icontext, int* nprow, int* npcolumn, int* myrow,
        int* mycolumn);

    void blacs_gridmap_(int* icontext, int* usermap, int* ldumap, int* nprow, int* npcolumn);
    

    int numroc_(const int* n, const int* nb, const int* iproc,
        const int* isrcproc, const int* nprocs);

	int indxg2p_(const int* n, const int* nb, const int* iproc,
		const int* isrcproc, const int* nprocs);
    
	//note: meaning of element of descinit_
	//parameter( block_cyclic_2d = 1, dlen_ = 9, dtype_ = 1,
	//           ctxt_ = 2, m_ = 3, n_ = 4, mb_ = 5, nb_ = 6,
	//			 rsrc_ = 7, csrc_ = 8, lld_ = 9 )
    void descinit_(int* desc, const int* m, const int* n,
        const int* mb, const int* nb, const int* irsrc,
        const int* icsrc, const int* ictxt, const int* lld,
        int* info);


    void pdsyevx_(const char* jobz, const char* range, const char* uplo, const int* n,
        const double* a, const int* ia, const int* ja, const int* desca,
        const double* vl, const double* vu, const int* il, const int* iu,
        const double* abstol, int* m, int* nz, double* w, const double* orfac,
        double* z, const int* iz, const int* jz, const int* descz,
        double* work, const int* lwork, int* iwork, const int* liwork,
        int* ifail, int* iclustr, double* gap, int* info);
    
    void pdsyevd_(const char* jobz, const char* uplo, const int* n,
        const double* a, const int* ia, const int* ja, const int* desca,
        double* w, double* z, const int* iz, const int* jz, const int* descz, 
        double* work, const int* lwork, int* iwork, const int* liwork, int* info);


	void pdsytrd_(const char* uplo, const int* n,
		double* a, const int* ia, const int* ja, const int* desca,
		double* d, double* e, double* tau, 
		double* work, const int* lwork, int* info);
	//void pdsytrd_(const char* uplo, const int* n, 
		//double* a, const MKL_INT* ia, const MKL_INT* ja, const MKL_INT* desca, double* d, double* e, double* tau, double* work, const MKL_INT* lwork, MKL_INT* info);

	void pdlaset_(const char* uplo, const int* m, const int* n, const double* alpha, const double* beta, double* a, const int* ia, const int* ja, const int* desca);

	void pdstedc_(const char* compz, const int* n,
		double* d, double* e,
		double* z, const int* iz, const int* jz, const int* descz,
		double* work, int* lwork,
		int* iwork, const int* liwork, int* info);
	/*
	void pdstedc_(const char* compz, const int* n, 
		double* d, double* e, 
		double* q, const int* iq, const int* jq, const int* descq, 
		double* work, int* lwork, 
		int* iwork, const int* liwork, MKL_INT* info);
		*/
	void pdormtr_(const char* side, const char* uplo, const char* trans,
		const int* m, const int* n,
		const double* a, const int* ia, const int* ja, const int* desca,
		const double* tau,
		double* c, const int* ic, const int* jc, const int* descc,
		double* work, const int* lwork, int* info);
	/*
	void pdormtr_(const char* side, const char* uplo, const char* trans,
		const MKL_INT* m, const MKL_INT* n, 
		const double* a, const MKL_INT* ia, const MKL_INT* ja, const MKL_INT* desca,
		const double* tau, double* c, const MKL_INT* ic, const MKL_INT* jc, const MKL_INT* descc, 
		double* work, const MKL_INT* lwork, MKL_INT* info);
		*/
	void pdlared1d_(const int* n, const int* ia, const int* ja, const int* desc,
		const double* bycol, double* byall, double* work, const int* lwork);

    void pdsygvx_(const int* ibtype, const char* jobz, const char* range, const char* uplo, const int* n, double* a, const int* ia, const int* ja, const int* desca, double* b, const int* ib, const int* jb, const int* descb, const double* vl, const double* vu, const int* il, const int* iu, const double* abstol, int* m, int* nz, double* w, const double* orfac, double* z, const int* iz, const int* jz, const int* descz, double* work, const int* lwork, int* iwork, const int* liwork, int* ifail, int* iclustr, double* gap, int* info);
    

    void pzhegvx_(const int* ibtype, const char* jobz, const char* range, const char* uplo, const int* n, lapack_complex_double* a, const int* ia, const int* ja, const int* desca, lapack_complex_double* b, const int* ib, const int* jb, const int* descb, const double* vl, const double* vu, const int* il, const int* iu, const double* abstol, int* m, int* nz, double* w, const double* orfac, lapack_complex_double* z, const int* iz, const int* jz, const int* descz, lapack_complex_double* work, const int* lwork, double* rwork, const int* lrwork, int* iwork, const int* liwork, int* ifail, int* iclustr, double* gap, int* info);

	void pdpotrf_(const char* UPLO, const int* N, double* A, const int* IA, const int* JA, const int* DESCA, int* INFO);

	void pdsygst_(const int* IBTYPE, const char* UPLO, const int* N, double* A, const int* IA, const int* JA, const int* DESCA, const double* B, const int* IB, const int* JB, const int* DESCB, double* SCALE, int* INFO);

	void pdtrtrs_(const char* UPLO, const char* TRANS, const char* DIAG, const int* N, const int* NRHS,
		const double* A, const int* IA, const int* JA, const int* DESCA,
		double* B, const int* IB, const int* JB, const int* DESCB, int* INFO);

	void pzheevd_(const char* jobz, const char* uplo, const int* n, 
		const lapack_complex_double* a, const int* ia, const int* ja, const int* desca,
		double* w,
		lapack_complex_double* z, const int* iz, const int* jz, const int* descz,
		lapack_complex_double* work, const int* lwork, double* rwork, const int* lrwork,
		int* iwork, const int* liwork, int* info);

	void pzpotrf_(const char* UPLO, const int* N, lapack_complex_double* A, const int* IA, const int* JA, const int* DESCA, int* INFO);

	void pzhegst_(const int* IBTYPE, const char* UPLO, const int* N, lapack_complex_double* A, const int* IA, const int* JA, const int* DESCA, const lapack_complex_double* B, const int* IB, const int* JB, const int* DESCB, double* SCALE, int* INFO);
	
	void pztrtrs_(const char* UPLO, const char* TRANS, const char* DIAG, const int* N, const int* NRHS,
		const lapack_complex_double* A, const int* IA, const int* JA, const int* DESCA,
		lapack_complex_double* B, const int* IB, const int* JB, const int* DESCB, int* INFO);

	void pztrsm_(const char* SIDE, const char* UPLO, const char* TRANS, const char* DIAG, const int* M, const int* N, lapack_complex_double* ALPHA, lapack_complex_double* A, const int* IA, const int* JA, const int* DESCA, lapack_complex_double* B, const int* IB, const int* JB, const int* DESCB);

    void blacs_exit_(int* cont);


    void blacs_gridexit_(int* icontext);

}



struct BlacsGridInfo {
	int icontxt;
	int np_rows;
	int np_cols;
	int pos1;
	int pos2;
    MPI_Comm mpi_comm;
};

inline
BlacsGridInfo BeginBLACS(MPI_Comm& mpi_comm, int group_id) {

	int proc_id, num_procs;
	MPI_Comm_size(mpi_comm, &num_procs);
	MPI_Comm_rank(mpi_comm, &proc_id);

	int ZERO = 0;

	int np_rows = (int)(sqrt((float)num_procs));
	do {
		if ((num_procs % np_rows) == 0) break;
		np_rows--;
	} while (np_rows >= 2);

#if 1
	int np_cols = num_procs / np_rows;
#else
	int np_cols = np_rows;
	np_rows = num_procs / np_cols;
#endif


	int orgcontxt;
	blacs_get_(&ZERO, &ZERO, &orgcontxt); //get default blacs context
#ifdef DEBUG_PRINT
	printf("blacs_get_ [%d/%d contxt=%d]:group_id=%d\n", proc_id, num_procs, orgcontxt, group_id); fflush(stdout);
#endif

	//make multi-grid (corresponding MPI_Comm_split)//
	//reference: https://www.ibm.com/docs/en/pessl/5.5?topic=blacs-gridmap-routine
	int icontxt;
	int TEN = 10;
	int* usermap = new int[num_procs];
	for (int i = 0; i < num_procs; ++i) {
		usermap[i] = i + num_procs * group_id;
	}
	//blacs_get_(&orgcontxt, &TEN, &icontxt); //redefinition mode//
	icontxt = orgcontxt;

#ifdef DEBUG_PRINT
	printf("blacs_get_(2) [%d/%d contxt=%d]:icontxt=%d\n", proc_id, num_procs, orgcontxt, icontxt); fflush(stdout);
#endif

	blacs_gridmap_(&icontxt, &usermap[0], &np_rows, &np_rows, &np_cols);
#ifdef DEBUG_PRINT
	printf("blacs_gridmap_ [%d/%d contxt=%d]:icontxt=%d\n", proc_id, num_procs, orgcontxt, icontxt); fflush(stdout);
#endif
	delete[] usermap;
#ifdef DEBUG_PRINT
	if (proc_id == 0) {
		printf("Use ScaLAPACK [%d/%d,group:%d]: np_rows=%d, np_cols=%d\n", proc_id, num_procs, group_id, np_rows, np_cols);
	}
#endif

	int pos1, pos2;
	blacs_gridinfo_(&icontxt, &np_rows, &np_cols, &pos1, &pos2);
	return BlacsGridInfo{ icontxt, np_rows, np_cols , pos1, pos2, mpi_comm };
}


inline
BlacsGridInfo BeginBLACSFromMPIComm(MPI_Comm& mpi_comm, int np1, int np2) {

	int num_procs=1;
	int proc_id = 0;
	if (mpi_comm != MPI_COMM_NULL) {
		MPI_Comm_size(mpi_comm, &num_procs);
		MPI_Comm_rank(mpi_comm, &proc_id);
	}
	
	int pid_in_world;
	MPI_Comm_rank(MPI_COMM_WORLD, &pid_in_world);

	int ZERO = 0;

	int np_rows = np1;
	int np_cols = np2;


	int orgcontxt;
	blacs_get_(&ZERO, &ZERO, &orgcontxt); //get default blacs context
#ifdef DEBUG_PRINT
	printf("blacs_get_ [%d/%d contxt=%d]:group_id=%d\n", proc_id, num_procs, orgcontxt, pid_in_world); fflush(stdout);
#endif

	//make multi-grid (corresponding MPI_Comm_split)//
	//reference: https://www.ibm.com/docs/en/pessl/5.5?topic=blacs-gridmap-routine
	int icontxt = orgcontxt;
	if (mpi_comm != MPI_COMM_WORLD) {
		int* usermap = new int[num_procs];
		if (mpi_comm != MPI_COMM_NULL) {
			MPI_Allgather(&pid_in_world, 1, MPI_INT, usermap, 1, MPI_INT, mpi_comm);
#ifdef DEBUG_PRINT
			printf("[%d]usermap:", proc_id);
			for (int i = 0; i < num_procs; ++i) {
				printf(" %d", usermap[i]);
			}
			printf("\n");
			fflush(stdout);
#endif
		} else {
			usermap[0] = 0;
			np_rows = 1;
			np_cols = 1;
		}


		//icontxt = orgcontxt;

#ifdef DEBUG_PRINT
		printf("blacs_get_(2) [%d/%d contxt=%d]:icontxt=%d, %d x %d\n", proc_id, num_procs, orgcontxt, icontxt, np_rows, np_cols); fflush(stdout);
#endif
        //if (mpi_comm != MPI_COMM_NULL) 
        {
            blacs_gridmap_(&icontxt, &usermap[0], &np_rows, &np_rows, &np_cols);
        }
#ifdef DEBUG_PRINT
		printf("blacs_gridmap_ [%d/%d contxt=%d]:icontxt=%d\n", proc_id, num_procs, orgcontxt, icontxt); fflush(stdout);
#endif
		delete[] usermap;
	} else {
		char order = 'c';
		blacs_gridinit_(&icontxt, &order, &np_rows, &np_cols);
	}
#ifdef DEBUG_PRINT
	if (proc_id == 0) {
		printf("Use ScaLAPACK [%d/%d,in_world:%d]: np_rows=%d, np_cols=%d\n", proc_id, num_procs, pid_in_world, np_rows, np_cols);
	}
#endif
	
	int pos1=0, pos2=0;
    if (mpi_comm != MPI_COMM_WORLD) {
        blacs_gridinfo_(&icontxt, &np_rows, &np_cols, &pos1, &pos2);
    }

	return BlacsGridInfo{ icontxt, np_rows, np_cols , pos1, pos2, mpi_comm };
}


inline
void EndBLACS(const BlacsGridInfo& blacs_grid) {
    if (blacs_grid.mpi_comm != MPI_COMM_NULL) {
        int icontxt = blacs_grid.icontxt;
        blacs_gridexit_(&icontxt);
    }
}


inline int Numroc(
	int n,
	int nb,
	int iproc,
	int isrcproc,  //サイクリックの開始点となるプロセス番号//
	int nprocs)
{
	const int mydist =
		(nprocs + iproc - isrcproc) % nprocs;

	const int nblocks = n / nb;
	const int extrablks = nblocks % nprocs;

	int nlocal = (nblocks / nprocs) * nb;

	if (mydist < extrablks) {
		nlocal += nb;
	} else if (mydist == extrablks) {
		nlocal += n % nb;
	}
	return nlocal;
}


inline 
int scalapack_MatrixSize(int N, int width, const BlacsGridInfo& blacs_grid) {

	
	int np1 = blacs_grid.np_rows; //here np_rows is num procs in column//
	int np2 = blacs_grid.np_cols;
	int icontxt = blacs_grid.icontxt;

	int pos1, pos2;

	blacs_gridinfo_(&icontxt, &np1, &np2, &pos1, &pos2);
	const int ZERO = 0;
	const int local_row_size = numroc_(&N, &width, &pos1, &ZERO, &np1);
	const int local_col_size = numroc_(&N, &width, &pos2, &ZERO, &np2);
	return local_row_size * local_col_size;
}

/*
* scatterやgatherするための情報を集約する関数.
* rootだけはxy_inodにバッファを入れる必要がある。
* バッファのサイズは4 * num_procsで、各プロセスのmyrow, pos2, local_row_size, local_col_sizeを格納する。
*/
inline 
void scalapack_SizeCollection(int N, int nblk, int* xy_inod, const MPI_Comm& mpi_comm, const int root_id, const BlacsGridInfo& blacs_grid) {
	const int np_rows = blacs_grid.np_rows;
	const int np_cols = blacs_grid.np_cols;
	const int pos1 = blacs_grid.pos1;
	const int pos2 = blacs_grid.pos2;
	const int icontxt = blacs_grid.icontxt;

	int proc_id, num_procs;
	MPI_Comm_size(mpi_comm, &num_procs);
	MPI_Comm_rank(mpi_comm, &proc_id);

	const int ZERO = 0;
	const int local_row_size = numroc_(&N, &nblk, &pos1, &ZERO, &np_rows);
	const int local_col_size = numroc_(&N, &nblk, &pos2, &ZERO, &np_cols);

	/*
	//int* xy_inod = nullptr;
	//int* local_size_list = nullptr;
	//int* displs = nullptr;
	if (proc_id == root_id) {
		xy_inod = new int[4 * num_procs];
		//local_size_list = new int[num_procs];
		//displs = new int[num_procs];
	}
	*/

	int my_inod[4];
	my_inod[0] = pos1;
	my_inod[1] = pos2;
	my_inod[2] = local_row_size;
	my_inod[3] = local_col_size;

	MPI_Gather(my_inod, 4, MPI_INT, xy_inod, 4, MPI_INT, root_id, mpi_comm);

}

//deplicated: xy_inodを使わなくてもよいscalapack_ScatterMatrix2を使うべし//
template<class T>
void scalapack_ScatterMatrix(int N, 
	int nblk, const int* xy_inod, 
	const T* A, T* local_a, const MPI_Comm& mpi_comm, const int root_id, const BlacsGridInfo& blacs_grid) {
	
	MPI_Datatype datatype = MPI_DOUBLE;
	if constexpr (sizeof(T) == 16) datatype = MPI_DOUBLE_COMPLEX;
	if constexpr (sizeof(T) == 4) datatype = MPI_FLOAT;

	int proc_id, num_procs;
	MPI_Comm_size(mpi_comm, &num_procs);
	MPI_Comm_rank(mpi_comm, &proc_id);

	const int np_rows = blacs_grid.np_rows;
	const int np_cols = blacs_grid.np_cols;
	const int pos1 = blacs_grid.pos1;
	const int pos2 = blacs_grid.pos2;

	const int ZERO = 0;
	const int local_row_size = numroc_(&N, &nblk, &pos1, &ZERO, &np_rows);
	const int local_col_size = numroc_(&N, &nblk, &pos2, &ZERO, &np_cols);

	if (proc_id == root_id) {
		int* local_size_list = new int[num_procs];
		int* displs = new int[num_procs];

		int total_grid_size = 0;
		for (int pid = 0; pid < num_procs; ++pid) {
			const int LOCr = xy_inod[pid * 4 + 2];
			const int LOCc = xy_inod[pid * 4 + 3];
			local_size_list[pid] = LOCr * LOCc; //複素の時はA, Bの2つ分で2倍//
			displs[pid] = total_grid_size;
			total_grid_size += LOCr * LOCc; //複素の時はA, Bの2つ分で2倍//
		}

		T* z_buffers = new T[total_grid_size];


		for (int pid = 0; pid < num_procs; ++pid) {
			const int irow = xy_inod[pid * 4 + 0];
			const int icol = xy_inod[pid * 4 + 1];
			const int LOCr = xy_inod[pid * 4 + 2];
			const int LOCc = xy_inod[pid * 4 + 3];
			int offset = displs[pid];

			for (int j = 0; j < LOCc; j++) {
				for (int i = 0; i < LOCr; i++) {
					const int ig = np_rows * nblk * ((i) / nblk) + (i) % nblk + ((np_rows + irow) % np_rows) * nblk;
					const int jg = np_cols * nblk * ((j) / nblk) + (j) % nblk + ((np_cols + icol) % np_cols) * nblk;
					z_buffers[offset + j * LOCr + i] = A[ig + N * jg]; //A
				}
			}

		}

		MPI_Scatterv(z_buffers, local_size_list, displs, datatype, local_a, local_row_size * local_col_size, datatype, root_id, mpi_comm);

		delete[] z_buffers;
		delete[] local_size_list;
		delete[] displs;
	} else {


		MPI_Scatterv(nullptr, nullptr, nullptr, datatype, local_a, local_row_size * local_col_size, datatype, root_id, mpi_comm);
	}

}


template<class T>
void scalapack_ScatterMatrix2(int N,
	int nblk, const char proc_order, /*const int* xy_inod,*/
	const T* A, T* local_a, const MPI_Comm& mpi_comm, const int root_id, const BlacsGridInfo& blacs_grid) {

	MPI_Datatype datatype = MPI_DOUBLE;
	if constexpr (sizeof(T) == 16) datatype = MPI_DOUBLE_COMPLEX;
	if constexpr (sizeof(T) == 4) datatype = MPI_FLOAT;

	int proc_id, num_procs;
	MPI_Comm_size(mpi_comm, &num_procs);
	MPI_Comm_rank(mpi_comm, &proc_id);

	const int np_rows = blacs_grid.np_rows;
	const int np_cols = blacs_grid.np_cols;
	const int pos1 = blacs_grid.pos1;
	const int pos2 = blacs_grid.pos2;

	const int ZERO = 0;
	const int local_row_size = Numroc(N, nblk, pos1, ZERO, np_rows); //numroc_(&N, &nblk, &pos1, &ZERO, &np_rows);
	const int local_col_size = Numroc(N, nblk, pos2, ZERO, np_cols); //numroc_(&N, &nblk, &pos2, &ZERO, &np_cols);

	if (proc_id == root_id) {
		int* local_size_list = new int[num_procs];
		int* displs = new int[num_procs];
		T* z_buffers = new T[N * N];

		const int nblocks = N / nblk;
		const int extrablks2 = nblocks % np_cols;
		const int nlocal2 = (nblocks / np_cols) * nblk;
		const int extrablks1 = nblocks % np_rows;
		const int nlocal1 = (nblocks / np_rows) * nblk;


		auto Clip = [&](int pid1, int pid2, const int offset) {
			const int LOCc = (pid2 < extrablks2) ? nlocal2 + nblk : (pid2 == extrablks2) ? nlocal2 + (N % nblk) : nlocal2;
			const int LOCr = (pid1 < extrablks1) ? nlocal1 + nblk : (pid1 == extrablks1) ? nlocal1 + (N % nblk) : nlocal1;

			//const int offset = total_grid_size;

			const int& irow = pid1;
			const int& icol = pid2;

			for (int j = 0; j < LOCc; j++) {
				for (int i = 0; i < LOCr; i++) {
					const int ig = np_rows * nblk * ((i) / nblk) + (i) % nblk + ((np_rows + irow) % np_rows) * nblk;
					const int jg = np_cols * nblk * ((j) / nblk) + (j) % nblk + ((np_cols + icol) % np_cols) * nblk;
					z_buffers[offset + j * LOCr + i] = A[ig + N * jg]; //A
				}
			}

			return LOCr * LOCc;
		};


		int total_grid_size = 0;
		if ((proc_order == 'c') || (proc_order == 'C')) {
			//printf("CHECK: scatter C\n"); fflush(stdout); fflush(stdout);
			
			for (int pid2 = 0; pid2 < np_cols; ++pid2) {
				for (int pid1 = 0; pid1 < np_rows; ++pid1) {
					const int pid = pid1 + np_rows * pid2;



					const int local_size = Clip(pid1, pid2, total_grid_size);
					
					local_size_list[pid] = local_size; 
					displs[pid] = total_grid_size;
					total_grid_size += local_size; 

#if 0
					//test//
					const int LOCc2 = (pid2 < extrablks2) ? nlocal2 + nblk : (pid2 == extrablks2) ? nlocal2 + (N % nblk) : nlocal2;
					const int LOCr2 = (pid1 < extrablks1) ? nlocal1 + nblk : (pid1 == extrablks1) ? nlocal1 + (N % nblk) : nlocal1;
					const int irow = xy_inod[pid * 4 + 0];
					const int icol = xy_inod[pid * 4 + 1];
					const int LOCr = xy_inod[pid * 4 + 2];
					const int LOCc = xy_inod[pid * 4 + 3];
					if (irow != pid1) {
						printf("ERROR: irow(%d) != pid1(%d)\n", irow, pid1); fflush(stdout);
					}
					if (icol != pid2) {
						printf("ERROR: icol(%d) != pid2(%d)\n", icol, pid2); fflush(stdout);
					}
					if (LOCr != LOCr2) {
						printf("ERROR: LOCr(%d) != LOCr2(%d)\n", LOCr, LOCr2); fflush(stdout);
					}
					if (LOCc != LOCc2) {
						printf("ERROR: LOCc(%d) != LOCc2(%d)\n", LOCc, LOCc2); fflush(stdout);
					}
					if (LOCr* LOCc != local_size) {
						printf("ERROR: LOCr*LOCc(%d) != local_size(%d)\n", LOCr * LOCc, local_size); fflush(stdout);
					}
#endif
				}
			}
		} else {
			//printf("CHECK: scatter R\n"); fflush(stdout); fflush(stdout);

			for (int pid1 = 0; pid1 < np_rows; ++pid1) {
				for (int pid2 = 0; pid2 < np_cols; ++pid2) {
					const int pid = pid2 + np_cols * pid1;

					const int local_size = Clip(pid1, pid2, total_grid_size);

					local_size_list[pid] = local_size;
					displs[pid] = total_grid_size;
					total_grid_size += local_size;
				}
			}
		}

		if (total_grid_size != N * N) {
			printf("ERROR: ScatterMatrix2, N^2 != total_grid_size = %d\n", total_grid_size);
		}


		MPI_Scatterv(z_buffers, local_size_list, displs, datatype, local_a, local_row_size * local_col_size, datatype, root_id, mpi_comm);

		delete[] z_buffers;
		delete[] local_size_list;
		delete[] displs;
	} else {


		MPI_Scatterv(nullptr, nullptr, nullptr, datatype, local_a, local_row_size * local_col_size, datatype, root_id, mpi_comm);
	}

}

template<class T>
void scalapack_ScatterMatrixHermite(char UPLO, int N,
	int nblk, const int* xy_inod,
	const T* A, T* local_a, const MPI_Comm& mpi_comm, const int root_id, const BlacsGridInfo& blacs_grid) {


	struct double2 {
		double x; double y;
	};
	
	
	int proc_id, num_procs;
	MPI_Comm_size(mpi_comm, &num_procs);
	MPI_Comm_rank(mpi_comm, &proc_id);

	const int np_rows = blacs_grid.np_rows;
	const int np_cols = blacs_grid.np_cols;
	const int pos1 = blacs_grid.pos1;
	const int pos2 = blacs_grid.pos2;

	const int ZERO = 0;
	const int local_row_size = numroc_(&N, &nblk, &pos1, &ZERO, &np_rows);
	const int local_col_size = numroc_(&N, &nblk, &pos2, &ZERO, &np_cols);


	int total_grid_size = 0;
	double2* z_buffers = nullptr;
	if (proc_id == 0) {
		int* local_size_list = new int[num_procs];
		int* displs = new int[num_procs];

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

		z_buffers = new double2[total_grid_size];
		for (int i = 0; i < total_grid_size; ++i) {
			z_buffers[i] = { 0.0,0.0 };
		}

#ifdef DEBUG_PRINT
		printf("MPI_Gatherv [%d/%d contxt=%d]:\n", proc_id, num_procs, icontxt); fflush(stdout);
#endif

		const double2* Ad = (const double2*)A;
		//const double2* Bd = (const double2*)B;

		for (int pid = 0; pid < num_procs; ++pid) {
			int irow = xy_inod[pid * 4 + 0];
			int icol = xy_inod[pid * 4 + 1];
			int LOCr = xy_inod[pid * 4 + 2];
			int LOCc = xy_inod[pid * 4 + 3];
			int offset = displs[pid];

			if (UPLO == 'L') {
				for (int j = 0; j < LOCc; j++) {
					for (int i = 0; i < LOCr; i++) {
						int ig = np_rows * nblk * ((i) / nblk) + (i) % nblk + ((np_rows + irow) % np_rows) * nblk;
						int jg = np_cols * nblk * ((j) / nblk) + (j) % nblk + ((np_cols + icol) % np_cols) * nblk;
						if (ig >= jg) {
							z_buffers[offset + j * LOCr + i] = Ad[ig + N * jg];
						} else {
							z_buffers[offset + j * LOCr + i].x = Ad[jg + N * ig].x;
							z_buffers[offset + j * LOCr + i].y = -Ad[jg + N * ig].y;
						}
					}
				}
			} else {
				for (int j = 0; j < LOCc; j++) {
					for (int i = 0; i < LOCr; i++) {
						int ig = np_rows * nblk * ((i) / nblk) + (i) % nblk + ((np_rows + irow) % np_rows) * nblk;
						int jg = np_cols * nblk * ((j) / nblk) + (j) % nblk + ((np_cols + icol) % np_cols) * nblk;
						if (ig <= jg) {
							z_buffers[offset + j * LOCr + i] = Ad[ig + N * jg];
						} else {
							z_buffers[offset + j * LOCr + i].x = Ad[jg + N * ig].x;
							z_buffers[offset + j * LOCr + i].y = -Ad[jg + N * ig].y;
						}
					}
				}
			}

		}


		MPI_Scatterv(z_buffers, local_size_list, displs, MPI_DOUBLE_COMPLEX, local_a, local_row_size * local_col_size, MPI_DOUBLE_COMPLEX, 0, mpi_comm);

		//delete[] xy_inod;
		delete[] local_size_list;
		delete[] displs;
		delete[] z_buffers;
	} else {

		MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DOUBLE_COMPLEX, local_a, local_row_size * local_col_size, MPI_DOUBLE_COMPLEX, 0, mpi_comm);

	}


}


template<class T>
void scalapack_ScatterMatrixHermite2(char UPLO, int N,
	int nblk, const char proc_order,
	const T* A, T* local_a, const MPI_Comm& mpi_comm, const int root_id, const BlacsGridInfo& blacs_grid) {


	struct double2 {
		double x; double y;
	};


	int proc_id, num_procs;
	MPI_Comm_size(mpi_comm, &num_procs);
	MPI_Comm_rank(mpi_comm, &proc_id);

	const int np_rows = blacs_grid.np_rows;
	const int np_cols = blacs_grid.np_cols;
	const int pos1 = blacs_grid.pos1;
	const int pos2 = blacs_grid.pos2;

	const int ZERO = 0;
	const int local_row_size = numroc_(&N, &nblk, &pos1, &ZERO, &np_rows);
	const int local_col_size = numroc_(&N, &nblk, &pos2, &ZERO, &np_cols);


	int total_grid_size = 0;
	
	if (proc_id == 0) {
		int* local_size_list = new int[num_procs];
		int* displs = new int[num_procs];
		double2* z_buffers = new double2[N*N];

		const int nblocks = N / nblk;
		const int extrablks2 = nblocks % np_cols;
		const int nlocal2 = (nblocks / np_cols) * nblk;
		const int extrablks1 = nblocks % np_rows;
		const int nlocal1 = (nblocks / np_rows) * nblk;


		const double2* Ad = (const double2*)A;

		auto Clip = [&](int pid1, int pid2, const int offset) {
			const int LOCc = (pid2 < extrablks2) ? nlocal2 + nblk : (pid2 == extrablks2) ? nlocal2 + (N % nblk) : nlocal2;
			const int LOCr = (pid1 < extrablks1) ? nlocal1 + nblk : (pid1 == extrablks1) ? nlocal1 + (N % nblk) : nlocal1;

			const int& irow = pid1;
			const int& icol = pid2;

			if (UPLO == 'L') {
				for (int j = 0; j < LOCc; j++) {
					for (int i = 0; i < LOCr; i++) {
						const int ig = np_rows * nblk * ((i) / nblk) + (i) % nblk + ((np_rows + irow) % np_rows) * nblk;
						const int jg = np_cols * nblk * ((j) / nblk) + (j) % nblk + ((np_cols + icol) % np_cols) * nblk;
						if (ig >= jg) {
							z_buffers[offset + j * LOCr + i] = Ad[ig + N * jg];
						} else {
							z_buffers[offset + j * LOCr + i].x = Ad[jg + N * ig].x;
							z_buffers[offset + j * LOCr + i].y = -Ad[jg + N * ig].y;
						}
					}
				}
			} else {
				for (int j = 0; j < LOCc; j++) {
					for (int i = 0; i < LOCr; i++) {
						const int ig = np_rows * nblk * ((i) / nblk) + (i) % nblk + ((np_rows + irow) % np_rows) * nblk;
						const int jg = np_cols * nblk * ((j) / nblk) + (j) % nblk + ((np_cols + icol) % np_cols) * nblk;
						if (ig <= jg) {
							z_buffers[offset + j * LOCr + i] = Ad[ig + N * jg];
						} else {
							z_buffers[offset + j * LOCr + i].x = Ad[jg + N * ig].x;
							z_buffers[offset + j * LOCr + i].y = -Ad[jg + N * ig].y;
						}
					}
				}
			}


			return LOCr * LOCc;
			};


		int total_grid_size = 0;
		if ((proc_order == 'c') || (proc_order == 'C')) {
			//printf("CHECK: scatter C\n"); fflush(stdout); fflush(stdout);

			for (int pid2 = 0; pid2 < np_cols; ++pid2) {
				for (int pid1 = 0; pid1 < np_rows; ++pid1) {
					const int pid = pid1 + np_rows * pid2;



					const int local_size = Clip(pid1, pid2, total_grid_size);

					local_size_list[pid] = local_size;
					displs[pid] = total_grid_size;
					total_grid_size += local_size;

#if 0
					//test//
					const int LOCc2 = (pid2 < extrablks2) ? nlocal2 + nblk : (pid2 == extrablks2) ? nlocal2 + (N % nblk) : nlocal2;
					const int LOCr2 = (pid1 < extrablks1) ? nlocal1 + nblk : (pid1 == extrablks1) ? nlocal1 + (N % nblk) : nlocal1;
					const int irow = xy_inod[pid * 4 + 0];
					const int icol = xy_inod[pid * 4 + 1];
					const int LOCr = xy_inod[pid * 4 + 2];
					const int LOCc = xy_inod[pid * 4 + 3];
					if (irow != pid1) {
						printf("ERROR: irow(%d) != pid1(%d)\n", irow, pid1); fflush(stdout);
					}
					if (icol != pid2) {
						printf("ERROR: icol(%d) != pid2(%d)\n", icol, pid2); fflush(stdout);
					}
					if (LOCr != LOCr2) {
						printf("ERROR: LOCr(%d) != LOCr2(%d)\n", LOCr, LOCr2); fflush(stdout);
					}
					if (LOCc != LOCc2) {
						printf("ERROR: LOCc(%d) != LOCc2(%d)\n", LOCc, LOCc2); fflush(stdout);
					}
					if (LOCr * LOCc != local_size) {
						printf("ERROR: LOCr*LOCc(%d) != local_size(%d)\n", LOCr * LOCc, local_size); fflush(stdout);
					}
#endif
				}
			}
		} else {
			//printf("CHECK: scatter R\n"); fflush(stdout); fflush(stdout);

			for (int pid1 = 0; pid1 < np_rows; ++pid1) {
				for (int pid2 = 0; pid2 < np_cols; ++pid2) {
					const int pid = pid2 + np_cols * pid1;

					const int local_size = Clip(pid1, pid2, total_grid_size);

					local_size_list[pid] = local_size;
					displs[pid] = total_grid_size;
					total_grid_size += local_size;
				}
			}
		}

		if (total_grid_size != N * N) {
			printf("ERROR: ScatterMatrix2, N^2 != total_grid_size = %d\n", total_grid_size);
		}



		MPI_Scatterv(z_buffers, local_size_list, displs, MPI_DOUBLE_COMPLEX, local_a, local_row_size * local_col_size, MPI_DOUBLE_COMPLEX, 0, mpi_comm);

		//delete[] xy_inod;
		delete[] local_size_list;
		delete[] displs;
		delete[] z_buffers;
	} else {

		MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DOUBLE_COMPLEX, local_a, local_row_size * local_col_size, MPI_DOUBLE_COMPLEX, 0, mpi_comm);

	}


}


/*
* xy_inodには
	my_inod[0] = pos1;
	my_inod[1] = pos2;
	my_inod[2] = local_row_size;
	my_inod[3] = local_col_size;
	をMPI_Gatherした値がrootにだけ入っている
	root以外では自分の値が入っている//
* 
*/
template<class T>
void scalapack_GatherMatrix(int N, 
	int nblk, const int* xy_inod, 
	const T* Z_local, T* Z, const MPI_Comm& mpi_comm, const int root_id, const BlacsGridInfo& blacs_grid) {

	MPI_Datatype datatype = MPI_DOUBLE;
	if constexpr (sizeof(T) == 16) datatype = MPI_DOUBLE_COMPLEX;
	if constexpr (sizeof(T) == 4) datatype = MPI_FLOAT;


	int proc_id, num_procs;
	MPI_Comm_size(mpi_comm, &num_procs);
	MPI_Comm_rank(mpi_comm, &proc_id);

	const int np_rows = blacs_grid.np_rows;
	const int np_cols = blacs_grid.np_cols;
	const int pos1 = blacs_grid.pos1;
	const int pos2 = blacs_grid.pos2;


	const int ZERO = 0;
	const int local_row_size = numroc_(&N, &nblk, &pos1, &ZERO, &np_rows);
	const int local_col_size = numroc_(&N, &nblk, &pos2, &ZERO, &np_cols);


	int total_grid_size = 0;	
	if (proc_id == root_id) {

		int* local_size_list = new int[num_procs];
		int* displs = new int[num_procs];

		for (int pid = 0; pid < num_procs; ++pid) {
			const int LOCr = xy_inod[pid * 4 + 2];
			const int LOCc = xy_inod[pid * 4 + 3];
			local_size_list[pid] = LOCr * LOCc;
			displs[pid] = total_grid_size;
			total_grid_size += LOCr * LOCc;
		}

		T* z_buffers = new T[total_grid_size];
		MPI_Gatherv(Z_local, local_row_size * local_col_size, datatype, z_buffers, local_size_list, displs, datatype, root_id, mpi_comm);


		for (int pid = 0; pid < num_procs; ++pid) {
			const int irow = xy_inod[pid * 4 + 0];
			const int icol = xy_inod[pid * 4 + 1];
			const int LOCr = xy_inod[pid * 4 + 2];
			const int LOCc = xy_inod[pid * 4 + 3];
			const int offset = displs[pid];

			for (int j = 0; j < LOCc; j++) {
				for (int i = 0; i < LOCr; i++) {
					const int ig = np_rows * nblk * ((i) / nblk) + (i) % nblk + ((np_rows + irow) % np_rows) * nblk;
					const int jg = np_cols * nblk * ((j) / nblk) + (j) % nblk + ((np_cols + icol) % np_cols) * nblk;
					Z[ig + N * jg] = z_buffers[offset + j * LOCr + i];
				}
			}

		}

		delete[] local_size_list;
		delete[] displs;
		delete[] z_buffers;
	} else {

		MPI_Gatherv(Z_local, local_row_size * local_col_size, datatype, nullptr, nullptr, nullptr, datatype, root_id, mpi_comm);
	}

}


template<class T>
void scalapack_GatherMatrix2(int N,
	int nblk, const char proc_order,
	const T* Z_local, T* Z, const MPI_Comm& mpi_comm, const int root_id, const BlacsGridInfo& blacs_grid) {

	MPI_Datatype datatype = MPI_DOUBLE;
	if constexpr (sizeof(T) == 16) datatype = MPI_DOUBLE_COMPLEX;
	if constexpr (sizeof(T) == 4) datatype = MPI_FLOAT;


	int proc_id, num_procs;
	MPI_Comm_size(mpi_comm, &num_procs);
	MPI_Comm_rank(mpi_comm, &proc_id);

	const int np_rows = blacs_grid.np_rows;
	const int np_cols = blacs_grid.np_cols;
	const int pos1 = blacs_grid.pos1;
	const int pos2 = blacs_grid.pos2;


	const int ZERO = 0;
	const int local_row_size = numroc_(&N, &nblk, &pos1, &ZERO, &np_rows);
	const int local_col_size = numroc_(&N, &nblk, &pos2, &ZERO, &np_cols);


	int total_grid_size = 0;
	if (proc_id == root_id) {

		int* local_size_list = new int[num_procs];
		int* displs = new int[num_procs];
		T* z_buffers = new T[N*N];

		const int nblocks = N / nblk;
		const int extrablks2 = nblocks % np_cols;
		const int nlocal2 = (nblocks / np_cols) * nblk;
		const int extrablks1 = nblocks % np_rows;
		const int nlocal1 = (nblocks / np_rows) * nblk;


		auto Paste = [&](int pid1, int pid2, const int offset) {
			const int LOCc = (pid2 < extrablks2) ? nlocal2 + nblk : (pid2 == extrablks2) ? nlocal2 + (N % nblk) : nlocal2;
			const int LOCr = (pid1 < extrablks1) ? nlocal1 + nblk : (pid1 == extrablks1) ? nlocal1 + (N % nblk) : nlocal1;

			//const int offset = total_grid_size;

			const int& irow = pid1;
			const int& icol = pid2;


			for (int j = 0; j < LOCc; j++) {
				for (int i = 0; i < LOCr; i++) {
					const int ig = np_rows * nblk * ((i) / nblk) + (i) % nblk + ((np_rows + irow) % np_rows) * nblk;
					const int jg = np_cols * nblk * ((j) / nblk) + (j) % nblk + ((np_cols + icol) % np_cols) * nblk;
					Z[ig + N * jg] = z_buffers[offset + j * LOCr + i];
				}
			}

			return LOCr * LOCc;
		};


		if ((proc_order == 'c') || (proc_order == 'C')) {
			//printf("CHECK: scatter C\n"); fflush(stdout); fflush(stdout);

			for (int pid2 = 0; pid2 < np_cols; ++pid2) {
				for (int pid1 = 0; pid1 < np_rows; ++pid1) {
					const int pid = pid1 + np_rows * pid2;

					const int LOCc = (pid2 < extrablks2) ? nlocal2 + nblk : (pid2 == extrablks2) ? nlocal2 + (N % nblk) : nlocal2;
					const int LOCr = (pid1 < extrablks1) ? nlocal1 + nblk : (pid1 == extrablks1) ? nlocal1 + (N % nblk) : nlocal1;

					const int local_size = LOCc * LOCr;

					local_size_list[pid] = local_size;
					displs[pid] = total_grid_size;
					total_grid_size += local_size;
				}
			}
		} else {

			for (int pid1 = 0; pid1 < np_rows; ++pid1) {
				for (int pid2 = 0; pid2 < np_cols; ++pid2) {
					const int pid = pid2 + np_cols * pid1;

					const int LOCc = (pid2 < extrablks2) ? nlocal2 + nblk : (pid2 == extrablks2) ? nlocal2 + (N % nblk) : nlocal2;
					const int LOCr = (pid1 < extrablks1) ? nlocal1 + nblk : (pid1 == extrablks1) ? nlocal1 + (N % nblk) : nlocal1;

					const int local_size = LOCc * LOCr;

					local_size_list[pid] = local_size;
					displs[pid] = total_grid_size;
					total_grid_size += local_size;
				}
			}
		}

		MPI_Gatherv(Z_local, local_row_size * local_col_size, datatype, z_buffers, local_size_list, displs, datatype, root_id, mpi_comm);

		int offset = 0;
		if ((proc_order == 'c') || (proc_order == 'C')) {
			for (int pid2 = 0; pid2 < np_cols; ++pid2) {
				for (int pid1 = 0; pid1 < np_rows; ++pid1) {
					const int local_size = Paste(pid1, pid2, offset);
					offset += local_size;
				}
			}
		} else {
			for (int pid1 = 0; pid1 < np_rows; ++pid1) {
				for (int pid2 = 0; pid2 < np_cols; ++pid2) {
					const int local_size = Paste(pid1, pid2, offset);
					offset += local_size;
				}
			}
		}


		delete[] local_size_list;
		delete[] displs;
		delete[] z_buffers;
	} else {

		MPI_Gatherv(Z_local, local_row_size * local_col_size, datatype, nullptr, nullptr, nullptr, datatype, root_id, mpi_comm);
	}

}

/*
* NOTE:
* 2026/5/12, PDSYEVDを大きなサイズで100回程度呼び出すと
* comm->shm_numa_layout[my_numa_node].base_addr
* というエラーになることがあった。
* Intel MPIのバグのようで、次の環境へ変数を設定することで回避できた
* export I_MPI_SHM_HEAP_VSIZE=0
* 
* 
* 
* 
* 
*/

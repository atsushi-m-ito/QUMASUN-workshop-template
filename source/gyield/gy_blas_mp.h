#pragma once 
#include <mpi.h>
#include <complex>
#include <cstdint>
#include <type_traits>
#include "gy_blas_mp.h"
#include "gy_memory.h"

//#define DEBUG_PRINT

namespace gy {
	constexpr uint32_t ENTIRE = 0;
	constexpr uint32_t COL_DIVIDED = 0x1;
	constexpr uint32_t ROW_DIVIDED = 0x2;
	constexpr uint32_t FULL_DIVIDED = 0x3;
	
	struct BlasMpInfo {
		MPI_Comm mpi_comm;
		MPI_Comm mpi_comm_col; //縦方向の通信に関する
		MPI_Comm mpi_comm_row; //横方向の通信に関する
		int np_col;  //縦方向の分割数(number of process column)
		int np_row;  //横方向の分割数(number of process row)		
	};

	BlasMpInfo CreateBlasMP(const MPI_Comm& mpi_comm){
		BlasMpInfo blasmp;
		blasmp.mpi_comm = mpi_comm;
		int proc_id;
		MPI_Comm_rank(mpi_comm, &proc_id);
		int num_procs;
		MPI_Comm_size(mpi_comm, &num_procs);
		bool is_root = (proc_id == 0);
		
		//processの2次元分割//
		int np_rows = (int)(sqrt((float)num_procs));
		do {
			if ((num_procs % np_rows) == 0) break;
			np_rows--;
		} while (np_rows >= 2);
		//np_rows = num_procs ;
		int np_cols = num_procs / np_rows;

#ifdef DEBUG_PRINT
		if (is_root) {
			printf("processes are divided into %d x %d\n", np_cols, np_rows);
		}
#endif 

		blasmp.np_col = np_cols;
		blasmp.np_row = np_rows;

		MPI_Comm comm_row;
		MPI_Comm comm_col;
		MPI_Comm_split(mpi_comm, proc_id / np_cols, proc_id, &comm_col);
		MPI_Comm_split(mpi_comm, proc_id % np_cols, proc_id, &comm_row);
		blasmp.mpi_comm_col = comm_col;
		blasmp.mpi_comm_row = comm_row;
		return blasmp;
	}

	void ReleaseBlasMP(BlasMpInfo& blasmp) {
		MPI_Comm_free(&(blasmp.mpi_comm_col));
		MPI_Comm_free(&(blasmp.mpi_comm_row));
	}

	struct BlasMp_Matrix_t {
		uint32_t type;
		int height;
		int width;
		int ld; //leading dimension//
		int ld_w; //leading dimension in row direction//
		int org_height; //of original matrix
		int org_width; //of original matrix//
		int org_ld; //leading dimension of original matrix//

		static BlasMp_Matrix_t MakeOriginalMatrix(int height, int width, int leading_dim) {
			BlasMp_Matrix_t blasmp;
			blasmp.type = ENTIRE;
			blasmp.height = height;
			blasmp.width = width;
			blasmp.ld = leading_dim;
			blasmp.ld_w = width;
			blasmp.org_height = height;
			blasmp.org_width = width;
			blasmp.org_ld = leading_dim;
			return blasmp;
		}
	};

	namespace blasmp {
		namespace impl {

			//MPI_Datatype is estimated from template parameter
			template <class T>
			inline
				MPI_Datatype GetDatatype() {
				if constexpr (sizeof(T) == 16) {
					return MPI_DOUBLE_COMPLEX;
				} else {
					return MPI_DATATYPE_NULL;
				}

			}

			template <>
			inline
				MPI_Datatype GetDatatype<int>() { return MPI_INT; }

			template <>
			inline
				MPI_Datatype GetDatatype<int64_t>() { return MPI_INT64_T; }

			template <>
			inline
				MPI_Datatype GetDatatype<uint64_t>() { return MPI_UINT64_T; }

			template <>
			inline
				MPI_Datatype GetDatatype<double>() { return MPI_DOUBLE; }

			template <>
			inline
				MPI_Datatype GetDatatype<float>() { return MPI_FLOAT; }

			template <>
			inline
				MPI_Datatype GetDatatype<std::complex<double> >() { return MPI_C_DOUBLE_COMPLEX; }

		}
	}

	//C can be set same pointer of A//
	template<class T>
	void blasmp_join_col_direction(BlasMp_Matrix_t typeA, const T* A, BlasMp_Matrix_t* p_typeC, T* C, T* work, BlasMpInfo& blasmp) {
		using namespace blasmp::impl;

		BlasMp_Matrix_t& typeC = *p_typeC;
		typeC.type = typeA.type ^ COL_DIVIDED;
		typeC.height = typeA.org_height;
		typeC.width = typeA.width;
        typeC.ld = typeA.org_height;// typeA.org_ld;
		typeC.ld_w = typeA.ld_w;
		typeC.org_height = typeC.height;
		typeC.org_width = typeA.org_width;
		typeC.org_ld = typeC.ld;

		int proc_id;
		MPI_Comm_rank(blasmp.mpi_comm_col, &proc_id);

#ifdef DEBUG_PRINT
		printf("[%d]blasmp_join_col_direction: %d x %d => %d x %d\n", proc_id, typeA.ld, typeA.ld_w, typeC.ld, typeC.ld_w); fflush(stdout);

#endif

		MPI_Gather(A, typeA.ld * typeA.ld_w, GetDatatype<T>(),
			work, typeA.ld * typeA.ld_w, GetDatatype<T>(), 0, blasmp.mpi_comm_col);
		gy::Synchronize();


#ifdef DEBUG_PRINT
		printf("[%d]afterMPI_Gather: %f\n", proc_id, work[0]); fflush(stdout);

#endif

		if (proc_id == 0) {
#if 1
            
			for (int i = 0; i < typeC.width; ++i) {
				for (int n = 0; n < blasmp.np_col; ++n) {
					int min_height = std::min(typeA.ld * (n + 1), typeC.ld) - typeA.ld * n;
					gy::CopyMemory(C + typeA.ld * n + typeC.ld * i, work + typeA.ld * (i + typeA.ld_w * n), sizeof(T) * min_height);            
				}
			}
            
#else
			gy::CopyMemory(C, work, sizeof(T) * typeC.ld * typeC.width);
#endif
		}
		gy::Synchronize();

#ifdef DEBUG_PRINT
		printf("[%d]beforeMPI_Bcast: %d, %d, %f\n", proc_id, typeC.ld, typeC.ld_w, C[0]); fflush(stdout);

#endif
		MPI_Bcast(C, typeC.ld * typeC.ld_w, GetDatatype<T>(), 0, blasmp.mpi_comm_col);

#ifdef DEBUG_PRINT
        printf("[%d]afterMPI_Bcast: %d, %d, %f\n", proc_id, typeC.ld, typeC.ld_w, C[0]); fflush(stdout);

#endif
	}

	//C can be set same pointer of A//
	template<class T>
	void blasmp_join_row_direction(BlasMp_Matrix_t typeA, const T* A, BlasMp_Matrix_t* p_typeC, T* C, T* work, BlasMpInfo& blasmp) {
		using namespace blasmp::impl;

		BlasMp_Matrix_t& typeC = *p_typeC;
		typeC.type = typeA.type ^ ROW_DIVIDED;
		typeC.height = typeA.height;
		typeC.width = typeA.org_width;
		typeC.ld = typeA.ld;
		typeC.ld_w = typeA.org_width;
		typeC.org_height = typeA.org_height;
		typeC.org_width = typeA.org_width;
		typeC.org_ld = typeA.org_ld;

#ifdef DEBUG_PRINT
		printf("blasmp_join_row_direction:head = %f\n", A[0]);

        if (A == work) {
            printf("blasmp_join_row_direction:error = %s, %d, 0x%zx\n", __FILE__, __LINE__, A); fflush(stdout);
        }
#endif
		MPI_Allgather(A, typeA.ld * typeA.ld_w, GetDatatype<T>(),
			work, typeA.ld * typeA.ld_w , GetDatatype<T>(), blasmp.mpi_comm_row);
		gy::Synchronize();
		gy::CopyMemory(C, work, sizeof(T) * typeC.ld * typeC.width);
	}


    //分割された行列Aを通信して完全な行列Cを生成する//
    //C can be set same pointer of A//
    template<class T>
    void blasmp_join_to_entire(BlasMp_Matrix_t typeA, const T* A, BlasMp_Matrix_t* p_typeC, T* C, T* work, BlasMpInfo& blasmp) {
        if (typeA.type == FULL_DIVIDED) {
            BlasMp_Matrix_t typeB;
            T* B = work;
            auto w2 = work + typeA.ld * typeA.ld_w * blasmp.np_col;
            blasmp_join_col_direction(typeA, A, &typeB, B, w2, blasmp);

            blasmp_join_row_direction(typeB, B, p_typeC, C, w2, blasmp);
        } else if (typeA.type == COL_DIVIDED) {
            blasmp_join_col_direction(typeA, A, p_typeC, C, work, blasmp);
        } else if (typeA.type == ROW_DIVIDED) {
            blasmp_join_row_direction(typeA, A, p_typeC, C, work, blasmp);
        } else {
            *p_typeC = typeA;
            if (C != A) {
                gy::CopyMemory(C, A, sizeof(T) * typeA.ld * typeA.ld_w);
            }
        }
    }


	/*
	* buffer A is shurinked and update.
	*/
	template<class T>
	void blasmp_divide_col_row_direction(BlasMp_Matrix_t typeA, T* A, BlasMp_Matrix_t* p_typeAnew, BlasMpInfo& blasmp) {

		const bool is_able_divided_row = ((typeA.type & ROW_DIVIDED) == 0) && (blasmp.np_row > 1);
		const bool is_able_divided_col = ((typeA.type & COL_DIVIDED) == 0) && (blasmp.np_col > 1);

		if (is_able_divided_row || is_able_divided_col) {
			//col, row 両方向の分割を行う////////////////////////////////
			int proc_id;
			MPI_Comm_rank(blasmp.mpi_comm, &proc_id);
			const int np_col = blasmp.np_col;
			const int np_row = blasmp.np_row;
			const int pid_1 = is_able_divided_col ? (proc_id % np_col) : 0;
			const int pid_2 = is_able_divided_row ? (proc_id / np_col) : 0;

			int ld = is_able_divided_col ? ((typeA.height + np_col - 1) / np_col) : typeA.ld;
			int ld_w = is_able_divided_row ? ((typeA.width + np_row - 1) / np_row) : typeA.ld_w;
			int h_begin = ld * pid_1;
			int h_N = std::min(h_begin + ld, typeA.height) - h_begin;
			int w_begin = ld_w * pid_2;
			int w_N = std::min(w_begin + ld_w, typeA.width) - w_begin;

			BlasMp_Matrix_t& typeC = *p_typeAnew;
			typeC.type = is_able_divided_col ? (is_able_divided_row ? FULL_DIVIDED : COL_DIVIDED) : (is_able_divided_row ? ROW_DIVIDED : ENTIRE);
			typeC.height = h_N;
			typeC.width = w_N;
			typeC.ld = ld;
			typeC.ld_w = ld_w;
			typeC.org_height = typeA.org_height;
			typeC.org_width = typeA.org_width;
			typeC.org_ld = typeA.org_ld;


			if (is_able_divided_col) {
				if ((h_begin != 0) || (w_begin != 0)) {
					const int i = 0;
					gy::CopyMemory(A + ld * i, A + h_begin + (w_begin + i) * typeA.ld, sizeof(T) * h_N);
				}
				for (int i = 1; i < w_N; ++i) {
					gy::CopyMemory(A + ld * i, A + h_begin + (w_begin + i) * typeA.ld, sizeof(T) * h_N);
				}
			}else{
				if (w_begin != 0) {
					gy::CopyMemory(A, A + w_begin * typeA.ld, sizeof(T) * ld * ld_w);
				}
			}
		
		
		} else {
			*p_typeAnew = typeA;
		}
	}

	template<class T>
	void blasmp_t_GEMM(char transA, char transB, T alpha, BlasMp_Matrix_t typeA, const T* A, BlasMp_Matrix_t typeB, const T* B, T beta, BlasMp_Matrix_t* p_typeC, T* C, T* work, BlasMpInfo& blasmp) {
		int proc_id;
		MPI_Comm_rank(blasmp.mpi_comm, &proc_id);
		const int np_col = blasmp.np_col;
		const int np_row = blasmp.np_row;
		const int pid_1 = proc_id % np_col;
		const int pid_2 = proc_id / np_col;

#ifdef DEBUG_PRINT
        T* work0 = work;
#endif

		BlasMp_Matrix_t& typeC = *p_typeC;
		BlasMp_Matrix_t typeCd;
		blasmp_divide_col_row_direction(typeC, C, &typeCd, blasmp);
		typeC = typeCd;
#ifdef DEBUG_PRINT
        printf("typeC.ld=%d, %d, %d, %d\n", typeC.ld, typeC.org_ld, typeC.height, typeC.width); fflush(stdout);
        MPI_Barrier(blasmp.mpi_comm);
#endif
		const int h_begin = typeC.ld * pid_1;
		const int w_begin = typeC.ld_w * pid_2;
		const int h_N = typeC.height;
		const int w_N = typeC.width;

        const T* tgtA = nullptr;
		if(transA=='N'){
			if (typeA.type == ENTIRE) {
				int offsetA = (transA == 'N') ? h_begin : h_begin * typeA.ld;
				tgtA = A + offsetA;
			}else if (typeA.type == COL_DIVIDED) {
				tgtA = A;
			} else if ((typeA.type == ROW_DIVIDED)&&(np_col > 1)) {
				BlasMp_Matrix_t typeAg;
				T* A_fd = work;
                work += typeA.ld * typeA.ld_w;
				gy::CopyMemory(A_fd, A, typeA.ld * typeA.ld_w * sizeof(T));
				blasmp_divide_col_row_direction(typeA, A_fd, &typeAg, blasmp);
				typeA = typeAg;

				T* tA = work;
				work += typeA.ld * typeA.ld_w * np_row;
				blasmp_join_row_direction(typeA, A_fd, &typeAg, tA, work, blasmp);
				typeA = typeAg;
                tgtA = tA;
			} else /*if (typeA.type == FULL_DIVIDED)*/ {
				BlasMp_Matrix_t typeAg;
				T* tA = work;
				work += typeA.ld * typeA.ld_w * np_row;
				blasmp_join_row_direction(typeA, A, &typeAg, tA, work, blasmp);
				typeA = typeAg;
                tgtA = tA;
			}
		} else {
			if (typeA.type == ENTIRE) {
				int offsetA = (transA == 'N') ? h_begin : h_begin * typeA.ld;
				tgtA = A + offsetA;
			}
		}
    
#ifdef DEBUG_PRINT
        printf("typeA.ld=%d, %d, %d, %d\n", typeA.ld, typeA.org_ld, typeA.height, typeA.width); fflush(stdout);
        MPI_Barrier(blasmp.mpi_comm);
#endif
        const T* tgtB = nullptr;
		if (transB == 'N') {
			if (typeB.type == ENTIRE) {
				const int offsetB = (transB == 'N') ? w_begin * typeB.ld : w_begin;
				tgtB = B + offsetB;
			} else if (typeB.type == ROW_DIVIDED) {
				tgtB = B;
			} else if ((typeB.type == COL_DIVIDED) && (np_row>1)) {

                //printf("typeB0.ld=%d, %d, %d, %d\n", typeB.ld, typeB.org_ld, typeB.height, typeB.width); fflush(stdout);
                //MPI_Barrier(blasmp.mpi_comm);

				BlasMp_Matrix_t typeBc;
				T* B_fd = work;
                work += typeB.ld * typeB.ld_w;
				gy::CopyMemory(B_fd, B, typeB.ld * typeB.ld_w * sizeof(T));
				blasmp_divide_col_row_direction(typeB, B_fd, &typeBc, blasmp);
				typeB = typeBc;

                //printf("typeB1.ld=%d, %d, %d, %d\n", typeB.ld, typeB.org_ld, typeB.height, typeB.width); fflush(stdout);
                //MPI_Barrier(blasmp.mpi_comm);


                T* tB = work;
                work += typeB.ld * typeB.ld_w * np_col;
#ifdef DEBUG_PRINT
                printf("used_work_size=%zd\n", (size_t)(work - work0));
#endif
                T* w2 = work;
				blasmp_join_col_direction<T>(typeB, B_fd, &typeBc, tB, w2, blasmp);
				typeB = typeBc;
                tgtB = tB;
			} else /*if (typeB.type == FULL_DIVIDED)*/ {
                
				//join to col 
				BlasMp_Matrix_t typeBc;
				T* tB = work;
                work += typeB.ld* typeB.ld_w* np_col;
                auto w2 = work;
				blasmp_join_col_direction<T>(typeB, B, &typeBc, tB, w2, blasmp);
				typeB = typeBc;
                tgtB = tB;
			}
		}

#ifdef DEBUG_PRINT
        printf("used_work_size=%zd\n", (size_t)(work - work0));
#endif
        //printf("typeB.ld=%d, %d, %d, %d\n", typeB.ld, typeB.org_ld, typeB.height, typeB.width); fflush(stdout);
        MPI_Barrier(blasmp.mpi_comm);



		if constexpr (std::is_same_v<double, T>) {
			blas_DGEMM(transA, transB,
				h_N, w_N, typeA.width, alpha,
				tgtA, typeA.ld,
				tgtB, typeB.ld,
				beta, C, typeC.ld);
		} else if constexpr (sizeof(T) == 16) {
			blas_ZGEMM(transA, transB,
				h_N, w_N, typeA.width, &alpha,
				tgtA, typeA.ld,
				tgtB, typeB.ld,
				&beta, C, typeC.ld);
		}


	}


	void blasmp_DGEMM(char transA, char transB, double alpha, BlasMp_Matrix_t typeA, const double* A, BlasMp_Matrix_t typeB, const double* B, double beta, BlasMp_Matrix_t* p_typeC, double* C, double* work, BlasMpInfo& blasmp) {
		blasmp_t_GEMM<double>(transA, transB, alpha, typeA, A, typeB, B, beta, p_typeC, C, work, blasmp);
	}

	template<class COMPLEX>
	void blasmp_ZGEMM(char transA, char transB, const COMPLEX alpha, BlasMp_Matrix_t typeA, const COMPLEX* A, BlasMp_Matrix_t typeB, const COMPLEX* B,const COMPLEX beta, BlasMp_Matrix_t* p_typeC, COMPLEX* C, COMPLEX* work, BlasMpInfo& blasmp) {
		blasmp_t_GEMM<COMPLEX>(transA, transB, alpha, typeA, A, typeB, B, beta, p_typeC, C, work, blasmp);
	}

}

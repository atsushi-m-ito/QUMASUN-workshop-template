#pragma once
#include "qumasun_base1.h"
#include <cstdio>
#include <string>
#include "cube_writer.h"

inline
void QUMASUN_BASE1::OutputDensity(QUMASUN::OUTPUT_TARGET target, const char* filepath) {
	using namespace QUMASUN;

	if (!is_construction_successful)return;

    if (IsRoot(m_same_ddm_place_comm)) {

        watch.Restart();
        double* outbuf = IsRoot(m_ddm_comm) ? new double[m_size_3d] : nullptr;
        switch (target) {
        case OUTPUT_TARGET::Density:
            mGatherField(outbuf, ml_rho);

            break;
        case OUTPUT_TARGET::DiffDensity:
            if (ml_rho_diff == nullptr) return;
            mGatherField(outbuf, ml_rho_diff);
            break;
        case OUTPUT_TARGET::Vhart:
            mGatherField(outbuf, ml_Vhart);
            break;
            /*
        case OUTPUT_TARGET::DensityHR:
            frame.grid_x *= m_HR_ratio_x;
            frame.grid_y *= m_HR_ratio_y;
            frame.grid_z *= m_HR_ratio_z;
            writer.SaveCube(filepath, frame, m_num_nuclei, m_nuclei, m_hr_rho);
            break;*/
        }


        if (IsRoot(m_ddm_comm)) {
            CubeWriter::Frame frame;
            frame.grid_x = m_size_x;
            frame.grid_y = m_size_y;
            frame.grid_z = m_size_z;
            frame.boxaxis[0] = m_box_x;
            frame.boxaxis[1] = 0.0;
            frame.boxaxis[2] = 0.0;
            frame.boxaxis[3] = 0.0;
            frame.boxaxis[4] = m_box_y;
            frame.boxaxis[5] = 0.0;
            frame.boxaxis[6] = 0.0;
            frame.boxaxis[7] = 0.0;
            frame.boxaxis[8] = m_box_z;
            frame.boxorg[0] = 0.0;
            frame.boxorg[1] = 0.0;
            frame.boxorg[2] = 0.0;

            CubeWriter writer;
            writer.SaveCube(filepath, frame, m_num_nuclei, m_nuclei, outbuf);

            delete[] outbuf;
        }
    }
    watch.Record(44);
	
}

#if 1

inline
void QUMASUN_BASE1::OutputEigenValue(const char* filepath) {
	using namespace QUMASUN;
	if (!is_construction_successful)return;

	FILE* fp = nullptr;
	const bool is_root_global = IsRoot(m_mpi_comm);
	const bool is_root_each_ddm = IsRoot(m_ddm_comm);
	
	if (is_root_each_ddm) {
		double* eigen_buf = nullptr;
		double* occupancy_buf = nullptr;
        if (is_root_global) {
            fp = fopen(filepath, "w");
            eigen_buf = new double[m_total_state * 2];
            occupancy_buf = eigen_buf + m_total_state;
        }
		
		const int num_same_place_procs = GetNumProcess(m_same_ddm_place_comm);
        auto nums_state_in_proc = std::make_unique<int[]>(num_same_place_procs);
        auto offset_list = std::make_unique<int[]>(num_same_place_procs);
        const int TAG = 40000;
		const int TAGO = 50000;
	
		for (int sk = 0; sk < m_all_kinds_spin_kpoint; ++sk) {
            int num_having_states = 0;
            double* target_ptr1 = nullptr;
            double* target_ptr2 = nullptr;
            if ((m_having_spin_kpoint_begin <= sk) && (sk < m_having_spin_kpoint_begin + m_num_having_spin_kpoint)) {
                num_having_states = num_solution;
                const auto& ws = ml_wave_set[sk - m_having_spin_kpoint_begin];
                target_ptr1 = ws.eigen_values;
                target_ptr2 = ws.occupancy;
            }

            MPI_Gather(&num_having_states, 1, MPI_INT, nums_state_in_proc.get(), 1, MPI_INT, 0, m_same_ddm_place_comm);
            if (is_root_global) {
                int count = 0;
                for (int i = 0; i < num_same_place_procs; ++i) {
                    offset_list[i] = count;
                    count += nums_state_in_proc[i];
                }
                //printf("count=%d\n", count); fflush(stdout);
            }
            MPI_Gatherv(target_ptr1, num_having_states, MPI_DOUBLE, eigen_buf, nums_state_in_proc.get(), offset_list.get(), MPI_DOUBLE, 0, m_same_ddm_place_comm);
            MPI_Gatherv(target_ptr2, num_having_states, MPI_DOUBLE, occupancy_buf, nums_state_in_proc.get(), offset_list.get(), MPI_DOUBLE, 0, m_same_ddm_place_comm);
                
            if (is_root_global) {
                int kpoint_spin[4];
                if ((m_having_spin_kpoint_begin <= sk) && (sk < m_having_spin_kpoint_begin + m_num_having_spin_kpoint)) {
                    const auto& ws = ml_wave_set[sk - m_having_spin_kpoint_begin];
                    kpoint_spin[0] = ws.kpoint_x;
                    kpoint_spin[1] = ws.kpoint_y;
                    kpoint_spin[2] = ws.kpoint_z;
                    kpoint_spin[3] = (ws.spin == SPIN::UP) ? 0 : 1;
                
                } else {
                    MPI_Status status;
                    //printf("[%d]Recv\n", GetProcessID(m_same_ddm_place_comm)); fflush(stdout);
                    MPI_Recv(kpoint_spin, 4, MPI_INT, MPI_ANY_SOURCE, TAG + sk, m_same_ddm_place_comm, &status);
                }
                int kx = kpoint_spin[0];
                int ky = kpoint_spin[1];
                int kz = kpoint_spin[2];
                int spin = kpoint_spin[3];

                //printf("Eigen values in kpoint and spin: %d/%d, %d/%d, %d/%d, %s\n", kx, m_kpoint_sampling[0], ky, m_kpoint_sampling[1], kz, m_kpoint_sampling[2], (spin == 0) ? "up" : "down"); fflush(stdout);
                fprintf(fp, "Eigen values in kpoint and spin: %d/%d, %d/%d, %d/%d, %s\n", kx, m_kpoint_sampling[0], ky, m_kpoint_sampling[1], kz, m_kpoint_sampling[2], (spin == 0) ? "up" : "down");
                for (int i = 0; i < m_total_state; ++i) {
                    fprintf(fp, "%d\t%f\t%f\n", i, eigen_buf[i], occupancy_buf[i]);
                }
            } else {//no-root//
				if ((m_having_spin_kpoint_begin <= sk) && (sk < m_having_spin_kpoint_begin + m_num_having_spin_kpoint)) {
                    
                    if (m_begin_state == 0) {
                        const auto& ws = ml_wave_set[sk - m_having_spin_kpoint_begin];
                        int kpoint_spin[4]{ ws.kpoint_x, ws.kpoint_y, ws.kpoint_z, ws.spin == SPIN::UP ? 0 : 1 };
                        //printf("[%d]Send\n", GetProcessID(m_same_ddm_place_comm)); fflush(stdout);
                        MPI_Send(kpoint_spin, 4, MPI_INT, 0, TAG + sk, m_same_ddm_place_comm);
                    }
				}
			}
		}
	
		if (is_root_global) {
			fclose(fp);
			delete[] eigen_buf;
		}
	}
}

#else
inline
void QUMASUN_BASE1::OutputEigenValue(const char* filepath) {
	using namespace QUMASUN;
	if (!is_construction_successful)return;

	FILE* fp = nullptr;
	const bool is_root_global = IsRoot(m_mpi_comm);
	const bool is_root_each_ddm = IsRoot(m_ddm_comm);
	
	if (is_root_each_ddm) {
		double* eigen_buf = nullptr;
		double* occupancy_buf = nullptr;
		if (is_root_global) {
			fp = fopen(filepath, "w");
			eigen_buf = new double[num_solution*2];
			occupancy_buf = eigen_buf + num_solution;
		}
		const int num_same_place_procs = GetNumProcess(m_same_ddm_place_comm);
		

		const int TAG = 40000;
		const int TAGO = 50000;
	

        std::vector<Kpoint3D> kpoint_list;
        int all_kinds_kpoint = ListupKpoints(m_kpoint_sampling[0], m_kpoint_sampling[1], m_kpoint_sampling[2], kpoint_symmetry, kpoint_list);



		for (int sk = 0; sk < m_all_kinds_spin_kpoint; ++sk) {
            if (is_root_global) {
                if (all_kinds_kpoint > sk) {
                    fprintf(fp, "Eigen values in kpoint and spin: %d/%d, %d/%d, %d/%d, %s\n", kpoint_list[sk].kx, m_kpoint_sampling[0], kpoint_list[sk].ky, m_kpoint_sampling[1], kpoint_list[sk].kz, m_kpoint_sampling[2], "up");
                } else {
                    fprintf(fp, "Eigen values in kpoint and spin: %d/%d, %d/%d, %d/%d, %s\n", kpoint_list[sk - all_kinds_kpoint].kx, m_kpoint_sampling[0], kpoint_list[sk - all_kinds_kpoint].ky, m_kpoint_sampling[1], kpoint_list[sk - all_kinds_kpoint].kz, m_kpoint_sampling[2], "down");
                }
            }

            std::vector<int> num_list(num_same_place_procs);
            bool has_kpoint = false;
            if ((m_having_spin_kpoint_begin <= sk) && (sk < m_having_spin_kpoint_begin + m_num_having_spin_kpoint)) {
                has_kpoint = true;
            }

            int num = has_kpoint ? num_solution : 0;

            MPI_Gather(&num, 1, MPI_INT, &num_list[0], num_same_place_procs, MPI_INT,0, m_same_ddm_place_comm);

            std::vector<int> heads(num_same_place_procs + 1);
            heads[0] = 0;
            for (int p = 0; p < m_num_procs_sd; ++p) {
                heads[p + 1] = heads[p] + num_list[p];
            }
            int total = heads[m_num_procs_sd];

            std::vector<double> occ_all(m_total_state);
            std::vector<double> eigen_all(m_total_state);
            double* ep = (has_kpoint) ? ml_wave_set[sk - m_having_spin_kpoint_begin].eigen_values : nullptr;
            double* eo = (has_kpoint) ? ml_wave_set[sk - m_having_spin_kpoint_begin].occupancy : nullptr;
                
            MPI_Gatherv(ep, num_solution, MPI_DOUBLE, &eigen_all[0], &num_list[0], &heads[0], MPI_DOUBLE, 0, m_same_ddm_place_comm);
            MPI_Gatherv(eo, num_solution, MPI_DOUBLE, &occ_all[0], &num_list[0], &heads[0], MPI_DOUBLE, 0, m_same_ddm_place_comm);
        
            if (is_root_global) {
                for (int i = 0; i < m_total_state; ++i) {
                    fprintf(fp, "%d\t%f\t%f\n", i, eigen_all[i], occ_all[i]);
                }
            }
		}
	
		if (is_root_global) {
			fclose(fp);
			delete[] eigen_buf;
		}
	}
}
#endif


/*
* mode:
*   0: output state vector (complex)
*   1: output orbital density that is |psi_i(x)|^2
*/
inline
void QUMASUN_BASE1::OutputEigenVector(int mode, const char* name_header, const char* list_path, const char* dir_path = nullptr) {
    using namespace QUMASUN;
    if (!is_construction_successful)return;

    watch.Restart();
    
    const bool is_root_global = IsRoot(m_mpi_comm);
    const bool is_root_each_ddm = IsRoot(m_ddm_comm);


    double* state_vector_re = nullptr;
    double* state_vector_im = nullptr;
    const size_t local_size = ml_grid.Size3D();
    double* l_p_rho = nullptr;
    if (mode == 1) {
        l_p_rho = new double[local_size];
    }


    if (is_root_each_ddm) {
        state_vector_re = new double[m_size_3d * 2];
        state_vector_im = state_vector_re + m_size_3d;
    }
    const int num_same_place_procs = GetNumProcess(m_same_ddm_place_comm);

    const int TAG = 40000;
    const int TAGO = 50000;

    const int all_kpoints = is_spin_on ? m_all_kinds_spin_kpoint / 2 : m_all_kinds_spin_kpoint;
    auto FileName = [&name_header, &all_kpoints](int sk, int n, const char* tail) {
        std::string filename(name_header);
        filename += (sk < all_kpoints) ? "_up" : "_down";
        filename += "_k" + std::to_string(sk % all_kpoints) + "_n" + std::to_string(n) + tail + ".cube";
        return filename;
        };

    auto FilePath = [&dir_path,&FileName](int sk, int n, const char* tail) {
        if (dir_path) return std::string(dir_path) + FileName(sk, n, tail);
        return FileName(sk, n, tail);
        };

    if (mode == 0) {
        if (is_root_global) {
            std::string path(list_path);
            if (dir_path) path = dir_path + path;
            FILE* fp = fopen(path.c_str(), "w");
            for (int sk = 0; sk < m_all_kinds_spin_kpoint; ++sk) {
                for (int n = 0; n < m_total_state; ++n) {
                    fprintf(fp, "%s\n", FileName(sk, n, "_re").c_str());
                    fprintf(fp, "%s\n", FileName(sk, n, "_im").c_str());
                }
            }
            fclose(fp);
        }
    }

    watch.Record(42);

    for (int sk = 0; sk < m_all_kinds_spin_kpoint; ++sk) {


        if ((m_having_spin_kpoint_begin <= sk) && (sk < m_having_spin_kpoint_begin + m_num_having_spin_kpoint)) {
            const auto& ws = ml_wave_set[sk - m_having_spin_kpoint_begin];



            for (int n = 0; n < num_solution; ++n) {

                if (mode == 1) {
                    const double* l_re = ws.l_psi_set[n].re;
                    const double* l_im = ws.l_psi_set[n].im;
                    for (int i = 0; i < local_size; ++i) {
                        l_p_rho[i] = l_re[i] * l_re[i] + l_im[i] * l_im[i];
                    }
                    mGatherField(state_vector_re, l_p_rho);
                } else { //for state //
                    mGatherField(state_vector_re, ws.l_psi_set[n].re);
                    mGatherField(state_vector_im, ws.l_psi_set[n].im);
                }
                watch.Record(43);

                if (is_root_each_ddm) {

                    CubeWriter::Frame frame;
                    frame.grid_x = m_size_x;
                    frame.grid_y = m_size_y;
                    frame.grid_z = m_size_z;
                    frame.boxaxis[0] = m_box_x;
                    frame.boxaxis[1] = 0.0;
                    frame.boxaxis[2] = 0.0;
                    frame.boxaxis[3] = 0.0;
                    frame.boxaxis[4] = m_box_y;
                    frame.boxaxis[5] = 0.0;
                    frame.boxaxis[6] = 0.0;
                    frame.boxaxis[7] = 0.0;
                    frame.boxaxis[8] = m_box_z;
                    frame.boxorg[0] = 0.0;
                    frame.boxorg[1] = 0.0;
                    frame.boxorg[2] = 0.0;
                    sprintf(frame.comment, "eigenvalue=%.10f, kpoint=%d/%d,%d/%d,%d/%d", ws.eigen_values[n], ws.kpoint_x, m_kpoint_sampling[0], ws.kpoint_y, m_kpoint_sampling[1], ws.kpoint_z, m_kpoint_sampling[2]);

                    if (mode == 1) {
                        CubeWriter writer;
                        auto filepath_rho = FilePath(sk, n + m_begin_state, "");
                        printf("save %s\n", filepath_rho.c_str()); fflush(stdout);
                        writer.SaveCube(filepath_rho.c_str(), frame, m_num_nuclei, m_nuclei, state_vector_re);

                    } else {
                        CubeWriter writer;
                        auto filepath_re = FilePath(sk, n + m_begin_state, "_re");
                        auto filepath_im = FilePath(sk, n + m_begin_state, "_im");
                        printf("save %s\n", filepath_re.c_str()); fflush(stdout);
                        writer.SaveCube(filepath_re.c_str(), frame, m_num_nuclei, m_nuclei, state_vector_re);
                        printf("save %s\n", filepath_im.c_str()); fflush(stdout);
                        writer.SaveCube(filepath_im.c_str(), frame, m_num_nuclei, m_nuclei, state_vector_im);
                    }

                    watch.Record(44);
                }
            }

        }

    }


    if (is_root_each_ddm) {
        delete[] state_vector_re;
    }
    delete[] l_p_rho;


}

/*
* mode:
*   0: output state vector (complex)
*   1: output orbital density that is |psi_i(x)|^2
*/
inline
void QUMASUN_BASE1::OutputEigenVectorList(int mode, const char* filepath_head, std::set<int>& target_states, const char* filepath_tail) {
    using namespace QUMASUN;
    if (!is_construction_successful)return;
    if (target_states.empty()) return;

    watch.Restart();

    const bool is_root_global = IsRoot(m_mpi_comm);
    const bool is_root_each_ddm = IsRoot(m_ddm_comm);

    double* state_vector_re = nullptr;
    double* state_vector_im = nullptr;
    const size_t local_size = ml_grid.Size3D();
    double* l_p_rho = nullptr;
    if (mode == 1) {
        l_p_rho = new double[local_size];        
    }

    if (is_root_each_ddm) {
        state_vector_re = new double[m_size_3d * 2];
        state_vector_im = state_vector_re + m_size_3d;
    }
    const int num_same_place_procs = GetNumProcess(m_same_ddm_place_comm);

    const int TAG = 40000;
    const int TAGO = 50000;

    const int all_kpoints = is_spin_on ? m_all_kinds_spin_kpoint / 2 : m_all_kinds_spin_kpoint;
    auto FileName = [&filepath_head, &all_kpoints](int sk, int n, const char* tail) {
        std::string filename(filepath_head);
        filename += (sk < all_kpoints) ? "_up" : "_down";
        filename += "_k" + std::to_string(sk % all_kpoints) + "_n" + std::to_string(n) + tail + ".cube";
        return filename;
        };

    //auto it_target = target_states.begin();
    
    for (int sk = 0; sk < m_all_kinds_spin_kpoint; ++sk) {


        if ((m_having_spin_kpoint_begin <= sk) && (sk < m_having_spin_kpoint_begin + m_num_having_spin_kpoint)) {
            const auto& ws = ml_wave_set[sk - m_having_spin_kpoint_begin];


            for (int n = 0; n < num_solution; ++n) {
                auto it = target_states.find(n + m_begin_state);
                if (it == target_states.end()) continue;

                if (mode == 1) {
                    const double* l_re = ws.l_psi_set[n].re;
                    const double* l_im = ws.l_psi_set[n].im;
                    for (int i = 0; i < local_size; ++i) {
                        l_p_rho[i] = l_re[i] * l_re[i] + l_im[i] * l_im[i];
                    }
                    mGatherField(state_vector_re, l_p_rho);
                } else { //for state //
                    mGatherField(state_vector_re, ws.l_psi_set[n].re);
                    mGatherField(state_vector_im, ws.l_psi_set[n].im);
                    watch.Record(43);
                }
                if (is_root_each_ddm) {

                    CubeWriter::Frame frame;
                    frame.grid_x = m_size_x;
                    frame.grid_y = m_size_y;
                    frame.grid_z = m_size_z;
                    frame.boxaxis[0] = m_box_x;
                    frame.boxaxis[1] = 0.0;
                    frame.boxaxis[2] = 0.0;
                    frame.boxaxis[3] = 0.0;
                    frame.boxaxis[4] = m_box_y;
                    frame.boxaxis[5] = 0.0;
                    frame.boxaxis[6] = 0.0;
                    frame.boxaxis[7] = 0.0;
                    frame.boxaxis[8] = m_box_z;
                    frame.boxorg[0] = 0.0;
                    frame.boxorg[1] = 0.0;
                    frame.boxorg[2] = 0.0;
                    sprintf(frame.comment, "eigenvalue=%.10f, kpoint=%d/%d,%d/%d,%d/%d", ws.eigen_values[n], ws.kpoint_x, m_kpoint_sampling[0], ws.kpoint_y, m_kpoint_sampling[1], ws.kpoint_z, m_kpoint_sampling[2]);

                    if (mode == 1) {
                        CubeWriter writer;
                        auto filepath_rho = FileName(sk, n + m_begin_state, (std::string("_t") + filepath_tail).c_str());
                        printf("save %s\n", filepath_rho.c_str()); fflush(stdout);
                        writer.SaveCube(filepath_rho.c_str(), frame, m_num_nuclei, m_nuclei, state_vector_re);

                    } else {//for state//
                        CubeWriter writer;
                        auto filepath_re = FileName(sk, n + m_begin_state, (std::string("_re_t") + filepath_tail).c_str());
                        auto filepath_im = FileName(sk, n + m_begin_state, (std::string("_im_t") + filepath_tail).c_str());
                        printf("save %s\n", filepath_re.c_str()); fflush(stdout);
                        writer.SaveCube(filepath_re.c_str(), frame, m_num_nuclei, m_nuclei, state_vector_re);
                        printf("save %s\n", filepath_im.c_str()); fflush(stdout);
                        writer.SaveCube(filepath_im.c_str(), frame, m_num_nuclei, m_nuclei, state_vector_im);
                    }

                    watch.Record(44);
                }
            }

        }

    }


    if (is_root_each_ddm) {
        delete[] state_vector_re;
    }
    delete[] l_p_rho;

}

void QUMASUN_BASE1::OutputOverlapMatrix(const char* filepath_head, const char* filepath_tail) {

    const int local_size = ml_grid.Size3D();
    const int N = num_solution;
    const int n2 = N * 2;
    double* Smat = m_work;
    double* Smat_red = m_work + N * N *2;
    double* temp_mat_d_2n2n = m_work + 4 * N * N;

    const double dV = m_dx * m_dy * m_dz;


    const int all_kpoints = is_spin_on ? m_all_kinds_spin_kpoint / 2 : m_all_kinds_spin_kpoint;
    auto NameSpinK = [&all_kpoints](int sk) {
        std::string filename((sk < all_kpoints) ? "_up" : "_down");
        filename += "_k" + std::to_string(sk % all_kpoints);
        return filename;
        };

    for (int sk = 0; sk < m_all_kinds_spin_kpoint; ++sk) {


        if ((m_having_spin_kpoint_begin <= sk) && (sk < m_having_spin_kpoint_begin + m_num_having_spin_kpoint)) {
            const auto& ws = ml_wave_set[sk - m_having_spin_kpoint_begin];



            double* l_re = ws.l_psi_set[0].re;
            blas_DGEMM_t(n2, n2, local_size, l_re, l_re, temp_mat_d_2n2n, n2, dV, 0.0);
            AoSComplexFrom2by2Real(num_solution, Smat, N, temp_mat_d_2n2n, n2);

            MPI_Reduce(Smat, Smat_red, N*N* 2, MPI_DOUBLE, MPI_SUM, 0, m_ddm_comm);

            if (IsRoot(m_ddm_comm)) {
                {
                    std::string filepath_re(std::string(filepath_head) + NameSpinK(sk) + "_re" + filepath_tail);
                    FILE* fp = fopen(filepath_re.c_str(), "w");
                    for (int j = 0; j < N; j++) {
                        for (int i = 0; i < N - 1; i++) {
                            fprintf(fp, "%f\t", Smat_red[(j + N * i) * 2]);
                        }
                        fprintf(fp, "%f\n", Smat_red[(j + N * (N - 1)) * 2]);
                    }
                    fclose(fp);
                }
                {
                    std::string filepath_re(std::string(filepath_head) + NameSpinK(sk) + "_im" + filepath_tail);
                    FILE* fp = fopen(filepath_re.c_str(), "w");
                    for (int j = 0; j < N; j++) {
                        for (int i = 0; i < N - 1; i++) {
                            fprintf(fp, "%f\t", Smat_red[(j + N * i) * 2 + 1]);
                        }
                        fprintf(fp, "%f\n", Smat_red[(j + N * (N - 1)) * 2 + 1]);
                    }
                    fclose(fp);
                }

            }
        }
    }

}


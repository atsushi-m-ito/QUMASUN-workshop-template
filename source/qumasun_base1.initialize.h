#pragma once
#ifdef USE_MPI
#include "mpi_helper.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include "qumasun_base1.h"
#include "vps_loader.h"
#include "pao_loader.h"
#include "cube_reader3.h"
#ifdef INIT_GRAM_SCHMIDT 
#include "GramSchmidt_mpi.h"
#endif
#include "GridFor.h"
#include "velocity_for_wave.h"
#include "shift_via_fft.h"
#include "convertHR.h"
#include "fftw_executor.h"
#include "grid_atomic_mask.h"
#include "cube_writer.h"



inline
void QUMASUN_BASE1::mInitializeState() {

    if (m_initial_state_files.empty()) {
        //test make mask////////////////////////////////////////////////
        double* mask_buf = nullptr;
        if (GetProcessID(m_ddm_comm) == 0) {
            mask_buf = new double[m_size_3d];
            memset(mask_buf, 0, sizeof(double) * m_size_3d);
            const double mother = 2.0 * 2.0;
            const double cutoff = 3.0;
            const double cutoff2 = cutoff * cutoff;
            MakeAtomicMask(m_size_x, m_size_y, m_size_z, m_dx, m_dy, m_dz,
                m_nuclei, m_num_nuclei, cutoff, [&](int64_t i, double r2) {
                    double val = (r2 < cutoff2) ? 1.0 : exp(-(r2 - cutoff2) / mother);
                    mask_buf[i] = std::max(mask_buf[i], val);
                });

            if (GetProcessID(m_mpi_comm) == 0) {
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
                writer.SaveCube("test_mask.cube", frame, m_num_nuclei, m_nuclei, mask_buf);
            }
        }
        //////////////////////////////////////////////////////////////////
        /*
        if (m_initial_state_mode == 2) {
            mInitializeStateWave( mask_buf);//表面系でロバストに動かないのでお蔵入り
        } else {
        */
            mInitializeStateRandom(nullptr);//mask_buf
        //}
        if (mask_buf)delete[] mask_buf;
    } else {
        mInitializeStateFile();
    }
}


/*
* mode == 0: random
* mode == 1: plane wave
* 
*/
inline
void QUMASUN_BASE1::mInitializeStateWave(const double* mask) {


    {
        //random state
        const int proc_id = GetProcessID(m_ddm_comm);
        const bool is_root_ddm = (proc_id == 0);
        const double dVol = m_dx * m_dy * m_dz;

        const double phase = 2.0 * M_PI * 0.0;// 0.0625;  //random phase rotation//
        const double cos_p = cos(phase);
        const double sin_p = sin(phase);
        const int local_size = ml_grid.Size3D();

        if (is_root_ddm) {

            double* psi_r = gy::AlignedAlloc<double>(m_size_3d);
            double* psi_i = gy::AlignedAlloc<double>(m_size_3d);

            const uint32_t seed = 123456789 + m_begin_state;
            std::mt19937 mt(seed);
            std::uniform_real_distribution<> distribution(-1.0, 1.0);

            int64_t iz_begin = 0;
            int64_t iz_end = m_size_z;
            if (mask == nullptr) {
                iz_begin = m_size_z;
                iz_end = 0;
                for (int ni = 0; ni < m_num_nuclei; ++ni) {
                    int64_t iz = (int64_t)(m_nuclei[ni].Rz / m_dz);
                    if (iz_begin > iz) iz_begin = iz;
                    if (iz_end < iz) iz_end = iz;
                }
                iz_begin = std::max<int64_t>(iz_begin - 5, 0);
                iz_end = std::min<int64_t>(iz_end + 5, m_size_z);
                printf("InitOrbital:z-range = %zd, %zd\n", iz_begin, iz_end);
            }

#if 1
            int maxX = (m_size_x - 1) / 2;
            int maxY = (m_size_y - 1) / 2;
            int maxZ = (m_size_z - 1) / 2;
#else
            //note: maxX is maximum  wave number in x direction, k_x.
            //ratio: c,
            // estX = c * (m_size_x - 1)/2 ;
            // estY = c * (m_size_y - 1)/2 ;
            // estZ = c * (m_size_z - 1)/2 ;
            // condition: estX * estY * estZ * 4(pi/3~1) > num_solution 
            // -> c^3 = 2 * num_solution / ((m_size_x - 1) * (m_size_y - 1) * (m_size_z - 1))
            double c = std::cbrt((double)(2 * num_solution) / (double)((m_size_x - 1) * (m_size_y - 1) * (m_size_z - 1)));
            int maxX = std::ceil(c * ((m_size_x - 1) / 2));
            int maxY = std::ceil(c * ((m_size_y - 1) / 2));
            int maxZ = std::ceil(c * ((m_size_z - 1) / 2));
#endif
            int wave_all_count = (maxX * 2 + 1) * (maxY * 2 + 1) * (maxZ * 2 + 1);


            struct WaveNumber {
                int kx; int ky; int kz; double Ek;
                bool operator<(const WaveNumber& b) {
                    return this->Ek < b.Ek;
                }
            };

            auto SQ = [](double a) {return a * a; };
            //方向毎のminimumなwaveのエネルギーを求めて、それを全体の下限とする
            const double max_wave_length = std::min(std::min(m_box_x, m_box_y), m_box_z);

            std::vector<WaveNumber> wave_list;
            wave_list.push_back(WaveNumber{ 0,0,0,0.0 });
            for (int bz = 1; bz <= maxZ; ++bz) {
                for (int by = 1; by <= maxY; ++by) {
                    for (int bx = 1; bx <= maxX; ++bx) {
                        const int kx = floor(m_box_x / (max_wave_length / (double)bx) + 0.5);
                        const int ky = floor(m_box_y / (max_wave_length / (double)by) + 0.5);
                        const int kz = floor(m_box_z / (max_wave_length / (double)bz) + 0.5);
                        double Ek_x = SQ(M_PI * ((double)kx / m_box_x)) / 2.0;
                        double Ek_y = SQ(M_PI * ((double)ky / m_box_y)) / 2.0;
                        double Ek_z = SQ(M_PI * ((double)kz / m_box_z)) / 2.0;                        
                        wave_list.push_back(WaveNumber{ kx,ky,kz, Ek_x + Ek_y + Ek_z });
                        wave_list.push_back(WaveNumber{ -kx,ky,kz, Ek_x + Ek_y + Ek_z });
                        wave_list.push_back(WaveNumber{ kx,-ky,kz, Ek_x + Ek_y + Ek_z });
                        wave_list.push_back(WaveNumber{ -kx,-ky,kz, Ek_x + Ek_y + Ek_z });
                        wave_list.push_back(WaveNumber{ kx,ky,-kz, Ek_x + Ek_y + Ek_z });
                        wave_list.push_back(WaveNumber{ -kx,ky,-kz, Ek_x + Ek_y + Ek_z });
                        wave_list.push_back(WaveNumber{ kx,-ky,-kz, Ek_x + Ek_y + Ek_z });
                        wave_list.push_back(WaveNumber{ -kx,-ky,-kz, Ek_x + Ek_y + Ek_z });
                    }
                }
            }
            std::sort(wave_list.begin(), wave_list.end());

            int64_t i_wave_begin = 0;
            /*
            int64_t i_wave_end = wave_list.size();
            for (; i_wave_begin < i_wave_end; ++i_wave_begin) {
                if (wave_list[i_wave_begin].Ek > Emin_limit) {
                    break;
                }
            }
            if (i_wave_end - i_wave_begin < num_solution) {
                i_wave_begin = i_wave_end - num_solution;
            }*/

            auto SetWave = [&](double* psi_r, int kx, int ky, int kz) {
                for (size_t iz = 0; iz < m_size_z; ++iz) {
                    double th_z = 2.0 * M_PI * (double)kz * (double)iz / (double)m_size_z;
                    double Fz = (kz < 0) ? sin(th_z) : cos(th_z);
                    for (size_t iy = 0; iy < m_size_y; ++iy) {
                        double th_y = 2.0 * M_PI * (double)ky * (double)iy / (double)m_size_y;
                        double Fy = (ky < 0) ? sin(th_y) : cos(th_y);
                        for (size_t ix = 0; ix < m_size_x; ++ix) {
                            double th_x = 2.0 * M_PI * (double)kx * (double)ix / (double)m_size_x;
                            double Fx = (kx < 0) ? sin(th_x) : cos(th_x);
                            const int64_t i = ix + m_size_x * (iy + m_size_y * iz);
                            psi_r[i] = Fx * Fy * Fz;
                        }
                    }
                }
                };

            for (int sk = 0; sk < m_num_having_spin_kpoint; ++sk) {
                double val = 1.0;

                for (int n = 0; n < num_solution; ++n) {
                    auto& wave = wave_list[n + i_wave_begin];
                    const int kz = wave.kz;
                    const int ky = wave.ky;
                    const int kx = wave.kx;
                    SetWave(psi_r, kx, ky, kz);

                    if (mask) {
                        //マスクをかける//
                        for (size_t i = 0; i < m_size_3d; ++i) {
                            psi_r[i] *= mask[i];
                        }
                    }
                    
                    memset(psi_i, 0, sizeof(double) * m_size_3d);
                    /*
                    for (size_t i = 0; i < m_size_3d; ++i) {
                        size_t ix = i % m_size_x;
                        double angle = 2.0 * M_PI * (double)ml_wave_set[sk].kpoint_x * m_dkx * ((double)ix * m_dx);

                        psi_i[i] = sin(-angle) * psi_r[i];
                        psi_r[i] *= cos(-angle);

                    }
                    */
                    //SoAC::SetZero(ml_wave_set[sk].l_psi_set[n], ml_grid.Size3D());
                    mScatterField(ml_wave_set[sk].l_psi_set[n].re, psi_r);
                    mScatterField(ml_wave_set[sk].l_psi_set[n].im, psi_i);
                    /*
                    for (size_t i = 0; i < local_size; ++i) {
                        size_t ix = i % m_size_x;
                        double angle = 2.0 * M_PI * (double)ml_wave_set[sk].kpoint_x * m_dkx * ((double)ix*m_dx);
                        ml_wave_set[sk].l_psi_set[n].im[i] = (-angle) * ml_wave_set[sk].l_psi_set[n].re[i];
                        ml_wave_set[sk].l_psi_set[n].re[i] *= cos_p;
                    }
                    */

                    val += fabs(distribution(mt));
                    ml_wave_set[sk].eigen_values[n] = -1.0 / val;

                }

                MPI_Bcast(ml_wave_set[sk].eigen_values, num_solution, MPI_DOUBLE, 0, ml_grid.mpi_comm);

#ifdef INIT_GRAM_SCHMIDT
                GramSchmidt_ddm(ml_grid, num_solution, ml_wave_set[sk].l_psi_set, dVol);
#endif
            }

            gy::AlignedFree(psi_r);
            gy::AlignedFree(psi_i);

        } else {
            //slave process//

            for (int sk = 0; sk < m_num_having_spin_kpoint; ++sk) {
                for (int n = 0; n < num_solution; ++n) {
                    //SoAC::SetZero(ml_wave_set[sk].l_psi_set[n], ml_grid.Size3D());
                    mScatterField(ml_wave_set[sk].l_psi_set[n].re, nullptr);
                    mScatterField(ml_wave_set[sk].l_psi_set[n].im, nullptr);
                    /*
                    for (size_t i = 0; i < local_size; ++i) {
                        ml_wave_set[sk].l_psi_set[n].im[i] = sin_p * ml_wave_set[sk].l_psi_set[n].re[i];
                        ml_wave_set[sk].l_psi_set[n].re[i] *= cos_p;
                    }*/

                }

                MPI_Bcast(ml_wave_set[sk].eigen_values, num_solution, MPI_DOUBLE, 0, ml_grid.mpi_comm);

#ifdef INIT_GRAM_SCHMIDT
                GramSchmidt_ddm(ml_grid, num_solution, ml_wave_set[sk].l_psi_set, dVol);
#endif
            }
        }


    }

}

inline
void QUMASUN_BASE1::mInitializeStateRandom(const double* mask) {


	{
		//random state
		const int proc_id = GetProcessID(m_ddm_comm);
		const bool is_root_ddm = (proc_id == 0);
		const double dVol = m_dx * m_dy * m_dz;

		const double phase = 2.0 * M_PI * 0.0;// 0.0625;  //random phase rotation//
		const double cos_p = cos(phase);
		const double sin_p = sin(phase);
		const int local_size = ml_grid.Size3D();

		if (is_root_ddm) {

			double* psi_r = gy::AlignedAlloc<double>(m_size_3d);
			double* psi_i = gy::AlignedAlloc<double>(m_size_3d);

			const uint32_t seed = 123456789 + m_begin_state;
			//std::mt19937 mt(seed);
			std::uniform_real_distribution<> distribution(-1.0, 1.0);

            int64_t iz_begin = 0;
            int64_t iz_end = m_size_z;
            if (mask == nullptr) {
                //when mask is null, artificially filtrate in z-direction//
                iz_begin = m_size_z;
                iz_end = 0;
                for (int ni = 0; ni < m_num_nuclei; ++ni) {
                    int64_t iz = (int64_t)(m_nuclei[ni].Rz / m_dz);
                    if (iz_begin > iz) iz_begin = iz;
                    if (iz_end < iz) iz_end = iz;
                }
                iz_begin = std::max<int64_t>(iz_begin - 5, 0);
                iz_end = std::min<int64_t>(iz_end + 5, m_size_z);
                printf("InitOrbital:z-range = %zd, %zd\n", iz_begin, iz_end);
            }

			for (int sk = 0; sk < m_num_having_spin_kpoint ; ++sk) {

                std::mt19937 mt(seed);
				double val = 1.0;

                for (int n = 0; n < num_solution; ++n) {
                    for (size_t iz = 0; iz < iz_begin; ++iz) {
                        for (size_t iy = 0; iy < m_size_y; ++iy) {
                            for (size_t ix = 0; ix < m_size_x; ++ix) {
                                const int64_t i = ix + m_size_x * (iy + m_size_y * iz);
                                psi_r[i] = 0.0;
                            }
                        }
                    }
                    for (size_t iz = iz_begin; iz < iz_end; ++iz) {
                        for (size_t iy = 0; iy < m_size_y; ++iy) {
                            for (size_t ix = 0; ix < m_size_x; ++ix) {
                                const int64_t i = ix + m_size_x * (iy + m_size_y * iz);
                                psi_r[i] = distribution(mt);
                            }
                        }
                    }
                    for (size_t iz = iz_end; iz < m_size_z; ++iz) {
                        for (size_t iy = 0; iy < m_size_y; ++iy) {
                            for (size_t ix = 0; ix < m_size_x; ++ix) {
                                const int64_t i = ix + m_size_x * (iy + m_size_y * iz);
                                psi_r[i] = 0.0;
                            }
                        }
                    }

                    if (mask) {
                        //マスクをかける//
                        for (size_t i = 0; i < m_size_3d; ++i) {
                            psi_r[i] *= mask[i];
                        }
                    }
#if 0

                    for (size_t i = 0; i < m_size_3d; ++i) {
                        psi_i[i] = 0.0;
                    }
#else
					for (size_t i = 0; i < m_size_3d; ++i) {
						size_t ix = i % m_size_x;
						double angle = 2.0 * M_PI * (double)ml_wave_set[sk].kpoint_x * m_dkx * ((double)ix * m_dx);

						psi_i[i] = sin( - angle)* psi_r[i];
						psi_r[i] *= cos( - angle);

                    }
#endif
		
                    mScatterField(ml_wave_set[sk].l_psi_set[n].re, psi_r);
					mScatterField(ml_wave_set[sk].l_psi_set[n].im, psi_i);
					

					val += fabs(distribution(mt));
					ml_wave_set[sk].eigen_values[n] = -1.0 / val;
					
				}

				MPI_Bcast(ml_wave_set[sk].eigen_values, num_solution, MPI_DOUBLE, 0, ml_grid.mpi_comm);
				
#ifdef INIT_GRAM_SCHMIDT
				GramSchmidt_ddm(ml_grid, num_solution, ml_wave_set[sk].l_psi_set, dVol);
#endif
			}

			gy::AlignedFree(psi_r);
            gy::AlignedFree(psi_i);

		} else {
			//slave process//

			for (int sk = 0; sk < m_num_having_spin_kpoint; ++sk) {
				for (int n = 0; n < num_solution; ++n) {
					//SoAC::SetZero(ml_wave_set[sk].l_psi_set[n], ml_grid.Size3D());
					mScatterField(ml_wave_set[sk].l_psi_set[n].re, nullptr);
					mScatterField(ml_wave_set[sk].l_psi_set[n].im, nullptr);
					/*
					for (size_t i = 0; i < local_size; ++i) {
						ml_wave_set[sk].l_psi_set[n].im[i] = sin_p * ml_wave_set[sk].l_psi_set[n].re[i];
						ml_wave_set[sk].l_psi_set[n].re[i] *= cos_p;
					}*/

				}

				MPI_Bcast(ml_wave_set[sk].eigen_values, num_solution, MPI_DOUBLE, 0, ml_grid.mpi_comm);
				
#ifdef INIT_GRAM_SCHMIDT
				GramSchmidt_ddm(ml_grid, num_solution, ml_wave_set[sk].l_psi_set, dVol);
#endif
			}
		}

		
	}

}

inline
void QUMASUN_BASE1::mInitializeStateFile() {
	
    bool allow_different_grid = true;

	const bool is_root_global = IsRoot(m_mpi_comm);
	const bool is_root_each_ddm = IsRoot(m_ddm_comm);

    FFTW_Executor* fft3d = nullptr;
    FFTW_Executor* fft3d_2 = nullptr;

    const double dV = m_dx * m_dy * m_dz;
	
	const int TAG = 30000;
	const int TAGO = 310000;

	for (int sk = 0; sk < m_all_kinds_spin_kpoint; ++sk) {
		if ((m_having_spin_kpoint_begin <= sk) && (sk < m_having_spin_kpoint_begin + m_num_having_spin_kpoint)) {
			const auto& ws = ml_wave_set[sk - m_having_spin_kpoint_begin];

            for (int n = 0; n < num_solution; ++n) {
                //watch.Record(0);

                double* state_vector_re = nullptr;
                double* state_vector_im = nullptr;
                if (is_root_each_ddm) {
                    auto& strfilepath_re = m_initial_state_files[sk * m_total_state * 2 + (n + m_begin_state) * 2];
                    auto& strfilepath_im = m_initial_state_files[sk * m_total_state * 2 + (n + m_begin_state) * 2 + 1];

                    const char* filepath_re = strfilepath_re.c_str();
                    const char* filepath_im = strfilepath_im.c_str();
                    std::string operation_key_re;
                    if (strfilepath_re[0] == '[') {
                        const size_t p_end = strfilepath_re.find_first_of(']');
                        operation_key_re = strfilepath_re.substr(1, p_end -1);  
                        filepath_re += p_end + 1;
                    }
                    std::string operation_key_im;
                    if (strfilepath_im[0] == '[') {
                        const size_t p_end = strfilepath_im.find_first_of(']');
                        operation_key_im = strfilepath_im.substr(1, p_end - 1);
                        filepath_im += p_end + 1;
                    }


                    printf("loading %s\n", filepath_re); fflush(stdout);
                    CubeReader3::Frame frame;
                    {
                        CubeReader3 cube_loader;
                        frame = cube_loader.LoadCube(filepath_re);
                        size_t size = frame.grid_x * frame.grid_y * frame.grid_z;
                        state_vector_re = gy::AlignedAlloc<double>(size);
                        cube_loader.GetMeshData(state_vector_re);
                    }
					{
                        const char word[] = "eigenvalue=";
                        const int length = strlen(word);
                        //if (std::strncmp(frame.comment, word, length) == 0) {
                        const char* p_ev = std::strstr(frame.comment, word);
                        if (p_ev) {
                            ws.eigen_values[n] = strtod(p_ev + length, nullptr);
                        } else {
                            ws.eigen_values[n] = 0.0;
                        }
                    }
					
                    int kpoint_x, kpoint_y, kpoint_z, kx_size, ky_size, kz_size;
                    if (m_initial_expand_from_kpoint) {
                        const char kword[] = "kpoint=";
                        const char* pk = std::strstr(frame.comment, kword);
                        if (pk) {
                            pk += strlen(kword);
                            char* pnext = nullptr;
                            kpoint_x = strtol(pk, &pnext, 10);
                            pk = pnext + 1;
                            kx_size = strtol(pk, &pnext, 10);
                            pk = pnext + 1;
                            kpoint_y = strtol(pk, &pnext, 10);
                            pk = pnext + 1;
                            ky_size = strtol(pk, &pnext, 10);
                            pk = pnext + 1;
                            kpoint_z = strtol(pk, &pnext, 10);
                            pk = pnext + 1;
                            kz_size = strtol(pk, &pnext, 10);
                            pk = pnext + 1;
                        }
                    }


                    printf("loading %s\n", filepath_im); fflush(stdout); 
                    CubeReader3::Frame frame_im;
                    {
                        CubeReader3 cube_loader;
                        frame_im = cube_loader.LoadCube(filepath_im);
                        size_t size = frame_im.grid_x * frame_im.grid_y * frame_im.grid_z;
                        state_vector_im = gy::AlignedAlloc<double>(size);
                        cube_loader.GetMeshData(state_vector_im);
                    }
                    if ((frame.grid_x != frame_im.grid_x) || (frame.grid_y != frame_im.grid_y) || (frame.grid_z != frame_im.grid_z)) {
                        printf("ERROR: grid size is inconsistent between real and imaginary part\n"); fflush(stdout);
                        gy::AlignedFree(state_vector_re);
                        gy::AlignedFree(state_vector_im);
                        return;
                    }

                    if ((frame.grid_x != m_size_x) || (frame.grid_y != m_size_y) || (frame.grid_z != m_size_z)) {
                        bool is_convert = false;
                        int new_x, new_y, new_z;
                        if (m_initial_expand_from_kpoint) {
                            if ((frame.grid_x * kx_size != m_size_x) ||
                                (frame.grid_y * ky_size != m_size_y) ||
                                (frame.grid_z * kz_size != m_size_z)) {

                                if (allow_different_grid) {
                                    if (((m_size_x % kx_size) == 0) && ((m_size_y % ky_size) == 0) && ((m_size_z % kz_size) == 0)) {
                                        //convert grid size via FFT//
                                        is_convert = true;
                                        new_x = m_size_x / kx_size;
                                        new_y = m_size_y / ky_size;
                                        new_z = m_size_z / kz_size;
                                    }
                                }
                            }
                        } else {
                            if (allow_different_grid) {
                                is_convert = true;
                                new_x = m_size_x;
                                new_y = m_size_y;
                                new_z = m_size_z;
                            }
                        }

                        if(is_convert){
                            //FFTしてupscaling//
                            if (fft3d == nullptr) {
                                fft3d = new FFTW_Executor;
                                fft3d->Initialize(frame.grid_x, frame.grid_y, frame.grid_z, FFTW_ESTIMATE);
                            }
                            if (fft3d_2 == nullptr) {
                                fft3d_2 = new FFTW_Executor;
                                fft3d_2->Initialize(new_x, new_y, new_z, FFTW_ESTIMATE);
                            }

                            const int64_t org_size = frame.grid_x * frame.grid_y * frame.grid_z;
                            const int64_t new_size = new_x * new_y * new_z;
                            OneComplex* src_x = (OneComplex*)m_work;
                            OneComplex* cnv_k = src_x + org_size;
                            for (int64_t i = 0; i < org_size; ++i) {
                                src_x[i].r = state_vector_re[i];
                                src_x[i].i = state_vector_im[i];
                            }
                            fft3d->ForwardDirect(src_x);
                            UpDownConvert_Kspace_3d_any(src_x, frame.grid_x, frame.grid_y, frame.grid_z, cnv_k, new_x, new_y, new_z);
                            fft3d_2->BackwardDirect(cnv_k);
                            gy::AlignedFree(state_vector_re);
                            gy::AlignedFree(state_vector_im);
                            state_vector_re = gy::AlignedAlloc<double>(new_size);
                            state_vector_im = gy::AlignedAlloc<double>(new_size);
                            double norm= 0.0;
                            for (int64_t i = 0; i < new_size; ++i) {
                                norm += (cnv_k[i].r* cnv_k[i].r + cnv_k[i].i* cnv_k[i].i) * dV;
                            }
                            const double isqrtn = 1.0 / sqrt(norm);
                            for (int64_t i = 0; i < new_size; ++i) {
                                state_vector_re[i] = cnv_k[i].r * isqrtn;
                                state_vector_im[i] = cnv_k[i].i * isqrtn;
                            }

                        } else if (!m_initial_expand_from_kpoint) {
                            printf("ERROR: grid size of initial state is inconsistent\n"); fflush(stdout);
                            gy::AlignedFree(state_vector_re);
                            gy::AlignedFree(state_vector_im);
                            return;
                        }
                    }
                
				
                    //kpoint sampleした状態を、拡張してガンマ点に読み込み
                    if (m_initial_expand_from_kpoint) {
                        if ((frame.grid_x * kx_size != m_size_x) ||
                            (frame.grid_y * ky_size != m_size_y) ||
                            (frame.grid_z * kz_size != m_size_z)) {
                            printf("ERROR: at the expansion of initial state from kpoint, grid size is inconsistent\n"); fflush(stdout);
                            gy::AlignedFree(state_vector_re);
                            gy::AlignedFree(state_vector_im);
                            return;
                        }

                        double* expanded_re = gy::AlignedAlloc<double>(m_size_x * m_size_y * m_size_z);
                        double* expanded_im = gy::AlignedAlloc<double>(m_size_x * m_size_y * m_size_z);
                        const double gx_dx = (double)kpoint_x * m_dkx * m_dx;
                        const double gy_dy = (double)kpoint_y * m_dky * m_dy;
                        const double gz_dz = (double)kpoint_z * m_dkz * m_dz;

                        const int src_size_x = frame.grid_x;
                        const int src_size_y = frame.grid_y;
                        const int src_size_z = frame.grid_z;
                        
                        const double scale = 1.0 / std::sqrt((double)((m_size_x / src_size_x) * (m_size_y / src_size_y) * (m_size_z / src_size_z)));


                        for (int iz = 0; iz < m_size_z; ++iz) {
                            const double kzz = gz_dz * (double)iz;
                            for (int iy = 0; iy < m_size_y; ++iy) {
                                const double kyy = gy_dy * (double)iy;
                                for (int ix = 0; ix < m_size_x; ++ix) {
                                    const double kxx = gx_dx * (double)ix;
                                    const int i = ix + m_size_x * (iy + m_size_y * iz);
                                    const int si = (ix% src_size_x) + src_size_x * ((iy % src_size_y) + src_size_y * (iz % src_size_z));

                                    double cosikx = cos(kxx + kyy + kzz);
                                    double sinikx = sin(kxx + kyy + kzz);

                                    expanded_re[i] = scale*(cosikx * state_vector_re[si] - sinikx * state_vector_im[si]);
                                    expanded_im[i] = scale*(sinikx * state_vector_re[si] + cosikx * state_vector_im[si]);

                                }
                            }
                        }
                        gy::AlignedFree(state_vector_re);
                        gy::AlignedFree(state_vector_im);
                        state_vector_re = expanded_re;
                        state_vector_im = expanded_im;
                    }
					
                    {//additional velocity for wave
                        auto it = m_velocity_for_wave.find(operation_key_re);
                        if (it != m_velocity_for_wave.end()) {
                            vec3d v{ it->second.velocity_x, it->second.velocity_y, it->second.velocity_z };
                            vec3d r0{ it->second.center_x, it->second.center_y, it->second.center_z };
                            AddVelocityForWave(m_global_grid, SoAComplex{ state_vector_re, state_vector_im }, v, r0, m_global_grid, m_dx, m_dy, m_dz);

                        }
                    }
                    
                    //test mode/////////////////////////////////////////
                    if (Abs(m_initial_shift_grid) > 1.0e-12) {
                        SoAComplex psi{ state_vector_re ,state_vector_im };
                        mShiftComplexFieldViaFFT(psi, m_initial_shift_grid.x * m_dx, m_initial_shift_grid.y * m_dy, m_initial_shift_grid.z * m_dz);

                    }
                    /////////////////////////////////////////test mode//


                    //watch.Record(40);
                }

                mScatterField(ws.l_psi_set[n].re, state_vector_re);
                mScatterField(ws.l_psi_set[n].im, state_vector_im);
                //watch.Record(41);

                if (is_root_each_ddm) {
                    gy::AlignedFree(state_vector_re);
                    gy::AlignedFree(state_vector_im);

                }
            }

		}

	}


    if (fft3d) delete fft3d;
    if (fft3d_2) delete fft3d_2;


}

inline
void QUMASUN_BASE1::mInitializeDensity() {
	if (m_initial_density == "none") {

		if (IsRoot(m_mpi_comm)) {
			gy::ZeroClear<double>(&m_rho[0], m_size_3d);
		}
		mSetDensityByPsi();

        if(IsRoot(m_same_ddm_place_comm)){
            double local_size = ml_grid.Size3D();
            double sum = 0.0;
            for (int i = 0; i < local_size; ++i) {
                sum += ml_rho[i];
            }
            sum *= m_dx * m_dy * m_dz;
            double total=0.0;
            MPI_Allreduce(&sum, &total, 1, MPI_DOUBLE, MPI_SUM, m_ddm_comm);
            if (num_electrons) {
                //scaling//
                total = (double)num_electrons / total;
                sum = 0.0;
                for (int i = 0; i < local_size; ++i) {
                    ml_rho[i] *= total;
                    sum += ml_rho[i];
                }
                sum *= m_dx * m_dy * m_dz;
                MPI_Reduce(&sum, &total, 1, MPI_DOUBLE, MPI_SUM, 0, m_ddm_comm);
            }
            if (IsRoot(m_ddm_comm)) {
                printf("total_rho(initial_correction) = %.15f\n", total);
            }
        }

	} else if (m_initial_density == "atom") {
		//rootで計算////////////////////////////////
		const bool is_root_global = IsRoot(m_mpi_comm);
		if (is_root_global) {
			auto& rho = m_rho;

			gy::ZeroClear<double>(&rho[0], m_size_3d);

            std::map<int, PseudoPot_MBK*> ps_list;
			std::map<int, Orbital_PAO> pao_list;
            
			for (const auto& p : m_pseudo_pot_set) {
                const int Z = p.first;
                //const auto& pp_path = p.second;
                bool flag = false;
                auto* ps = m_pp_SvF.mFindPseudoPot(Z);
                if (ps) {
                    if (ps->has_ve_density) {
                        ps_list.emplace(Z, ps);
                        flag = true;
                    }
                }
                if(!flag){
                    auto itr = m_atomic_wave_set.find(Z);
                    if (itr != m_atomic_wave_set.end()) {
                        pao_list.emplace(Z, LoadPAO(itr->second.c_str()));
                    } else {
                        printf("ERROR: file for atomic density (*.pao) should be not set in input file.\n");
                    }
                }
			}


			for (int n = 0; n < m_num_nuclei; ++n) {
                const int Z = m_nuclei[n].Z;
                const double R0_x = m_nuclei[n].Rx;
                const double R0_y = m_nuclei[n].Ry;
                const double R0_z = m_nuclei[n].Rz;
                const double* ve_density;
                double xi_min;
                double xi_delta;
                int num_radial_grids;

                auto itr = ps_list.find(Z);
                if (itr != ps_list.end()) {
                    auto& ps = itr->second;
                    num_radial_grids = ps->num_radial_grids;
                    xi_min = ps->xi_min;
                    xi_delta = ps->xi_delta;
                    ve_density = ps->ve_density;

                } else {
                    const auto& pao = pao_list[Z];
                    num_radial_grids = pao.num_radial_grids;
                    xi_min = pao.xi_min;
                    xi_delta = pao.xi_delta;
                    ve_density = pao.valence_charge;
                }
                const double rr_min = pow2(exp(xi_min));


                //const int pao_id = (READ_INITIAL_PAO - 1 > 0) ? (READ_INITIAL_PAO - 1) : 0;
                const int pao_id = n;
                ForXYZ(GridRange{ 0,0,0,m_size_x,m_size_y,m_size_z },
                    [&](int64_t i, int64_t ix, int64_t iy, int64_t iz) {
                        const double rz2 = pow2(fold(m_dz * (double)iz - R0_z, m_box_z));
                        const double ry2 = pow2(fold(m_dy * (double)iy - R0_y, m_box_y));
                        const double rx2 = pow2(fold(m_dx * (double)ix - R0_x, m_box_x));

                        const double rr = rx2 + ry2 + rz2;
                        if (rr_min > rr) {
                            double val = ve_density[0];
                            rho[i] += val;
                        } else {
                            double val = RadialGrid2::GetValueBySquare(rr, ve_density, num_radial_grids, xi_min, xi_delta);
                            rho[i] += val;
                        }

                    });
                
			}


			double sum = 0.0;
			for (size_t i = 0; i < m_size_3d; ++i) {
				sum += rho[i];
			}
			sum *= m_dx * m_dy * m_dz;
			printf("init_rho=%f\n", sum);
            if (num_electrons) {
                const double scale = (double)num_electrons / sum;
                sum = 0.0;
                for (size_t i = 0; i < m_size_3d; ++i) {
                    rho[i] *= scale;
                    sum += rho[i];
                }
                sum *= m_dx * m_dy * m_dz;
                printf("init_rho(corrected)=%f\n", sum);
            }

			for (auto&& p : pao_list) {
				p.second.Release();
			}


			if (is_spin_on) {
                const double ratio_diff = (double)m_initial_spin_differnce / sum;
				for (size_t i = 0; i < m_size_3d; ++i) {
					m_rho_diff[i] = rho[i] * ratio_diff;
				}
			}
		
            mHierarchyScatterField(ml_rho, &m_rho[0]);
            if (is_spin_on) {
                mHierarchyScatterField(ml_rho_diff, &m_rho_diff[0]);
            }
        } else {

            mHierarchyScatterField(ml_rho, nullptr);
            if (is_spin_on) {
                mHierarchyScatterField(ml_rho_diff, nullptr);
            }
        }
        
    } else if (m_initial_density == "atom_spin") {//原子毎に初期のスピン偏極を設定する場合//
        //rootで計算////////////////////////////////
        const bool is_root_global = IsRoot(m_mpi_comm);
        if (is_root_global) {
            auto& rho = m_rho;

            gy::ZeroClear<double>(&rho[0], m_size_3d);

            std::map<int, PseudoPot_MBK*> ps_list;
            std::map<int, Orbital_PAO> pao_list;

            for (const auto& p : m_pseudo_pot_set) {
                const int Z = p.first;
                //const auto& pp_path = p.second;
                bool flag = false;
                auto* ps = m_pp_SvF.mFindPseudoPot(Z);
                if (ps) {
                    if (ps->has_ve_density) {
                        ps_list.emplace(Z, ps);
                        flag = true;
                    }
                }
                if (!flag) {
                    auto itr = m_atomic_wave_set.find(Z);
                    if (itr != m_atomic_wave_set.end()) {
                        pao_list.emplace(Z, LoadPAO(itr->second.c_str()));
                    } else {
                        printf("ERROR: file for atomic density (*.pao) should be not set in input file.\n");
                    }
                }
            }

            if (m_initial_atomic_spin_difference.size() != m_num_nuclei) {
                printf("ERROR: initial atomic apin difference is not correct. Check numbe of lines in input file.\n");
                fflush(stdout);
            }

            for (int n = 0; n < m_num_nuclei; ++n) {
                const int Z = m_nuclei[n].Z;
                const double R0_x = m_nuclei[n].Rx;
                const double R0_y = m_nuclei[n].Ry;
                const double R0_z = m_nuclei[n].Rz;
                const double* ve_density;
                double xi_min;
                double xi_delta;
                int num_radial_grids;
                double ve = 1.0;
                auto itr = ps_list.find(Z);
                if (itr != ps_list.end()) {
                    auto& ps = itr->second;
                    num_radial_grids = ps->num_radial_grids;
                    xi_min = ps->xi_min;
                    xi_delta = ps->xi_delta;
                    ve_density = ps->ve_density;
                    ve = (double)ps->valence_electron;
                } else {
                    const auto& pao = pao_list[Z];
                    num_radial_grids = pao.num_radial_grids;
                    xi_min = pao.xi_min;
                    xi_delta = pao.xi_delta;
                    ve_density = pao.valence_charge;
                }
                const double rr_min = pow2(exp(xi_min));


                //const int pao_id = (READ_INITIAL_PAO - 1 > 0) ? (READ_INITIAL_PAO - 1) : 0;
                const int pao_id = n;
                double* rho_diff = m_rho_diff.get();
                double spin_pol = m_initial_atomic_spin_difference[n]/ve;
                ForXYZ(GridRange{ 0,0,0,m_size_x,m_size_y,m_size_z },
                    [&](int64_t i, int64_t ix, int64_t iy, int64_t iz) {
                        const double rz2 = pow2(fold(m_dz * (double)iz - R0_z, m_box_z));
                        const double ry2 = pow2(fold(m_dy * (double)iy - R0_y, m_box_y));
                        const double rx2 = pow2(fold(m_dx * (double)ix - R0_x, m_box_x));

                        const double rr = rx2 + ry2 + rz2;
                        if (rr_min > rr) {
                            double val = ve_density[0];
                            rho[i] += val;
                            rho_diff[i] += val * spin_pol;
                        } else {
                            double val = RadialGrid2::GetValueBySquare(rr, ve_density, num_radial_grids, xi_min, xi_delta);
                            rho[i] += val;
                            rho_diff[i] += val * spin_pol;
                        }

                    });

            }


            double sum = 0.0;
            for (size_t i = 0; i < m_size_3d; ++i) {
                sum += rho[i];
            }
            sum *= m_dx * m_dy * m_dz;
            printf("init_rho=%f\n", sum);
            if (num_electrons) {
                const double scale = (double)num_electrons / sum;
                sum = 0.0;
                for (size_t i = 0; i < m_size_3d; ++i) {
                    rho[i] *= scale;
                    m_rho_diff[i] *= scale;
                    sum += rho[i];
                }
                sum *= m_dx * m_dy * m_dz;
                printf("init_rho(corrected)=%f\n", sum);
            }

            for (auto&& p : pao_list) {
                p.second.Release();
            }



            mHierarchyScatterField(ml_rho, &m_rho[0]);
            if (is_spin_on) {
                mHierarchyScatterField(ml_rho_diff, &m_rho_diff[0]);
            }
        } else {

            mHierarchyScatterField(ml_rho, nullptr);
            if (is_spin_on) {
                mHierarchyScatterField(ml_rho_diff, nullptr);
            }
        }


    } else {//(m_initial_density==cube)//

		const bool is_root_spin = IsRoot(m_mpi_comm);
        if (is_root_spin) {


            CubeReader3::Frame frame;
            {
                CubeReader3 cube_loader;
                frame = cube_loader.LoadCube(m_initial_density.c_str());
                size_t size = frame.grid_x * frame.grid_y * frame.grid_z;
                double* density = new double[size];
                cube_loader.GetMeshData(density);

                if (m_initial_expand_from_kpoint) {

                    const int src_size_x = frame.grid_x;
                    const int src_size_y = frame.grid_y;
                    const int src_size_z = frame.grid_z;

                    const double scale = 1.0 / (double)((m_size_x / src_size_x) * (m_size_y / src_size_y) * (m_size_z / src_size_z));

                    for (int iz = 0; iz < m_size_z; ++iz) {
                        for (int iy = 0; iy < m_size_y; ++iy) {
                            for (int ix = 0; ix < m_size_x; ++ix) {
                                const int i = ix + m_size_x * (iy + m_size_y * iz);
                                const int si = (ix % src_size_x) + src_size_x * ((iy % src_size_y) + src_size_y * (iz % src_size_z));
                                m_rho[i] = density[si];
                            }
                        }
                    }
                } else {
                    for (size_t i = 0; i < m_size_3d; ++i) {
                        m_rho[i] = density[i];
                    }
                }

                delete[] density;


                //test mode/////////////////////////////////////////
                if (Abs(m_initial_shift_grid) > 1.0e-12) {
                    mShiftRealFieldViaFFT(&m_rho[0], m_initial_shift_grid.x * m_dx, m_initial_shift_grid.y * m_dy, m_initial_shift_grid.z * m_dz);

                }
                /////////////////////////////////////////test mode//

            }

            if (is_spin_on) {
                if (m_initial_density_difference != "none") {
                    CubeReader3 cube_loader;
                    frame = cube_loader.LoadCube(m_initial_density_difference.c_str());
                    size_t size = frame.grid_x * frame.grid_y * frame.grid_z;
                    double* density_diff = new double[size];
                    cube_loader.GetMeshData(density_diff);

                    if (m_initial_expand_from_kpoint) {

                        const int src_size_x = frame.grid_x;
                        const int src_size_y = frame.grid_y;
                        const int src_size_z = frame.grid_z;

                        const double scale = 1.0 / (double)((m_size_x / src_size_x) * (m_size_y / src_size_y) * (m_size_z / src_size_z));

                        for (int iz = 0; iz < m_size_z; ++iz) {
                            for (int iy = 0; iy < m_size_y; ++iy) {
                                for (int ix = 0; ix < m_size_x; ++ix) {
                                    const int i = ix + m_size_x * (iy + m_size_y * iz);
                                    const int si = (ix % src_size_x) + src_size_x * ((iy % src_size_y) + src_size_y * (iz % src_size_z));
                                    m_rho_diff[i] = density_diff[si];
                                }
                            }
                        }
                    } else {
                        for (size_t i = 0; i < m_size_3d; ++i) {
                            m_rho_diff[i] = density_diff[i];
                        }
                    }


                    delete[] density_diff;


                    //test mode/////////////////////////////////////////
                    if (Abs(m_initial_shift_grid) > 1.0e-12) {
                        mShiftRealFieldViaFFT(&m_rho_diff[0], m_initial_shift_grid.x * m_dx, m_initial_shift_grid.y * m_dy, m_initial_shift_grid.z * m_dz);

                    }
                    /////////////////////////////////////////test mode//

                } else {
                    for (size_t i = 0; i < m_size_3d; ++i) {
                        m_rho_diff[i] = 0.0;
                    }
                }
            }

            mHierarchyScatterField(ml_rho, &m_rho[0]);
            if (is_spin_on) {
                mHierarchyScatterField(ml_rho_diff, &m_rho_diff[0]);
            }

        }else{
            mHierarchyScatterField(ml_rho, nullptr);
            if (is_spin_on) {
                mHierarchyScatterField(ml_rho_diff, nullptr);
            }

		}
        
	}


	if (IsRoot(m_mpi_comm)) {
		memcpy(&m_rho_prev[0], &m_rho[0], sizeof(double)*m_size_3d);
	}
    //printf("TEST: %s: %d\n", __FILE__, __LINE__);
}


inline
void QUMASUN_BASE1::mShiftRealFieldViaFFT(double* rho, double Rx, double Ry, double Rz) {

    
    auto* buf = m_fftw->GetBuffer();
    const int64_t size_3d = m_size_3d;
    for (int64_t i = 0; i < size_3d; ++i) {
        buf[i].r = rho[i];
        buf[i].i = 0.0;
    }
    m_fftw->ForwardDirect(buf);
    ShiftInKspace_update((OneComplex*)buf, Rx, Ry, Rz, m_size_x, m_size_y, m_size_z, m_dx, m_dy, m_dz);
    m_fftw->BackwardDirect(buf);
    const double inv = 1.0 / (double)size_3d;
    for (int64_t i = 0; i < size_3d; ++i) {
        rho[i] = buf[i].r * inv;
    }

}

inline
void QUMASUN_BASE1::mShiftComplexFieldViaFFT(SoAComplex& psi, double Rx, double Ry, double Rz) {
    auto* buf = m_fftw->GetBuffer();
    const int64_t size_3d = m_size_3d;
    for (int64_t i = 0; i < size_3d; ++i) {
        buf[i].r = psi.re[i];
        buf[i].i = psi.im[i];
    }
    m_fftw->ForwardDirect(buf);
    ShiftInKspace_update((OneComplex*)buf, Rx, Ry, Rz, m_size_x, m_size_y, m_size_z, m_dx, m_dy, m_dz);
    m_fftw->BackwardDirect(buf);
    const double inv = 1.0 / (double)size_3d;
    for (int64_t i = 0; i < size_3d; ++i) {
        psi.re[i] = buf[i].r * inv;
        psi.im[i] = buf[i].i * inv;
    }
}

#endif

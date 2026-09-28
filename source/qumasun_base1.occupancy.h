#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include "qumasun_base1.h"
#include "qumasun_base1.error.h"
#include "SetOccupancy.h"

//mpi supported//

inline
void QUMASUN_BASE1::mSetOccupancyInitial() {
    if (m_initial_occupancy.size() > 0) {
        mSetOccupancyIndicate();
    }else {
        mSetOccupancyZero();
    }
}


inline
void QUMASUN_BASE1::mSetOccupancy() {
    if (m_initial_occupancy.size() > 0) {
        mSetOccupancyIndicate();
    } else if (m_kbT < 0.1 * KbHartree) {
        mSetOccupancyZero();
    } else {
        mSetOccupancyTemperature();
    }
}

/*
* spin差が1以外の場合に対応できていないので要検討
*/
inline
void QUMASUN_BASE1::mSetOccupancyZero() {
    const int proc_id = GetProcessID(m_mpi_comm);

    if (IsRoot(m_ddm_comm)) {//only ddm root//
        const double total_k_weight = (double)(m_kpoint_sampling[0] * m_kpoint_sampling[1] * m_kpoint_sampling[2]);
        for (int sk = 0; sk < m_num_having_spin_kpoint; ++sk) {
            const double weight_electron = (is_spin_on ? 1.0 : 2.0) * (double)(ml_wave_set[sk].kpoint_weight) / total_k_weight;            

            double remained_electron = (double)num_electrons / 2.0 - m_begin_state;
            if (is_spin_on) {
                if (ml_wave_set[sk].spin == SPIN::UP) {
                    remained_electron += m_initial_spin_differnce / 2.0;
                } else {
                    remained_electron -= m_initial_spin_differnce / 2.0;
                }
            }
            
            for (int n = 0; n < num_solution; ++n) {
                if (remained_electron >= 1.0) {
                    ml_wave_set[sk].occupancy[n] = weight_electron;
                    remained_electron -= 1.0;
                } else if (remained_electron > 0.0) {
                    ml_wave_set[sk].occupancy[n] = weight_electron * remained_electron;
                    remained_electron = 0.0;
                } else {
                    ml_wave_set[sk].occupancy[n] = 0.0;
                }
            }
            
        }
    }

    for (int sk = 0; sk < m_num_having_spin_kpoint; ++sk) {
        MPI_Bcast(ml_wave_set[sk].occupancy, num_solution, MPI_DOUBLE, 0, m_ddm_comm);
    }

}

inline
void QUMASUN_BASE1::mSetOccupancyTemperature() {
	const int proc_id = GetProcessID(m_mpi_comm);

	if (IsRoot(m_ddm_comm)) {//only ddm root//
		
		const bool is_root = IsRoot(m_same_ddm_place_comm);

		//in all procceses////////////////////
		double l_min_eigen = DBL_MAX;
		double l_max_eigen = -DBL_MAX;
		for (int sk = 0; sk < m_num_having_spin_kpoint; ++sk) {
		//	printf("eigen_min,max = %f, %f\n", ml_wave_set[sk].eigen_values[0], ml_wave_set[sk].eigen_values[num_solution-1]); fflush(stdout);
			
			if (l_min_eigen > ml_wave_set[sk].eigen_values[0]) l_min_eigen = ml_wave_set[sk].eigen_values[0];
			if (l_max_eigen < ml_wave_set[sk].eigen_values[num_solution - 1]) l_max_eigen = ml_wave_set[sk].eigen_values[num_solution - 1];
		}

		printf("l_mu_min_max = %f, %f\n", l_min_eigen, l_max_eigen); fflush(stdout);
        MPI_Barrier(m_same_ddm_place_comm);

		double mu_min = 0.0;
		double mu_max = 0.0;
		MPI_Reduce(&l_min_eigen, &mu_min, 1, MPI_DOUBLE, MPI_MIN, 0, m_same_ddm_place_comm);
		MPI_Reduce(&l_max_eigen, &mu_max, 1, MPI_DOUBLE, MPI_MAX, 0, m_same_ddm_place_comm);

		//printf("red_mu_min_max = %f, %f\n", mu_min, mu_max); fflush(stdout);

		/*
		int total_kpoint = m_all_kinds_spin_kpoint;
		if (is_spin_on) {
			total_kpoint /= 2;
		}
		*/
		const double d_num_electron = (double)(num_electrons);// * total_kpoint
		//const double weight_electron = (is_spin_on) ? 1.0 : 2.0;
		double total_k_weight = (double)(m_kpoint_sampling[0] * m_kpoint_sampling[1] * m_kpoint_sampling[2]);
		if (is_spin_on) total_k_weight *= 2.0;


		constexpr double THRESHOLD_MU = 1.0e-9;
		double signal[2];
		double& mu = signal[0];
		mu = (is_root) ? (mu_min + mu_max) / 2.0 : 0.0;
		signal[1] = (fabs(mu_max - mu_min) > THRESHOLD_MU) ? 1.0 : -1.0; //signal for loop. If signal is negative, loop is stop//. 
		MPI_Bcast(&mu, 2, MPI_DOUBLE, 0, m_same_ddm_place_comm);

		int step = 0;
		if (is_root) {
			printf("init:mu = %f, %f, %f\n", mu, mu_min, mu_max); fflush(stdout);
		}
#if  0

		for (int sk = 0; sk < m_num_having_spin_kpoint; ++sk) {
			const double weight_electron = 2.0 * (double)(ml_wave_set[sk].kpoint_weight) / total_k_weight;
			printf("[%d] weight = (%d:%d,%d,%d), %d, %f\n", proc_id, sk, ml_wave_set[sk].kpoint_x, ml_wave_set[sk].kpoint_y, ml_wave_set[sk].kpoint_z, ml_wave_set[sk].kpoint_weight, weight_electron);
		}
#endif

		double num_e = 0.0;
		while (signal[1] > 0.0) {
			double l_num_e = 0.0;
			for (int sk = 0; sk < m_num_having_spin_kpoint; ++sk) {
				const double weight_electron = 2.0 * (double)(ml_wave_set[sk].kpoint_weight)/ total_k_weight;
				l_num_e += SetOccupancy(weight_electron, m_kbT, num_solution, ml_wave_set[sk].eigen_values, mu, ml_wave_set[sk].occupancy);
			}

			num_e = 0.0;
			MPI_Reduce(&l_num_e, &num_e, 1, MPI_DOUBLE, MPI_SUM, 0, m_same_ddm_place_comm);


			if (is_root) {
//#define DEBUG_PRINT1
#ifdef DEBUG_PRINT1
				printf("occupancy_num_e(%d) = %f, mu = %f, %f, %f\n", step, num_e, mu, mu_min, mu_max);
#endif

				if (num_e < d_num_electron) {
					mu_min = mu;
					mu = (mu_min + mu_max) / 2.0;
					signal[1] = (fabs(mu_max - mu_min) > THRESHOLD_MU) ? 1.0 : -1.0; //signal for loop. If signal is negative, loop is stop//. 
				} else if (num_e > d_num_electron) {
					mu_max = mu;
					mu = (mu_min + mu_max) / 2.0;
					signal[1] = (fabs(mu_max - mu_min) > THRESHOLD_MU) ? 1.0 : -1.0; //signal for loop. If signal is negative, loop is stop//. 
				} else {
					signal[1] = -1.0;
				//	break;
				}
				//mu = (mu_min + mu_max) / 2.0;
				
			}

			MPI_Bcast(&mu, 2, MPI_DOUBLE, 0, m_same_ddm_place_comm);
		//	printf("[%d] mu=%f, %f\n", proc_id, signal[0], signal[1]); fflush(stdout);

		}

		if (is_root) {
			printf("final:mu = %f, %f, %f, %f\n",  mu, mu_min, mu_max, num_e);
		}

//#define DEBUG_PRINT2
#ifdef DEBUG_PRINT2

		for (int sk = 0; sk < m_num_having_spin_kpoint; ++sk) {
			printf("[%d] kpoint=%d,%d,%d, spin=%s\n", proc_id, ml_wave_set[sk].kpoint_x, ml_wave_set[sk].kpoint_y, ml_wave_set[sk].kpoint_z, ml_wave_set[sk].spin == SPIN::UP ? "up" : "down");
			for (int s = 0; s < num_solution; ++s) {
				printf("occupancy[%d] = %f\n", s, ml_wave_set[sk].occupancy[s]);
			}
			fflush(stdout);
		}
#endif
//#undef DEBUG_PRINT2
	} //root of ddm for each spin and kpoint//


	for (int sk = 0; sk < m_num_having_spin_kpoint; ++sk) {
		MPI_Bcast(ml_wave_set[sk].occupancy, num_solution, MPI_DOUBLE, 0, m_ddm_comm);
	}
	

}


inline
void QUMASUN_BASE1::mSetOccupancyIndicate() {
    const int proc_id = GetProcessID(m_mpi_comm);

    if (IsRoot(m_ddm_comm)) {//only ddm root//
        double total_occ = 0.0;
        

        const double total_k_weight = (double)(m_kpoint_sampling[0] * m_kpoint_sampling[1] * m_kpoint_sampling[2]);
        for (int sk = 0; sk < m_num_having_spin_kpoint; ++sk) {
            const double weight_electron = (double)(ml_wave_set[sk].kpoint_weight) / total_k_weight;

            
            for (int n = 0; n < num_solution; ++n) {
                ml_wave_set[sk].occupancy[n] = 0.0;
            }

            if (is_spin_on) {
                if (ml_wave_set[sk].spin == SPIN::UP) {
                    const int n_range_begin = (sk + m_having_spin_kpoint_begin) * m_total_state + m_begin_state;
                    const int n_range_end = n_range_begin + num_solution;
                    //printf("[%d]occ:UP:n_range[%d, %d)\n", proc_id, n_range_begin, n_range_end);

                    int n = 0;
                    for (const auto& occ : m_initial_occupancy) {
                        const int n_end = n + occ.num_state;
                        for (; n < n_end; ++n) {
                            if ((n_range_begin <= n) && (n < n_range_end)) {
                                ml_wave_set[sk].occupancy[n - n_range_begin] = weight_electron * occ.occupancy_up;
                                //printf("[%d]occ:UP:set[%d] = %f\n", proc_id, n-n_range_begin, occ.occupancy_up);
                            }
                        }
                    }
                    if (n < num_solution) {
                        for (; n < num_solution; ++n) {
                            ml_wave_set[sk].occupancy[n - n_range_begin] = 0.0;                            
                        }
                    }
                } else {                    
                    const int n_range_begin = (sk + m_having_spin_kpoint_begin - m_all_kinds_spin_kpoint/2) * m_total_state + m_begin_state;
                    const int n_range_end = n_range_begin + num_solution;
                    //printf("[%d]occ:DOWN:n_range[%d, %d), \n", proc_id, n_range_begin, n_range_end);

                    int n = 0;
                    for (const auto& occ : m_initial_occupancy) {
                        const int n_end = n + occ.num_state;
                        for (; n < n_end; ++n) {
                            if ((n_range_begin <= n) && (n < n_range_end)) {
                                ml_wave_set[sk].occupancy[n - n_range_begin] = weight_electron * occ.occupancy_down;
                                //printf("[%d]occ:DOWN:set[%d] = %f\n", proc_id, n - n_range_begin, occ.occupancy_down);
                            }
                        }
                    }
                    if (n < num_solution) {
                        for (; n < num_solution; ++n) {
                            ml_wave_set[sk].occupancy[n - n_range_begin] = 0.0;
                        }
                    }
                }
            } else {
                const int n_range_begin = sk * m_total_state + m_begin_state;
                const int n_range_end = n_range_begin + num_solution;

                int n = 0;
                for (const auto& occ : m_initial_occupancy) {
                    const int n_end = n + occ.num_state;
                    for (; n < n_end; ++n) {
                        if ((n_range_begin <= n) && (n < n_range_end)) {
                            ml_wave_set[sk].occupancy[n - n_range_begin] = weight_electron * (occ.occupancy_up + occ.occupancy_down);
                        }
                    }
                }
                if (n < num_solution) {
                    for (; n < num_solution; ++n) {
                        ml_wave_set[sk].occupancy[n - n_range_begin] = 0.0;
                    }
                }
            }

            //for check//
            for (int n = 0; n < num_solution; ++n) {
                total_occ += ml_wave_set[sk].occupancy[n];
            }
        }

        double reduced_occ = 0;
        MPI_Reduce(&total_occ, &reduced_occ, 1, MPI_DOUBLE, MPI_SUM, 0, m_same_ddm_place_comm);
        if (IsRoot(m_same_ddm_place_comm)) {
            double diff_occ_electrons = reduced_occ - (double)num_electrons;
            if (fabs(diff_occ_electrons) > 1.0e-3) {
                printf("ERROR: occupancy is not agreement with num_electrons: %f != %d, diff=%f\n", reduced_occ, num_electrons, diff_occ_electrons);
                m_error_flag |= QUMASUN::ErrorCode::OCCUPANCY_NOT_AGREEMENT;
            } else {
                printf("OK: occupancy is agreement with num_electrons: %f == %d, diff=%f\n", reduced_occ, num_electrons, diff_occ_electrons);
            }
        }
    }
    
    for (int sk = 0; sk < m_num_having_spin_kpoint; ++sk) {
        MPI_Bcast(ml_wave_set[sk].occupancy, num_solution, MPI_DOUBLE, 0, m_ddm_comm);
    }

}


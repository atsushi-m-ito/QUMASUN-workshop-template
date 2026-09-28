
#pragma once
//#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <complex>
#include "qumasun_td.h"
#include "lobpcg_z_multi_mpi.h"


namespace QUMASUN {
    template <class OperationA, class WATCH>
    inline
    void TimeEvoH4(const GridRangeMPI& l_grid, const double dVol, const double dt, 
            const int num_solution, SoAComplex* x,
            OperationA OpeA, WATCH& watch)
    {

        const int proc_id = GetProcessID(l_grid.mpi_comm);
        const bool is_root = (proc_id == 0);
        const size_t local_size = l_grid.Size3D();

        double* buffer = new double[local_size * 4];

        SoAComplex Hx{ buffer, buffer + local_size };
        SoAComplex H2x{ buffer + local_size * 2, buffer + local_size * 3 };
        
        for (int n = 0; n < num_solution; ++n) {
            OpeA(Hx, x[n]);

            for (int i = 0; i < local_size; ++i) {
                x[n].re[i] += dt * Hx.im[i];
            }
            for (int i = 0; i < local_size; ++i) {
                x[n].im[i] += -dt * Hx.re[i];
            }

            OpeA(H2x, Hx);
            const double coef2 = dt * dt / 2.0;
            for (int i = 0; i < local_size; ++i) {
                x[n].re[i] += -coef2 * H2x.re[i];
            }
            for (int i = 0; i < local_size; ++i) {
                x[n].im[i] += -coef2 * H2x.im[i];
            }

            auto& H3x = Hx;
            OpeA(H3x, H2x);
            const double coef3 = coef2 * dt / 3.0;
            for (int i = 0; i < local_size; ++i) {
                x[n].re[i] += -coef3 * H3x.im[i];
            }
            for (int i = 0; i < local_size; ++i) {
                x[n].im[i] += coef3 * H3x.re[i];
            }

            auto& H4x = H2x;
            OpeA(H4x, H3x);
            const double coef4 = coef3 * dt / 4.0;
            for (int i = 0; i < local_size; ++i) {
                x[n].re[i] += coef4 * H4x.re[i];
            }
            for (int i = 0; i < local_size; ++i) {
                x[n].im[i] += coef4 * H4x.im[i];
            }

        }
        
        delete[] buffer;
    }


    template <class OperationKV, class OperationNL, class WATCH>
    inline
    void TimeEvoH4_bundle(const GridRangeMPI& l_grid, const double dVol, const double dt,
            const int num_solution, int num_bundle, SoAComplex* x,
        OperationKV OpeKV, OperationNL OpeNL, WATCH& watch)
    {
        //OpeKV: oeprator for kinetic and potential(xc,PP-local,hart)
        //OpneNL: operator for PP-nonlocal

        const int proc_id = GetProcessID(l_grid.mpi_comm);
        const bool is_root = (proc_id == 0);
        const size_t local_size = l_grid.Size3D();

        double* buffer = new double[local_size * num_bundle * 4];

        SoAComplex* Hx = new SoAComplex[num_bundle];
        for (int i = 0; i < num_bundle; ++i) {
            Hx[i].re = buffer + local_size * 2 * i;
            Hx[i].im = buffer + local_size * (2 * i + 1);
        }
        
        SoAComplex* H2x=new SoAComplex[num_bundle];
        for (int i = 0; i < num_bundle; ++i) {
            H2x[i].re = buffer + local_size * 2 * (num_bundle + i);
            H2x[i].im = buffer + local_size * (2 * (num_bundle + i) + 1);
        }
        watch.Record(5);
        for (int n = 0; n < num_solution; n+= num_bundle) {
            
            const int n_end = std::min(num_solution, n + num_bundle);
            OpeKV(Hx, x + n, 0, n_end - n);
            OpeNL(Hx, x + n, 0, n_end - n);

            for (int m = n; m < n_end; ++m) {
                for (int i = 0; i < local_size; ++i) {
                    x[m].re[i] += dt * Hx[m - n].im[i];
                }
                for (int i = 0; i < local_size; ++i) {
                    x[m].im[i] += -dt * Hx[m - n].re[i];
                }
            }

            OpeKV(H2x, Hx, 0, n_end - n);
            OpeNL(H2x, Hx, 0, n_end - n);
            const double coef2 = dt * dt / 2.0;
            for (int m = n; m < n_end; ++m) {
                for (int i = 0; i < local_size; ++i) {
                    x[m].re[i] += -coef2 * H2x[m - n].re[i];
                }
                for (int i = 0; i < local_size; ++i) {
                    x[m].im[i] += -coef2 * H2x[m - n].im[i];
                }
            }

            auto& H3x = Hx;
            OpeKV(H3x, H2x, 0, n_end - n);
            OpeNL(H3x, H2x, 0, n_end - n);             
            const double coef3 = coef2 * dt / 3.0;
            for (int m = n; m < n_end; ++m) {
                for (int i = 0; i < local_size; ++i) {
                    x[m].re[i] += -coef3 * H3x[m - n].im[i];
                }
                for (int i = 0; i < local_size; ++i) {
                    x[m].im[i] += coef3 * H3x[m - n].re[i];
                }
            }

            auto& H4x = H2x;
            OpeKV(H4x, H3x, 0, n_end - n);
            OpeNL(H4x, H3x, 0, n_end - n);
            const double coef4 = coef3 * dt / 4.0;
            for (int m = n; m < n_end; ++m) {
                for (int i = 0; i < local_size; ++i) {
                    x[m].re[i] += coef4 * H4x[m - n].re[i];
                }
                for (int i = 0; i < local_size; ++i) {
                    x[m].im[i] += coef4 * H4x[m - n].im[i];
                }
            }
        }

        delete[] buffer;
        delete[] Hx;
        delete[] H2x;
    }
}

inline
size_t QUMASUN_TD::mWorkSizeTimeEvoH4(int local_size, int num_solution) {
    return 0; // means for complex //
}

//mpi supported//
//四次精度の時間発展演算子を作用する//
inline
void QUMASUN_TD::mTimeEvolutionH4(double time_step_dt) {
	const int proc_id = GetProcessID(m_mpi_comm);
	const bool is_ddm_root = IsRoot(m_ddm_comm);
	const size_t local_size = ml_grid.Size3D();


	for(int sk = 0; sk < m_num_having_spin_kpoint;++sk){
		const int kpoint_x = ml_wave_set[sk].kpoint_x;
		const int kpoint_y = ml_wave_set[sk].kpoint_y;
		const int kpoint_z = ml_wave_set[sk].kpoint_z;
		auto& V_tot = (ml_wave_set[sk].spin == SPIN::UP) ? ml_Vtot : ml_Vtot_down;


        QUMASUN::TimeEvoH4_bundle(ml_grid, m_dx * m_dy * m_dz, time_step_dt, num_solution, 16, 
            ml_wave_set[sk].l_psi_set,
            [&V_tot, &kpoint_x, &kpoint_y, &kpoint_z, &sk, this](SoAComplex* l_Hp, const SoAComplex* l_phi, int begin_n, int end_n) {
                //this->mHamiltonianMatrix_ddm(l_Hp, V_tot, l_phi, kpoint_x, kpoint_y, kpoint_z, sk);
                for (int n = begin_n; n < end_n; ++n) {
                    //calculate q = V p , where V is effective potential////////////////
                    mPotentialMatrix_ddm(l_Hp[n].re, V_tot, l_phi[n].re);
                    mPotentialMatrix_ddm(l_Hp[n].im, V_tot, l_phi[n].im);
                    watch.Record(12);
                
                    
                    //calculate qk += K pk , where K is kinetic energy operator. //////////
                    //差分法, FFTは使わない//
                    mKineticMatrixAdd_ddm_kpint(l_Hp[n], l_phi[n], kpoint_x, kpoint_y, kpoint_z);
                    watch.Record(11);
                
                }
            },
            [&, this](SoAComplex* l_Hp, const SoAComplex* l_phi, int begin_n, int end_n) {
#ifndef IGNORE_NONLOCAL
                m_pp_SvF.ProjectionPP_bundle(l_Hp, l_phi, begin_n, end_n, ml_grid, m_nuclei, m_num_nuclei, sk, (std::byte*)m_work);
#endif
                watch.Record(13);
            },
            watch);


	}
	
}



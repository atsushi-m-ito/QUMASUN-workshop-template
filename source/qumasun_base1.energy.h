#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include "qumasun_base1.h"
#include "Vxc.h"
#include "field_interpolation.h"
#include "GridFor.h"
#include "innerprod_hermite.h"
#include "LaplacianDDMFFT.h"
#include "debug_print.h"




//partially mpi supported//
inline
double QUMASUN_BASE1::mGetTotalEnergy(bool with_kinetic, bool with_nonlocal_force) {
	const bool is_root_global = IsRoot(m_mpi_comm);

	const int proc_id = GetProcessID(m_mpi_comm);
	const int num_procs = GetNumProcess(m_mpi_comm);

    DEBUG_PRINTF("[%d] mGetTotalEnergy 1\n", proc_id); 
    
    if (with_kinetic) {
        m_Ekin = mGetEnergyKineticKspace();
    }
    const double Ekin = m_Ekin;
    
    watch.Record(60);
	DEBUG_PRINTF("[%d] mGetTotalEnergy 2\n", proc_id); 

	double E_pp = 0.0;
	if (m_hamiltonian_type == HAMILTONIAN::KohnSham_PP) {
#ifndef IGNORE_NONLOCAL        
        if (with_nonlocal_force) {
            m_force_nonlocal = std::make_unique<vec3d[]>(m_num_nuclei);
            E_pp = mGetForcePseudoNonlocal((double*)&m_force_nonlocal[0]);
            //mGetForcePseudoNonlocal2((double*)&forces_n_nonlocal[0]);
            watch.Record(70);           
        }else{
            E_pp = mGetEnergyPseudoNonlocal();
            watch.Record(61);
        }
#endif
	}
    DEBUG_PRINTF("[%d] mGetTotalEnergy 3\n", proc_id); 

    double ret_Etot = 0.0;

	if (IsRoot(m_same_ddm_place_comm)) {

        const double Eext = mGetEnergyExtByVextRho();
        const double Ehart = mGetEnergyHartree();

        watch.Record(62);

#ifdef DDM_FFT
        const double Eext_HR = mGetEnergyExtByVhartNuclRho_HR();
#else
        const double Eext_HR = mGetEnergyExtByVextRho_HR();
#endif
        const double E_corecore_nucl_density_HR = mGetEnergyCoreCoreNuclDensity_HR();
        watch.Record(63);

		double Ex = 0.0;
		double Ec = 0.0;
		double VxRho = 0.0;
		double VcRho = 0.0;
		double VxRho_down = 0.0;
		double VcRho_down = 0.0;

		double Exc = 0.0;
#ifndef IGNORE_XC
        if (m_xc_type == XC_TYPE::GGA) {
            Ex = m_E_x;
            Ec = m_E_c;
            Exc = m_E_x + m_E_c;
        }else{ //LDA (LSDA)
#ifdef USE_PCC_HR
            if (!is_spin_on) {
                Exc = mGetEnergyXC(&Ex, &Ec, &VxRho, &VcRho);
            } else {
                Exc = mGetEnergyXCSpin_HR(&Ex, &Ec, &VxRho, &VcRho, &VxRho_down, &VcRho_down);
            }
#else
            if (!is_spin_on) {
                Exc = mGetEnergyXC(&Ex, &Ec, &VxRho, &VcRho);
            } else {
                Exc = mGetEnergyXCSpin(&Ex, &Ec, &VxRho, &VcRho, &VxRho_down, &VcRho_down);
            }
#endif
        }
        watch.Record(64);
#endif

        const double Etot8 = Ekin + Eext_HR + Ehart + Exc + E_pp + E_corecore_nucl_density_HR - m_Ecore_self_HR + m_Enn_close_correction;

        ret_Etot = Etot8;

        DEBUG_PRINTF("[%d] mGetTotalEnergy 8\n", proc_id); 
        EnergyInfo ene;
        ene.Ekin = Ekin;
        ene.Eext = Eext;
        ene.Ehart = Ehart;
        ene.Exc = Exc;
        ene.E_pp = E_pp;
        ene.E_corecore_HR = E_corecore_nucl_density_HR;
        ene.Eext_HR = Eext_HR;
        ene.Ex = Ex;
        ene.Ec = Ec;
        ene.VxRho = VxRho;
        ene.VcRho = VcRho;
        ene.VxRho_down = VxRho_down;
        ene.VcRho_down = VcRho_down;
        if (is_root_global) PrintEnergyAll(proc_id, ene);
	}

    DEBUG_PRINTF("[%d] mGetTotalEnergy 14\n", proc_id);
	return ret_Etot;
	
}

inline void QUMASUN_BASE1::PrintEnergyAll(const int& proc_id, const EnergyInfo& ene)
{
    

    DEBUG_PRINTF("[%d] mGetTotalEnergy 9\n", proc_id);
    //const double Eext_nucl_point = mGetEnergyExtByVhartAtPoint();
    //const double E_corecore_Vext = mGetEnergyCoreCoreByVextAtPoint();
    watch.Record(65);
    DEBUG_PRINTF("[%d] mGetTotalEnergy 10\n", proc_id);
    const double E_corecore_direct = mGetEnergyCoreCore_direct();
    watch.Record(66);
    DEBUG_PRINTF("[%d] mGetTotalEnergy 11\n", proc_id);

    
    const double Etot7 = ene.Ekin + ene.Eext + ene.Ehart + ene.Exc + ene.E_pp + ene.E_corecore_HR - m_Ecore_self_HR + m_Enn_close_correction;
    const double Etot8 = ene.Ekin + ene.Eext_HR + ene.Ehart + ene.Exc + ene.E_pp + ene.E_corecore_HR - m_Ecore_self_HR + m_Enn_close_correction;


    printf("Etot = %f (test: %f)\n", Etot8, Etot7);
    printf("Ekin = %f\n", ene.Ekin);

    if (m_hamiltonian_type == HAMILTONIAN::KohnSham_PP) {

        printf("E_PP_local_HR = %f\n", ene.Eext_HR);
        printf("E_PP_local    = %f (non-use)\n", ene.Eext);
        printf("E_PP_nonlocal = %f\n", ene.E_pp);
    } else {
        printf("Eext = %f\n", ene.Eext);
    }
    printf("Ehart = %f\n", ene.Ehart);
    printf("Exc = %f\n", ene.Exc);
    printf("  Ex = %f\n", ene.Ex);
    printf("  Ec = %f\n", ene.Ec);
    if (!is_spin_on) {
        printf("  VxRho = %f (non-use)\n", ene.VxRho);
        printf("  VcRho = %f (non-use)\n", ene.VcRho);
    } else {
        printf("  VxRho = %f, %f (non-use)\n", ene.VxRho, ene.VxRho_down);
        printf("  VcRho = %f, %f (non-use)\n", ene.VcRho, ene.VcRho_down);
    }
    printf("Enn(Vlocal*rho_nucl-self_v2)HR = %f\n", ene.E_corecore_HR - m_Ecore_self_HR);
    printf("  Enn(Vlocal*rho_nucl)HR = %f\n", ene.E_corecore_HR);
    printf("  Enn(self_v2)HR = %f\n", m_Ecore_self_HR);
    printf("  Enn(self_v2) = %f (non-use)\n", m_Ecore_self_v2);
    printf("  Enn(self) = %f (non-use)\n", -m_Ecore_self);
    //printf("  Enn(Vlocal) = %f (non-use)\n", E_corecore_Vext);
    printf("  Enn(simpleCoulomb) = %f (non-use)\n", E_corecore_direct);
    printf("correct_Coulomb_Vlocal = %f (non-use)\n", m_diff_Coulomb_Vlocal);
    printf("correct_Enn_v2 = %f\n", m_Enn_close_correction);
    //fflush(stdout);

    DEBUG_PRINTF("[%d] mGetTotalEnergy 13\n", proc_id);
    watch.Record(67);

    
}


//差分による運動エネルギー項の見積もり//
//FFTは使わない//
//mpi supported//
inline
double QUMASUN_BASE1::mGetEnergyKinetic_8th_diff() {

	const int proc_id = GetProcessID(m_mpi_comm);
	const int local_size = ml_grid.Size3D();

	double Ekin_tot = 0.0;

	for (int sk = 0; sk < m_num_having_spin_kpoint; ++sk) {
        double EK = 0.0;
		for (int s = 0; s < num_solution; ++s) {
			const double occupancy = ml_wave_set[sk].occupancy[s];
			if (occupancy > 0.0) {
				auto& p = ml_wave_set[sk].l_psi_set[s];

				
				SoAComplex Kp{ m_work, m_work + local_size };
				SoAC::SetZero(Kp, local_size);

				DEBUG_PRINTF("[%d]mGetEnergyKinetic 1\n",proc_id); 

				mKineticMatrixAdd_ddm_kpint(Kp, p, ml_wave_set[sk].kpoint_x, ml_wave_set[sk].kpoint_y, ml_wave_set[sk].kpoint_z);

                EK += SoAC::InnerProdReal(p, Kp, local_size) * occupancy;
			}
		}

        Ekin_tot += EK;
	}

    Ekin_tot *= m_dx * m_dy * m_dz;
	//printf("EK=%f\n"); fflush(stdout);
	
	double tot_EK = 0.0;
	MPI_Reduce(&Ekin_tot, &tot_EK, 1, MPI_DOUBLE, MPI_SUM, 0, m_mpi_comm);
	return tot_EK;
	
}


namespace QUMASUN {

#if 0
    /*
    * gamma点でしか使えない実装
    */
    inline
    double KineticEnergyInKspace(const OneComplex* psi_k, int size_x, int size_y, int size_z, double dx, double dy, double dz,
            int kx_end = 0) {

        //kx_end is used for FFTW_R2C mode//
        if (kx_end == 0) {
            kx_end = size_x;
        }
        const int kz_end = size_z;

        //NOTE: 
        //  V = c rho, where c = 4pi / (kx^2 + ky^2 + kz^2) //
        //    = 1.0/ ((pi*kx^2/box_x^2) + (pi*ky^2/box_y^2) + (pi*kz^2/box_z^2)) 

        const double coef1_x = (2.0 * M_PI / (dx * (double)size_x));
        const double coef1_y = (2.0 * M_PI / (dy * (double)size_y));
        const double coef1_z = (2.0 * M_PI / (dz * (double)size_z));

        auto SQ = [](double x) { return x * x; };

        double Ekin = 0.0;

        for (int kz = 0; kz < kz_end; ++kz) {
            const double kz1 = (coef1_z * (double)(kz * 2 > size_z ? kz - size_z : kz));
            for (int ky = 0; ky < size_y; ++ky) {
                const double ky1 = (coef1_y * (double)(ky * 2 > size_y ? ky - size_y : ky));

                for (int kx = 0; kx < kx_end; ++kx) {
                    const double kx1 = (coef1_x * (double)(kx * 2 > size_x ? kx - size_x : kx));
                    const size_t i = kx + kx_end * (ky + (size_y * kz));

                    const double phase = (kx1 * kx1 + ky1 * ky1 + kz1 * kz1) / 2.0;
                    const double psi_re = psi_k[i].r;
                    const double psi_im = psi_k[i].i;
                    
                    Ekin += phase * (psi_re * psi_re + psi_im * psi_im);
                }

            }
        }
        return Ekin * dx*dy*dz;
    }
#endif

    /*
    * k点サンプリング用のg点が加味してある
    * 
    */
    inline
    double KineticEnergyInKspaceB(const OneComplex* psi_k, int size_x, int size_y, int size_z, double dx, double dy, double dz,
            double gx, double gy, double gz, int kx_end = 0) {

        //kx_end is used for FFTW_R2C mode//
        if (kx_end == 0) {
            kx_end = size_x;
        }
        const int kz_end = size_z;

        //NOTE: 
        //  V = c rho, where c = 4pi / (kx^2 + ky^2 + kz^2) //
        //    = 1.0/ ((pi*kx^2/box_x^2) + (pi*ky^2/box_y^2) + (pi*kz^2/box_z^2)) 

        const double coef1_x = (2.0 * M_PI / (dx * (double)size_x));
        const double coef1_y = (2.0 * M_PI / (dy * (double)size_y));
        const double coef1_z = (2.0 * M_PI / (dz * (double)size_z));


        double Ekin = 0.0;

        for (int kz = 0; kz < kz_end; ++kz) {
            const double kz1 = (coef1_z * (double)(kz * 2 > size_z ? kz - size_z : kz));
            for (int ky = 0; ky < size_y; ++ky) {
                const double ky1 = (coef1_y * (double)(ky * 2 > size_y ? ky - size_y : ky));

                for (int kx = 0; kx < kx_end; ++kx) {
                    const double kx1 = (coef1_x * (double)(kx * 2 > size_x ? kx - size_x : kx));
                    //const double kx1 = (coef1_x * (double)(kx * 2 == size_x ? 0 : kx * 2 > size_x ? kx - size_x : kx));
                    const size_t i = kx + kx_end * (ky + (size_y * kz));

                    const double phase = ((kx1+gx) * (kx1+gx) + (ky1+gy) * (ky1+gy) + (kz1+gz) * (kz1+gz)) / 2.0;
                    const double psi_re = psi_k[i].r;
                    const double psi_im = psi_k[i].i;

                    Ekin += phase * (psi_re * psi_re + psi_im * psi_im);
                }

            }
        }
        return Ekin * dx * dy * dz;
    }

    /*
    * k点サンプリング用のg点が加味してある
    *
    */
    inline
    void KineticEnergyInKspace_bundle(int num_bundle, double* __restrict energies, const OneComplex* __restrict psi_k, int size_x, int size_y, int size_z, double dx, double dy, double dz,
            double gx, double gy, double gz) {

        
        //NOTE: 
        //  V = c rho, where c = 4pi / (kx^2 + ky^2 + kz^2) //
        //    = 1.0/ ((pi*kx^2/box_x^2) + (pi*ky^2/box_y^2) + (pi*kz^2/box_z^2)) 

        const double coef1_x = (2.0 * M_PI / (dx * (double)size_x));
        const double coef1_y = (2.0 * M_PI / (dy * (double)size_y));
        const double coef1_z = (2.0 * M_PI / (dz * (double)size_z));
        const int64_t Nall = (int64_t)size_x * (int64_t)size_y * (int64_t)size_z;

#ifdef GY_WITH_CUDA_OR_HIP
        const double scale = dx * dy * dz;
        gy::For_1d_bundle_reduce<int64_t, double>(Nall, num_bundle, energies, scale,
            GY_LAMBDA(int64_t i, int n){

                const int kx = i % size_x;
                const int ky = (i / size_x) % size_y;
                const int kz = (i / (size_x * size_y));

                const double kz1 = (coef1_z * (double)(kz * 2 > size_z ? kz - size_z : kz)) + gx;
                const double ky1 = (coef1_y * (double)(ky * 2 > size_y ? ky - size_y : ky)) + gy;
                const double kx1 = (coef1_x * (double)(kx * 2 > size_x ? kx - size_x : kx)) + gz;

                const double phase = (kx1 * kx1 + ky1 * ky1 + kz1 * kz1) / 2.0;
                const double psi_re = psi_k[i + Nall * n].r;
                const double psi_im = psi_k[i + Nall * n].i;

                return phase * (psi_re * psi_re + psi_im * psi_im);

            });
#else
        for (int n = 0; n < num_bundle; ++n) {
            double Ekin = 0.0;

            for (int kz = 0; kz < size_z; ++kz) {
                const double kz1 = (coef1_z * (double)(kz * 2 > size_z ? kz - size_z : kz)) + gz;
                for (int ky = 0; ky < size_y; ++ky) {
                    const double ky1 = (coef1_y * (double)(ky * 2 > size_y ? ky - size_y : ky)) + gy;

                    for (int kx = 0; kx < size_x; ++kx) {
                        const double kx1 = (coef1_x * (double)(kx * 2 > size_x ? kx - size_x : kx)) + gx;
                        //const double kx1 = (coef1_x * (double)(kx * 2 == size_x ? 0 : kx * 2 > size_x ? kx - size_x : kx));
                        const size_t i = kx + size_x * (ky + (size_y * kz));

                        //const double phase = ((kx1 + gx) * (kx1 + gx) + (ky1 + gy) * (ky1 + gy) + (kz1 + gz) * (kz1 + gz)) / 2.0;
                        const double phase = (kx1 * kx1 + ky1 * ky1 + kz1 * kz1) / 2.0;
                        const double psi_re = psi_k[i + Nall * n].r;
                        const double psi_im = psi_k[i + Nall * n].i;

                        Ekin += phase * (psi_re * psi_re + psi_im * psi_im);
                    }

                }
            }
            
            energies[n] = Ekin * dx * dy * dz;
        }
#endif
    }
}


inline 
double QUMASUN_BASE1::mGetEnergyKineticKspace_algorithm2() {
    const int proc_id = GetProcessID(m_mpi_comm);
    const int local_size = ml_grid.Size3D();
    const int64_t global_size = m_global_grid.Size3D();
    const double invV = 1.0 / global_size;
    
    auto IntervalPointers = [](double* a, size_t N, size_t num_solution) {
        std::vector<SoAComplex> pointers(num_solution);
        for (size_t i = 0; i < num_solution; ++i) {
            pointers[i].re = a + (i * 2) * N;
            pointers[i].im = a + (i * 2 + 1) * N;
        }
        return pointers;
        };
    auto l_Kx = IntervalPointers(m_work, local_size, num_solution);

    double Ekin_tot = 0.0;

    for (int sk = 0; sk < m_num_having_spin_kpoint; ++sk) {
        auto& l_psi = ml_wave_set[sk].l_psi_set;
        const double gx = (double)ml_wave_set[sk].kpoint_x * m_dkx;
        const double gy = (double)ml_wave_set[sk].kpoint_y * m_dky;
        const double gz = (double)ml_wave_set[sk].kpoint_z * m_dkz;

        memset(m_work, 0, sizeof(double) * local_size * num_solution*2);

        LaplacianDDMFFT(ml_grid, m_global_grid,
            *m_ddm_exchanger,
#ifndef USE_FFT_MANY
            m_fft_x, m_fft_y, m_fft_z,
#endif
            &l_Kx[0], l_psi, num_solution,
            m_dx, m_dy, m_dz,
            gx, gy, gz, watch);

        for (int s = 0; s < num_solution; ++s) {
            const double occupancy = ml_wave_set[sk].occupancy[s];
            if (occupancy > 0.0) {
                Ekin_tot += SoAC::InnerProdReal(l_psi[s], l_Kx[s], local_size) * occupancy;
            }
        }
    }

    Ekin_tot *= m_dx * m_dy * m_dz;

    double tot_EK = 0.0;
    MPI_Reduce(&Ekin_tot, &tot_EK, 1, MPI_DOUBLE, MPI_SUM, 0, m_mpi_comm);
    return tot_EK;
}



inline
double QUMASUN_BASE1::mGetEnergyKineticKspace_algorithm1() {
    const int proc_id = GetProcessID(m_mpi_comm);
    const int local_size = ml_grid.Size3D();
    const int64_t global_size = m_global_grid.Size3D();
    const double invV = 1.0 / global_size;
    DDMGatherScatter exchanger;

    double Ekin_tot = 0.0;

    for (int sk = 0; sk < m_num_having_spin_kpoint; ++sk) {
        auto& l_psi = ml_wave_set[sk].l_psi_set;
        const double gx = (double)ml_wave_set[sk].kpoint_x * m_dkx;
        const double gy = (double)ml_wave_set[sk].kpoint_y * m_dky;
        const double gz = (double)ml_wave_set[sk].kpoint_z * m_dkz;


        const auto range = exchanger.Estimate(m_global_grid, ml_grid, num_solution);
        exchanger.GatherExchangeD2Z(range, m_global_grid, m_work, ml_grid, l_psi[0].re, num_solution, m_work + 2 * m_size_3d * range.num_bundle);
        


        if (m_fft_many_3d.GetNumBundle() != range.num_bundle) {
            m_fft_many_3d.Initialize(m_size_x, m_size_y, m_size_z, range.num_bundle, (OneComplex*)&m_work[0]);
        }
        m_fft_many_3d.ForwardDirect((OneComplex*)&m_work[0]);
        

        double EK = 0.0;

#if 1
        double* energies = gy::AlignedAlloc<double>(range.num_bundle + 1);

        const double* occupancy_list = &(ml_wave_set[sk].occupancy[range.offset]);
        auto* psi_whole_complx = (OneComplex*)&m_work[0];

        QUMASUN::KineticEnergyInKspace_bundle(range.num_bundle, &energies[0], psi_whole_complx, m_size_x, m_size_y, m_size_z, m_dx, m_dy, m_dz, gx, gy, gz);
        gy::Synchronize();

#if 0
        gy::For_1d_bundle_reduce<int, double>(range.num_bundle, 1, &energies[range.num_bundle], 1.0,
            GY_LAMBDA(int n, int dummy){
                return occupancy_list[n] * energies[n];
            });
        EK = energies[range.num_bundle];
#else
        for (int n = 0; n < range.num_bundle; ++n) {
            EK += occupancy_list[n] * energies[n];
        }


#endif
    

#else
        for (int s = 0; s < range.num_bundle; ++s) {
            const double occupancy = ml_wave_set[sk].occupancy[range.offset + s];
            if (occupancy > 0.0) {
                //auto& p = ml_wave_set[sk].l_psi_set[s];


                auto* psi_whole_complx = (OneComplex*)&m_work[s * 2 * global_size];

                double res_K = QUMASUN::KineticEnergyInKspaceB(psi_whole_complx, m_size_x, m_size_y, m_size_z, m_dx, m_dy, m_dz, gx, gy, gz);
                EK += occupancy * res_K;
            }
        }
#endif

        Ekin_tot += EK * invV;

    }

    
    //exchanger.MergeTimer(95, watch);


    double tot_EK = 0.0;
    MPI_Reduce(&Ekin_tot, &tot_EK, 1, MPI_DOUBLE, MPI_SUM, 0, m_mpi_comm);
    return tot_EK;
}




inline
double QUMASUN_BASE1::mGetEnergy_V_rho(const double* V, const double* rho) {

	const int64_t size_3d = ml_grid.Size3D();
	double sum2 = 0.0;
	for (int i = 0; i < size_3d; ++i) {
		sum2 += V[i] * rho[i];
	}
	sum2 *= m_dx * m_dy * m_dz;

	double sum2g = 0.0;
	MPI_Reduce(&sum2, &sum2g, 1, MPI_DOUBLE, MPI_SUM, m_root_id, ml_grid.mpi_comm);
#if 0
printf("DEBUG(%d)mGetEnergy_V_rho=%f, %f\n", GetProcessID(ml_grid.mpi_comm), sum2, sum2g);

    {
        double sum3 = 0.0;
        double sum4 = 0.0;
        for (int i = 0; i < size_3d; ++i) {
            sum4 += V[i]*V[i];
            sum3 += rho[i];
        }
        sum4 *= m_dx * m_dy * m_dz;
        sum3 *= m_dx * m_dy * m_dz;

        printf("DEBUG(%d)sum3,4=%f, %f\n", GetProcessID(ml_grid.mpi_comm), sum3, sum4);
    }
#endif
	return sum2g;
}

inline
double QUMASUN_BASE1::mGetEnergy_V_rho_HR(const double* V, const double* rho) {

    const int64_t size_3d = ml_grid.Size3D() * m_HR_ratio_x * m_HR_ratio_y * m_HR_ratio_z;
    double sum2 = 0.0;
    for (int i = 0; i < size_3d; ++i) {
        sum2 += V[i] * rho[i];
    }
    sum2 *= m_dx * m_dy * m_dz / (double)(m_HR_ratio_x * m_HR_ratio_y * m_HR_ratio_z);

    double sum2g = 0.0;
    MPI_Reduce(&sum2, &sum2g, 1, MPI_DOUBLE, MPI_SUM, m_root_id, ml_grid.mpi_comm);
//printf("DEBUG(%d)mGetEnergy_V_rho_HR=%f, %f\n", GetProcessID(ml_grid.mpi_comm), sum2, sum2g);
    return sum2g;
}


//mpi supported//
inline
double QUMASUN_BASE1::mGetEnergyExtByVextRho() {
	const double EV = mGetEnergy_V_rho(ml_Vext, ml_rho);
	return EV;
}

inline
double QUMASUN_BASE1::mGetEnergyExtByVextRho_HR() {
    const double EV = mGetEnergy_V_rho_HR(ml_hr_Vext, ml_hr_rho);
    return EV;
}

//mpi supported//
/*
* calculate
* (1/2) \int \rho V_hart(\rho) dx
*/
inline
double QUMASUN_BASE1::mGetEnergyHartree() {
	const double EV = mGetEnergy_V_rho(ml_Vhart, ml_rho);
	return EV / 2.0;
}


inline
double QUMASUN_BASE1::mGetEnergyExtByVhartNuclRho_HR() {
    const double EV = mGetEnergy_V_rho_HR(ml_hr_Vhart, ml_hr_nucl_rho);
    return EV;
}

inline
double QUMASUN_BASE1::mGetEnergyCoreCoreNuclDensity_HR() {
    const double EV = mGetEnergy_V_rho_HR(ml_hr_Vext, ml_hr_nucl_rho);
    return EV / 2.0;
}


//mpi supported//
inline
double QUMASUN_BASE1::mGetEnergyXC(double* pEx, double* pEc, double* pVxRho, double* pVcRho) {
	const int64_t size_3d = ml_grid.Size3D();
	const auto& rho = ml_rho;
	const auto& pcc_rho = ml_pcc_rho;
	double sum_buf[4];
	auto& E_x = sum_buf[0];
	auto& E_c = sum_buf[1];
	auto& VxRho  = sum_buf[2];
	auto& VcRho = sum_buf[3];

	E_x = 0.0;
	E_c = 0.0;
	VxRho = 0.0;
	VcRho = 0.0;
	for (size_t i = 0; i < size_3d; ++i) {
		//pcc_charge should be added//
		const double rho_i = rho[i] + pcc_rho[i];
#ifdef XC_IMPLE_VER2
        auto ret = Calc_XC_LSDA_v2(rho[i] + pcc_rho[i], 0.0);
        E_x += ret.rho_E_den_x;
        E_c += ret.rho_E_den_c;
        VxRho += ret.V_x_up * rho_i;
        VcRho += ret.V_c_up * rho_i;

#else
        auto ret = Calc_XC_LDA(rho_i);
		
		E_x += ret.rho_E_den_x;
		E_c += ret.rho_E_den_c;
		VxRho += ret.V_x * rho_i;
		VcRho += ret.V_c * rho_i;
#endif
    }
	E_x *= m_dx * m_dy * m_dz;
	E_c *= m_dx * m_dy * m_dz;
	VxRho *= m_dx * m_dy * m_dz;
	VcRho *= m_dx * m_dy * m_dz;

	
	double sum_buf_g[4];
	MPI_Reduce(sum_buf, sum_buf_g, 4, MPI_DOUBLE, MPI_SUM, m_root_id, ml_grid.mpi_comm);
	for (int i = 0; i < 4; ++i) {
		sum_buf[i] = sum_buf_g[i];
	}
	
	if (pVxRho)*pVxRho = VxRho;
	if (pVcRho)*pVcRho = VcRho;
	*pEx = E_x;
	*pEc = E_c;
	return E_x + E_c;
}

//mpi supported//
inline
double QUMASUN_BASE1::mGetEnergyXCSpin(double* pEx, double* pEc, double* pVxRho_up, double* pVcRho_up, double* pVxRho_down, double* pVcRho_down) {
    const int64_t size_3d = ml_grid.Size3D();
    const auto& rho = ml_rho;
    const auto& pcc_rho = ml_pcc_rho;
    const auto& rho_diff = ml_rho_diff;
    double sum_buf[6]{ 0.0 };
    auto& E_x = sum_buf[0];
    auto& E_c = sum_buf[1];
    auto& VxRho = sum_buf[2];
    auto& VcRho = sum_buf[3];
    auto& VxRho_down = sum_buf[4];
    auto& VcRho_down = sum_buf[5];

    E_x = 0.0;
    E_c = 0.0;
    VxRho = 0.0;
    VcRho = 0.0;
    for (size_t i = 0; i < size_3d; ++i) {
        //pcc_charge should be added//
        const double rho_i = rho[i] + pcc_rho[i];
#ifdef XC_IMPLE_VER2
        auto ret = Calc_XC_LSDA_v2(rho_i, rho_diff[i]);
#else
        auto ret = Calc_XC_LSDA(rho_i, rho_diff[i] / rho_i);
#endif

        E_x += ret.rho_E_den_x;
        E_c += ret.rho_E_den_c;
        VxRho += ret.V_x_up * rho_i;
        VcRho += ret.V_c_up * rho_i;
        VxRho_down += ret.V_x_down * rho_i;
        VcRho_down += ret.V_c_down * rho_i;
    }
    E_x *= m_dx * m_dy * m_dz;
    E_c *= m_dx * m_dy * m_dz;
    VxRho *= m_dx * m_dy * m_dz;
    VcRho *= m_dx * m_dy * m_dz;
    VxRho_down *= m_dx * m_dy * m_dz;
    VcRho_down *= m_dx * m_dy * m_dz;


    double sum_buf_g[6];
    MPI_Reduce(sum_buf, sum_buf_g, 6, MPI_DOUBLE, MPI_SUM, m_root_id, ml_grid.mpi_comm);
    for (int i = 0; i < 6; ++i) {
        sum_buf[i] = sum_buf_g[i];
    }

    if (pVxRho_up)*pVxRho_up = VxRho;
    if (pVcRho_up)*pVcRho_up = VcRho;
    if (pVxRho_down)*pVxRho_down = VxRho_down;
    if (pVcRho_down)*pVcRho_down = VcRho_down;
    *pEx = E_x;
    *pEc = E_c;
    return E_x + E_c;
}


#ifdef USE_PCC_HR
//mpi supported//
inline
double QUMASUN_BASE1::mGetEnergyXCSpin_HR(double* pEx, double* pEc, double* pVxRho_up, double* pVcRho_up, double* pVxRho_down, double* pVcRho_down) {
    const int64_t size_3d = ml_grid.Size3D() * m_HR_ratio_x * m_HR_ratio_y * m_HR_ratio_z;
    const auto& rho = ml_hr_rho;
    const auto& pcc_rho = ml_hr_pcc;
    const auto& rho_diff = ml_hr_rho_diff;
    double sum_buf[6]{ 0.0 };
    auto& E_x = sum_buf[0];
    auto& E_c = sum_buf[1];
    auto& VxRho = sum_buf[2];
    auto& VcRho = sum_buf[3];
    auto& VxRho_down = sum_buf[4];
    auto& VcRho_down = sum_buf[5];

    E_x = 0.0;
    E_c = 0.0;
    VxRho = 0.0;
    VcRho = 0.0;
    for (size_t i = 0; i < size_3d; ++i) {
        //pcc_charge should be added//
        const double rho_i = rho[i] + pcc_rho[i];
#ifdef XC_IMPLE_VER2
        auto ret = Calc_XC_LSDA_v2(rho_i, rho_diff[i]);
#else
        auto ret = Calc_XC_LSDA(rho_i, rho_diff[i] / rho_i);
#endif

        E_x += ret.rho_E_den_x;
        E_c += ret.rho_E_den_c;
        VxRho += ret.V_x_up * rho_i;
        VcRho += ret.V_c_up * rho_i;
        VxRho_down += ret.V_x_down * rho_i;
        VcRho_down += ret.V_c_down * rho_i;
    }

    const double dV = m_dx * m_dy * m_dz / (double)(m_HR_ratio_x * m_HR_ratio_y * m_HR_ratio_z);
    E_x *= dV;
    E_c *= dV;
    VxRho *= dV;
    VcRho *= dV;
    VxRho_down *= dV;
    VcRho_down *= dV;


    double sum_buf_g[6];
    MPI_Reduce(sum_buf, sum_buf_g, 6, MPI_DOUBLE, MPI_SUM, m_root_id, ml_grid.mpi_comm);
    for (int i = 0; i < 6; ++i) {
        sum_buf[i] = sum_buf_g[i];
    }

    if (pVxRho_up)*pVxRho_up = VxRho;
    if (pVcRho_up)*pVcRho_up = VcRho;
    if (pVxRho_down)*pVxRho_down = VxRho_down;
    if (pVcRho_down)*pVcRho_down = VcRho_down;
    *pEx = E_x;
    *pEc = E_c;
    return E_x + E_c;
}

#endif

//mpi supported//
inline
double QUMASUN_BASE1::mGetEnergyPseudoNonlocal() {
	double EV = 0.0;
	for (int sk = 0; sk < m_num_having_spin_kpoint; ++sk) {



#if 1//def GY_WITH_CUDA_OR_HIP
        const int bundle_width = num_solution;
#else
        const int bundle_width = 16;
#endif
        for (int s = 0; s < num_solution; s += bundle_width) {
            const int min_bundle = std::min(bundle_width, num_solution - s);
            //const double occupancy = ml_wave_set[sk].occupancy[s];

            //if (occupancy <= 0.0) continue;
            double pp_ene = m_pp_SvF.EnergyPPnonlocal_bundle(&ml_wave_set[sk].l_psi_set[s], &(ml_wave_set[sk].occupancy[s]), 0, min_bundle, ml_grid, m_nuclei, m_num_nuclei, sk, (std::byte*)m_work);

            EV += pp_ene;
        }

	}

	if (IsRoot(m_ddm_comm)) {
		double tot_EV = 0.0;
		MPI_Reduce(&EV, &tot_EV, 1, MPI_DOUBLE, MPI_SUM, m_root_id, m_same_ddm_place_comm);
		return tot_EV;
	} else {
		return 0.0;
	}
	return EV;
}

inline
double QUMASUN_BASE1::mGetEnergyCoreCore_direct() {
	double EV = 0.0;

	auto Folding = [](double x, double box_width) {
		return x - floor(x / box_width) * box_width;
		};

	auto Distance = [](double x, double box_width) {
		return (fabs(x) < box_width * 0.5) ? x : -copysign(box_width - fabs(x), x);
		};

	for (int i = 0; i < m_num_nuclei; ++i) {
		const double Qi = m_nuclei_valence_elecron[i];
		double xi = Folding(m_nuclei[i].Rx, m_box_x);
		double yi = Folding(m_nuclei[i].Ry, m_box_y);
		double zi = Folding(m_nuclei[i].Rz, m_box_z);
		for (int k = i + 1; k < m_num_nuclei; ++k) {
			const double Qk = m_nuclei_valence_elecron[k];

			double xk = Folding(m_nuclei[k].Rx, m_box_x);
			double yk = Folding(m_nuclei[k].Ry, m_box_y);
			double zk = Folding(m_nuclei[k].Rz, m_box_z);
			double dx = Distance(xk - xi, m_box_x);
			double dy = Distance(yk - yi, m_box_y);
			double dz = Distance(zk - zi, m_box_z);
			double r = sqrt(dx * dx + dy * dy + dz * dz);

			EV += Qi * Qk / r;
		}
	}

	return EV;
}


inline
double QUMASUN_BASE1::mGetEnergyPotentialAtNucl(const double* V) {
	double EV = 0.0;

	auto Folding = [](double x, double box_width) {
		return x - floor(x / box_width) * box_width;
		};

	for (int i = 0; i < m_num_nuclei; ++i) {
		const double Qi = m_nuclei_valence_elecron[i];
		double x = Folding(m_nuclei[i].Rx, m_box_x);
		double y = Folding(m_nuclei[i].Ry, m_box_y);
		double z = Folding(m_nuclei[i].Rz, m_box_z);
		

		const double val_V_ext = Interpolation<double>(x / m_dx, y / m_dy, z / m_dz, V, { 0, 0, 0, m_size_x, m_size_y, m_size_z },
			{ m_size_x, m_size_y, m_size_z });

		EV += val_V_ext * (-Qi);

	}


	return EV;
}

inline
double QUMASUN_BASE1::mGetEnergyCoreCoreByVextAtPoint() {
	double EV = mGetEnergyPotentialAtNucl(&m_Vext[0]);
	return EV / 2.0;
}

inline
double QUMASUN_BASE1::mGetEnergyExtByVhartAtPoint() {
	double EV = mGetEnergyPotentialAtNucl(&m_Vhart[0]);
	return EV;
}



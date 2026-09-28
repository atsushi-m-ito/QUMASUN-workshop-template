#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include "qumasun_base1.h"
#include "Vxc.h"
#include "field_interpolation.h"
#include "GridGradient8th.h"
#include "GridFor.h"
#include "vec3.h"
#include "Grad_DDMFFT.h"


//#define PRINT_ALL_FORCE

//partially mpi supported//
/*
* If nonlocal force was already estimated in mGetTotalEnergy,
* we can skip the estimation of nonlocal force here by stand the flag "is_nonlocal_already_".
* 
*/
inline
void QUMASUN_BASE1::mGetForce(bool is_nonlocal_already) {
    const bool is_root_spin = IsRoot(m_mpi_comm);

    const int proc_id = GetProcessID(m_mpi_comm);
    const int num_procs = GetNumProcess(m_mpi_comm);

    m_nucl_forces = std::make_unique<vec3d[]>(m_num_nuclei);
    //printf("TEST: %s: %d\n", __FILE__, __LINE__);
    //nuclei-electron term from nonlocal potential//
#ifdef IGNORE_NONLOCAL
    m_force_nonlocal = std::make_unique<vec3d[]>(m_num_nuclei);
    memset(&m_force_nonlocal[0], 0, sizeof(double) * m_num_nuclei * 3);    
#else
    if(!is_nonlocal_already){
        if (m_hamiltonian_type == HAMILTONIAN::KohnSham_PP) {
            watch.Record(8);
            m_force_nonlocal = std::make_unique<vec3d[]>(m_num_nuclei);
            mGetForcePseudoNonlocal((double*)&m_force_nonlocal[0]);
            watch.Record(70);
            //printf("TEST: %s: %d\n", __FILE__, __LINE__);
        }
    }
#endif

    auto forces_hr_nn = std::make_unique<vec3d[]>(m_num_nuclei); //nuclei-nuclei term from Hartree potential//
    auto forces_hr_n_hart = std::make_unique<vec3d[]>(m_num_nuclei); //nuclei-electron term from Hartree potential//
    auto forces_xc_pcc = std::make_unique<vec3d[]>(m_num_nuclei); //E_xc term by PCC charge//
    auto forces_tot = std::make_unique<vec3d[]>(m_num_nuclei); //nuclei-nuclei term from Hartree potential//
    
    //nuclei-electron term from Hartree potential//
    for (int i = 0; i < m_num_nuclei; ++i) {
        forces_hr_nn[i].x = 0.0;
        forces_hr_nn[i].y = 0.0;
        forces_hr_nn[i].z = 0.0;
    }

    mGetForceCoreCoreByVextNuclRho_HR((double*)&forces_hr_nn[0]);
    //printf("TEST: %s: %d\n", __FILE__, __LINE__);
    mGetForceHartreeNuclRho_HR((double*)&forces_hr_n_hart[0]);
    //printf("TEST: %s: %d\n", __FILE__, __LINE__);
#ifdef USE_PCC_HR
    mGetForceXcPcc_HR((double*)&forces_xc_pcc[0]);
#else
    mGetForceXcPcc((double*)&forces_xc_pcc[0]);
#endif
    //printf("TEST: %s: %d\n", __FILE__, __LINE__);
	if (IsRoot(m_same_ddm_place_comm)) {
        

        //accept Enn correction when close distance//
        if (is_root_spin) {
            if (m_hamiltonian_type == HAMILTONIAN::KohnSham_PP) {
                for (int i = 0; i < m_num_nuclei; ++i) {
                    forces_hr_nn[i] += m_force_nn_TF[i] - m_force_nn_correction[i];
                }
            }
        }

		for (int i = 0; i < m_num_nuclei; ++i) {
			//forces_tot[i] = forces_nn[i] + forces_n_hart[i];
		    forces_tot[i] = forces_hr_nn[i] + forces_hr_n_hart[i] + forces_xc_pcc[i];
		}
		if (m_hamiltonian_type == HAMILTONIAN::KohnSham_PP) {
			for (int i = 0; i < m_num_nuclei; ++i) {
				forces_tot[i] += m_force_nonlocal[i];
			}
		}

        for (int i = 0; i < m_num_nuclei; ++i) {
            m_nucl_forces[i] = forces_tot[i];
        }

		if (is_root_spin) {

            auto TotalForce = [](int m_num_nuclei, const auto& forces) {
                vec3d total_force{ 0.0,0.0,0.0 };
                for (int ni = 0; ni < m_num_nuclei; ++ni) {
                    total_force += forces[ni];
                }
                return total_force;
                };


            {
                printf("\nForce =========================\n");
                    const vec3d total_force = TotalForce(m_num_nuclei, forces_tot);
                printf("Ftot: %8.5f  %8.5f  %8.5f\n\n", total_force.x, total_force.y, total_force.z);
                for (int ni = 0; ni < m_num_nuclei; ++ni) {
                    printf("   %d: %8.5f  %8.5f  %8.5f\n", ni, forces_tot[ni].x, forces_tot[ni].y, forces_tot[ni].z);
                }
            }

            {
                printf("\nForce n-n (HR) including correction for close distance =====================\n");
                const vec3d total_force = TotalForce(m_num_nuclei, forces_hr_nn);
                printf("F_nn_HR: %8.5f  %8.5f  %8.5f\n\n", total_force.x, total_force.y, total_force.z);
#ifdef PRINT_ALL_FORCE
                for (int ni = 0; ni < m_num_nuclei; ++ni) {
                    printf("   %d: %8.5f  %8.5f  %8.5f\n", ni, forces_hr_nn[ni].x, forces_hr_nn[ni].y, forces_hr_nn[ni].z);
                }
#endif
            }


            {
            printf("\nForce n-hart (HR) =====================\n");
                const vec3d total_force = TotalForce(m_num_nuclei, forces_hr_n_hart);
                printf("F_hart_HR: %8.5f  %8.5f  %8.5f\n\n", total_force.x, total_force.y, total_force.z);
#ifdef PRINT_ALL_FORCE
                for (int ni = 0; ni < m_num_nuclei; ++ni) {
                    printf("   %d: %8.5f  %8.5f  %8.5f\n", ni, forces_hr_n_hart[ni].x, forces_hr_n_hart[ni].y, forces_hr_n_hart[ni].z);
                }
#endif
            }


			if (m_hamiltonian_type == HAMILTONIAN::KohnSham_PP) {
				printf("\nForce n-nonlocal =====================\n");
				const vec3d total_force = TotalForce(m_num_nuclei, m_force_nonlocal);
                printf("F_nonlocal: %8.5f  %8.5f  %8.5f\n\n", total_force.x, total_force.y, total_force.z);
#ifdef PRINT_ALL_FORCE
                for (int ni = 0; ni < m_num_nuclei; ++ni) {
					printf("   %d: %8.5f  %8.5f  %8.5f\n", ni, m_force_nonlocal[ni].x, m_force_nonlocal[ni].y, m_force_nonlocal[ni].z);
				}
#endif
			}


            {
                printf("\nForce XC depending PCC =====================\n");
                const vec3d total_force = TotalForce(m_num_nuclei, forces_xc_pcc);
                printf("F_xc_pcc: %8.5f  %8.5f  %8.5f\n\n", total_force.x, total_force.y, total_force.z);
#ifdef PRINT_ALL_FORCE
                for (int ni = 0; ni < m_num_nuclei; ++ni) {
                    printf("   %d: %8.5f  %8.5f  %8.5f\n", ni, forces_xc_pcc[ni].x, forces_xc_pcc[ni].y, forces_xc_pcc[ni].z);
                }
#endif
            }

#if 1
            {
                printf("\nForce n-n correct =====================\n");
                const vec3d total_force = TotalForce(m_num_nuclei, m_force_nn_correction);
                printf("Fnn_correct: %8.5f  %8.5f  %8.5f\n\n", total_force.x, total_force.y, total_force.z);
#ifdef PRINT_ALL_FORCE
                for (int ni = 0; ni < m_num_nuclei; ++ni) {
                    printf("   %d: %8.5f  %8.5f  %8.5f\n", ni, m_force_nn_correction[ni].x, m_force_nn_correction[ni].y, m_force_nn_correction[ni].z);
                }
#endif
            }

            {
                printf("\nForce n-n TF =====================\n");
                const vec3d total_force = TotalForce(m_num_nuclei, m_force_nn_TF);
                printf("Fnn_TF: %8.5f  %8.5f  %8.5f\n\n", total_force.x, total_force.y, total_force.z);
#ifdef PRINT_ALL_FORCE
                for (int ni = 0; ni < m_num_nuclei; ++ni) {
                    printf("   %d: %8.5f  %8.5f  %8.5f\n", ni, m_force_nn_TF[ni].x, m_force_nn_TF[ni].y, m_force_nn_TF[ni].z);
                }
#endif
            }
#endif
			fflush(stdout);

			
		}
	}
    //printf("TEST: %s: %d\n", __FILE__, __LINE__);
	//must do Bcast forces////
    MPI_Bcast(&m_nucl_forces[0], m_num_nuclei * 3, MPI_DOUBLE, 0, m_mpi_comm);
    //printf("TEST: %s: %d\n", __FILE__, __LINE__);
}





/*
* calculate froce from
* -\int n_nucl V_any(\rho) dx
*/

inline
void QUMASUN_BASE1::mGetForceOnNuclRho_gradV_HR(double* force, const double* l_dV_dx, double* l_dV_dy, double* l_dV_dz) {

    auto& m_pp_integrator = m_pp_SvF;

    size_t local_size = ml_grid.Size3D() * m_HR_ratio_x * m_HR_ratio_y * m_HR_ratio_z;


    const int stride = 3;
    m_pp_integrator.InnerForChargeVlocal(force, stride, l_dV_dx, ml_grid, m_nuclei, m_num_nuclei, true);
    DEBUG_PRINTF("InnerForChargeVlocal x\n");
    m_pp_integrator.InnerForChargeVlocal(force + 1, stride, l_dV_dy, ml_grid, m_nuclei, m_num_nuclei, true);
    DEBUG_PRINTF("InnerForChargeVlocal y\n");
    m_pp_integrator.InnerForChargeVlocal(force + 2, stride, l_dV_dz, ml_grid, m_nuclei, m_num_nuclei, true);
    DEBUG_PRINTF("InnerForChargeVlocal z\n");

    for (int i = 0; i < m_num_nuclei * 3; ++i) {
        force[i] = -force[i];
    }

    watch.Record(74);
}

inline
void QUMASUN_BASE1::mGetForceOnNuclRho_diff_proj_HR(double* force, const double* l_hr_Vpot) {

    auto& m_pp_integrator = m_pp_SvF;

    size_t local_size = ml_grid.Size3D() * m_HR_ratio_x * m_HR_ratio_y * m_HR_ratio_z;


    const int stride = 3;
    m_pp_integrator.InnerForChargeVlocalDifferential(force, stride, l_hr_Vpot, ml_grid, m_nuclei, m_num_nuclei, true);
    /*
    for (int i = 0; i < m_num_nuclei * 3; ++i) {
        force[i] = -force[i];
    }
    */
    watch.Record(74);
}

inline 
void QUMASUN_BASE1::mGetForceOnNuclRho_HR(double* force, const double* l_hr_Vpot){


    auto& m_pp_integrator = m_pp_SvF;

    size_t local_size = ml_grid.Size3D() * m_HR_ratio_x * m_HR_ratio_y * m_HR_ratio_z;


    double* l_dV_dx = gy::AlignedAlloc<double>(local_size * 3);
    double* l_dV_dy = l_dV_dx + local_size;
    double* l_dV_dz = l_dV_dx + local_size * 2;

    auto hr_grid = ScaleRange(ml_grid, m_HR_ratio_x, m_HR_ratio_y, m_HR_ratio_z);
    DEBUG_PRINTF("[%d]before Gradient8th_ddm: %zd\n", GetProcessID(ml_grid.mpi_comm), local_size);
    Gradient8th_ddm(hr_grid, l_dV_dx, l_dV_dy, l_dV_dz, l_hr_Vpot, -1.0, m_dx / (double)m_HR_ratio_x, m_dy / (double)m_HR_ratio_y, m_dz / (double)m_HR_ratio_z);
    DEBUG_PRINTF("after Gradient8th_ddm\n");
    watch.Record(73);

    const int stride = 3;
    m_pp_integrator.InnerForChargeVlocal(force, stride, l_dV_dx, ml_grid, m_nuclei, m_num_nuclei, true);
    DEBUG_PRINTF("InnerForChargeVlocal x\n");
    m_pp_integrator.InnerForChargeVlocal(force + 1, stride, l_dV_dy, ml_grid, m_nuclei, m_num_nuclei, true);
    DEBUG_PRINTF("InnerForChargeVlocal y\n");
    m_pp_integrator.InnerForChargeVlocal(force + 2, stride, l_dV_dz, ml_grid, m_nuclei, m_num_nuclei, true);
    DEBUG_PRINTF("InnerForChargeVlocal z\n");
    gy::AlignedFree(l_dV_dx);

    watch.Record(74);
}




/*
* calculate froce from
* -\int n_nucl V_nucl(\rho) dx
*/
inline
void QUMASUN_BASE1::mGetForceCoreCoreByVextNuclRho_HR(double* force) {

#ifdef DIFF_NUCL_RHO_LOCAL
    mGetForceOnNuclRho_diff_proj_HR(force, ml_hr_Vext);
#elif defined(GRAD_VEXT_FFT)
    mGetForceOnNuclRho_gradV_HR(force, ml_hr_dVext_dx, ml_hr_dVext_dy, ml_hr_dVext_dz);
#else
    mGetForceOnNuclRho_HR(force, ml_hr_Vext);
#endif
}


/*
* calculate froce from
* -\int n_nucl V_hart(\rho) dx
*/
inline
void QUMASUN_BASE1::mGetForceHartreeNuclRho_HR(double* force) {

#ifdef DIFF_NUCL_RHO_LOCAL
    mGetForceOnNuclRho_diff_proj_HR(force, ml_hr_Vhart);
#elif defined(GRAD_VHART_FFT)
    mGetForceOnNuclRho_gradV_HR(force, ml_hr_dVhart_dx, ml_hr_dVhart_dy, ml_hr_dVhart_dz);
#else
    mGetForceOnNuclRho_HR(force, ml_hr_Vhart);
#endif
}


/*
* calculate froce due to the Pcc dependency of E_xc of 
* -\int \rho_pcc dV_xc(\rho)/dr dx
* is spin is not neutral, V_xc is mean of V_xc_up and V_xc_down
*/
inline
void QUMASUN_BASE1::mGetForceXcPcc(double* force) {


    auto& m_pp_integrator = m_pp_SvF;


    size_t local_size = ml_grid.Size3D();
    double* l_V_xc = m_work;
    double* l_dV_dx = m_work + local_size;
    double* l_dV_dy = m_work + local_size * 2;
    double* l_dV_dz = m_work + local_size * 3;
    if (is_spin_on) {
        for (int i = 0; i < local_size; ++i) {
            l_V_xc[i] = (ml_Vx[i] + ml_Vx_down[i] + ml_Vc[i] + ml_Vc_down[i]) / 2.0;
        }
    } else {
        for (int i = 0; i < local_size; ++i) {
            l_V_xc[i] = ml_Vx[i] + ml_Vc[i];
        }
    }
    //printf("TEST: %s: %d\n", __FILE__, __LINE__);
#ifdef GRAD_VXC_FFT
#ifdef USE_FFT_MANY
    GradientDDMFFT(ml_grid, m_global_grid, *m_ddm_exchanger, m_fft_many_x, m_fft_many_y, m_fft_many_z, l_V_xc, l_dV_dx, l_dV_dy, l_dV_dz, -1.0, m_dx, m_dy, m_dz, false);
#else
    GradientDDMFFT(ml_grid, m_global_grid, *m_ddm_exchanger, m_fft_x, m_fft_y, m_fft_z, l_V_xc, l_dV_dx, l_dV_dy, l_dV_dz, -1.0, m_dx, m_dy, m_dz, false);
#endif
    watch.Record(71);
#else
    Gradient8th_ddm(ml_grid, l_dV_dx, l_dV_dy, l_dV_dz, l_V_xc, -1.0, m_dx, m_dy, m_dz);
    watch.Record(71);
#endif
    //printf("TEST: %s: %d\n", __FILE__, __LINE__);
    const int stride = 3;
    m_pp_integrator.InnerForChargePcc(force, stride, l_dV_dx, ml_grid, m_nuclei, m_num_nuclei);
    m_pp_integrator.InnerForChargePcc(force + 1, stride, l_dV_dy, ml_grid, m_nuclei, m_num_nuclei);
    m_pp_integrator.InnerForChargePcc(force + 2, stride, l_dV_dz, ml_grid, m_nuclei, m_num_nuclei);
    watch.Record(72);
    //printf("TEST: %s: %d\n", __FILE__, __LINE__);
}

#ifdef USE_PCC_HR
/*
* calculate froce due to the Pcc dependency of E_xc of
* -\int \rho_pcc dV_xc(\rho)/dr dx
* is spin is not neutral, V_xc is mean of V_xc_up and V_xc_down
*/
inline
void QUMASUN_BASE1::mGetForceXcPcc_HR(double* force) {


    auto& m_pp_integrator = m_pp_SvF;

    auto hr_grid = ScaleRange(ml_grid, m_HR_ratio_x, m_HR_ratio_y, m_HR_ratio_z);
    size_t local_size = hr_grid.Size3D();
    double* l_V_xc = m_work;
    double* l_dV_dx = m_work + local_size;
    double* l_dV_dy = m_work + local_size * 2;
    double* l_dV_dz = m_work + local_size * 3;
    if (is_spin_on) {
        for (int i = 0; i < local_size; ++i) {
            l_V_xc[i] = (ml_hr_Vx[i] + ml_hr_Vx_down[i] + ml_hr_Vc[i] + ml_hr_Vc_down[i]) / 2.0;
        }
    } else {
        for (int i = 0; i < local_size; ++i) {
            l_V_xc[i] = ml_hr_Vx[i] + ml_hr_Vc[i];
        }
    }

#ifdef GRAD_VXC_FFT
    auto hr_global_grid = ScaleRange(m_global_grid, m_HR_ratio_x, m_HR_ratio_y, m_HR_ratio_z);
    GradientDDMFFT_HR2(hr_grid, hr_global_grid, *m_ddm_exchanger, m_HR_fft_x, m_HR_fft_y, m_HR_fft_z, l_V_xc, l_dV_dx, l_dV_dy, l_dV_dz, -1.0, m_dx/(double)m_HR_ratio_x, m_dy / (double)m_HR_ratio_y, m_dz / (double)m_HR_ratio_z, false, m_HR_ratio_x, m_HR_ratio_y, m_HR_ratio_z);
    //GradientDDMFFT_HR(ml_grid, m_global_grid, *m_ddm_exchanger, m_HR_fft_x, m_HR_fft_y, m_HR_fft_z, l_V_xc, l_dV_dx, l_dV_dy, l_dV_dz, -1.0, m_dx, m_dy, m_dz, false, m_HR_ratio_x, m_HR_ratio_y, m_HR_ratio_z);
    watch.Record(71);
#else
    Gradient8th_ddm(hl_grid, l_dV_dx, l_dV_dy, l_dV_dz, l_V_xc, -1.0, m_dx/(double)m_HR_ratio_x, m_dy/(double)m_HR_ratio_y, m_dz/(double)m_HR_ratio_z);
    watch.Record(71);
#endif
    const int stride = 3;
    m_pp_integrator.InnerForChargePcc_HR(force, stride, l_dV_dx, ml_grid, m_nuclei, m_num_nuclei);
    m_pp_integrator.InnerForChargePcc_HR(force + 1, stride, l_dV_dy, ml_grid, m_nuclei, m_num_nuclei);
    m_pp_integrator.InnerForChargePcc_HR(force + 2, stride, l_dV_dz, ml_grid, m_nuclei, m_num_nuclei);
    watch.Record(72);

}
#endif

//mpi supported//
inline
double QUMASUN_BASE1::mGetForcePseudoNonlocal(double* force) {
	
	size_t local_size = ml_grid.Size3D();

	for (int i = 0; i < m_num_nuclei*3; ++i) {
		force[i] = 0;
	}


#ifdef FORCE_DIFF_PROJ
#ifdef GY_WITH_CUDA_OR_HIP
    const int bundle_width = num_solution;
#else
    const int bundle_width = NL_FORCE_BANDLE_WIDTH;
#endif

    double EV = 0.0;
    for (int sk = 0; sk < m_num_having_spin_kpoint; ++sk) {
        const double gx = (double)(ml_wave_set[sk].kpoint_x) * m_dkx;
        const double gy = (double)(ml_wave_set[sk].kpoint_y) * m_dky;
        const double gz = (double)(ml_wave_set[sk].kpoint_z) * m_dkz;


        for (int s = 0; s < num_solution; s += bundle_width) {
            const int min_bundle = std::min(bundle_width, num_solution - s);
            const double coef = 1.0;

            watch.Record(75);
            double pp_ene = m_pp_SvF.ForceNonlocal_bundle(force, &(ml_wave_set[sk].l_psi_set[s]), &(ml_wave_set[sk].occupancy[s]), min_bundle, ml_grid, m_nuclei, m_num_nuclei, sk, gx * coef, gy * coef, gz * coef, (std::byte*)m_work);
            watch.Record(76);
            EV += pp_ene;
        }
    }
#else
    const int bundle_width = NL_FORCE_BANDLE_WIDTH;
    std::vector<SoAComplex> psi_and_derivatives_xyz(bundle_width * 4);
    double* p_work = m_work;


    double EV = 0.0;
    for (int sk = 0; sk < m_num_having_spin_kpoint; ++sk) {
        const double gx = (double)(ml_wave_set[sk].kpoint_x) * m_dkx;
        const double gy = (double)(ml_wave_set[sk].kpoint_y) * m_dky;
        const double gz = (double)(ml_wave_set[sk].kpoint_z) * m_dkz;


        for (int s = 0; s < num_solution; s += bundle_width) {
            const int min_bundle = std::min(bundle_width, num_solution - s);
            const double coef = 1.0;

            for (int b = 0; b < min_bundle; ++b) {
                psi_and_derivatives_xyz[b * 4] = ml_wave_set[sk].l_psi_set[s + b];
                psi_and_derivatives_xyz[b * 4 + 1].re = p_work + local_size * (6 * b);
                psi_and_derivatives_xyz[b * 4 + 1].im = p_work + local_size * (6 * b + 1);
                psi_and_derivatives_xyz[b * 4 + 2].re = p_work + local_size * (6 * b + 2);
                psi_and_derivatives_xyz[b * 4 + 2].im = p_work + local_size * (6 * b + 3);
                psi_and_derivatives_xyz[b * 4 + 3].re = p_work + local_size * (6 * b + 4);
                psi_and_derivatives_xyz[b * 4 + 3].im = p_work + local_size * (6 * b + 5);

                Gradient8th_ddm(ml_grid, psi_and_derivatives_xyz[b * 4 + 1].re, psi_and_derivatives_xyz[b * 4 + 2].re, psi_and_derivatives_xyz[b * 4 + 3].re, ml_wave_set[sk].l_psi_set[s + b].re, coef, m_dx, m_dy, m_dz);
                Gradient8th_ddm(ml_grid, psi_and_derivatives_xyz[b * 4 + 1].im, psi_and_derivatives_xyz[b * 4 + 2].im, psi_and_derivatives_xyz[b * 4 + 3].im, ml_wave_set[sk].l_psi_set[s + b].im, coef, m_dx, m_dy, m_dz);
            }

            watch.Record(75);
            double pp_ene = m_pp_SvF.ForceNonlocal_bundle_withGradPsi(force, &psi_and_derivatives_xyz[0], &(ml_wave_set[sk].occupancy[s]), min_bundle, ml_grid, m_nuclei, m_num_nuclei, sk, gx * coef, gy * coef, gz * coef);
            watch.Record(76);
            EV += pp_ene;
        }
    }
#endif


    gyCheckError(gyGetLastError(), (std::string("DEV-ERROR") + std::to_string(__LINE__)).c_str());

	if (IsRoot(m_ddm_comm)) {
		double tot_EV = 0.0;
		MPI_Reduce(&EV, &tot_EV, 1, MPI_DOUBLE, MPI_SUM, m_root_id, m_same_ddm_place_comm);
        
		auto force_sum = std::make_unique<double[]>(m_num_nuclei * 3);
		MPI_Reduce(force, &force_sum[0], m_num_nuclei * 3, MPI_DOUBLE, MPI_SUM, m_root_id, m_same_ddm_place_comm);

		if (IsRoot(m_same_ddm_place_comm)) {
			for (int i = 0; i < m_num_nuclei * 3; ++i) {
				force[i] = force_sum[i];
			}
		}

		return tot_EV;
	} else {
		return 0.0;
	}
	
}


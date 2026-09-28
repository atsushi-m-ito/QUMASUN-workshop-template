#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "mpi_helper.h"
#include "qumasun_note.h"
#include "qumasun_symplectic.h"
#include "Grad_DDMFFT.h"

//#define TEST_F_NONLOCAL

inline
void QUMASUN_SYMPLECTIC::mInitializeElectrons() {
    //printf("[%d] test0\n", proc_id); fflush(stdout);
    mInitializeState();
    watch.Record(0);
    //printf("[%d] test1\n", proc_id); fflush(stdout);

    //mSetOccupancy();
    mSetOccupancyInitial();
    watch.Record(2);
    //printf("[%d] test2\n", proc_id); fflush(stdout);
    //MPI_Barrier(m_mpi_comm);
    //printf("[%d] test2-2\n", proc_id); fflush(stdout);

    //calculate electron density in real space///////////////
    mInitializeDensity();
    watch.Record(7);
    //MPI_Barrier(m_mpi_comm);
    //printf("[%d] test3-2\n", proc_id); fflush(stdout);

    //calculate potential in real space//////////////////////
    mSetPotentialVhart();
    watch.Record(4);

}


inline
void QUMASUN_SYMPLECTIC::MoveNuclei(int num_nuclei, const Nucleus* next_nucleis) {

    mMoveNuclei(num_nuclei, next_nucleis);

    mPrepareCore();
    watch.Record(1);
}


inline
void QUMASUN_SYMPLECTIC::Initialize(int num_nuclei, const Nucleus* next_nucleis) {

    PrintCondition();
    if (!mCheckConditions()) {
        return;
    }

    mInitializeElectrons();
    MPI_Barrier(m_mpi_comm);
    watch.Restart();

    mMoveNuclei(num_nuclei, next_nucleis);

    mPrepareCore();
    watch.Record(1);
    
}


inline
double QUMASUN_SYMPLECTIC::EvolveKinetic(double dt) {
#if 1
    m_Ekin = mEvolveK_FFT(dt);
#else
    mEvolveK4(dt);
#endif
    watch.Record(80);
    return m_Ekin;
}


inline
void QUMASUN_SYMPLECTIC::mProductExpToV(SoAComplex* l_psi_set, const OneComplex* exp_v) {

    const size_t local_size = ml_grid.Size3D();
    auto& l_psi = l_psi_set;

    for (int s = 0; s < num_solution; ++s) {

        auto psi_l_re = l_psi[s].re;
        auto psi_l_im = l_psi[s].im;

        for (size_t i = 0; i < local_size; ++i) {
            const double p_re = psi_l_re[i];
            const double p_im = psi_l_im[i];

            psi_l_re[i] = exp_v[i].r * p_re - exp_v[i].i * p_im;
            psi_l_im[i] = exp_v[i].r * p_im + exp_v[i].i * p_re;
        }
    }
    
    
}



inline
void QUMASUN_SYMPLECTIC::EvolvePotential1(int num_nuclei, double* nucl_F, double dt) {
    mSetDensityByPsi();
    watch.Record(3);

    //calculate potential in real space//
    mSetPotentialVhart();
    mSetPotentialVxc();
    mSetPotentialVtot();
    watch.Record(4);

    auto forces = std::make_unique<vec3d[]>(num_nuclei);
    auto forces_nn = std::make_unique<vec3d[]>(num_nuclei);
    auto forces_xc_pcc = std::make_unique<vec3d[]>(num_nuclei);

    mGetForceHartreeNuclRho_HR((double*)&forces[0]);
    mGetForceCoreCoreByVextNuclRho_HR((double*)&forces_nn[0]);
#ifdef USE_PCC_HR
    mGetForceXcPcc_HR((double*)&forces_xc_pcc[0]);
#else
    mGetForceXcPcc((double*)&forces_xc_pcc[0]);
#endif

    if (IsRoot(m_mpi_comm)) {
        for (int i = 0; i < num_nuclei; ++i) {
            forces[i] += forces_nn[i] + m_force_nn_TF[i] - m_force_nn_correction[i] + forces_xc_pcc[i];
        }

        for (int i = 0; i < num_nuclei; ++i) {
            nucl_F[i * 3] = forces[i].x;
            nucl_F[i * 3 + 1] = forces[i].y;
            nucl_F[i * 3 + 2] = forces[i].z;
        }
    }


    const size_t local_size = ml_grid.Size3D();
    auto exp_v = std::make_unique<OneComplex[]>(local_size);
    int current_up_down = 0;

    for (int sk = 0; sk < m_num_having_spin_kpoint; ++sk) {
        if (ml_wave_set[sk].spin == SPIN::UP) {

            if (current_up_down != 1) {                
                for (size_t i = 0; i < local_size; ++i) {
                    double V = ml_Vtot[i];
                    exp_v[i].r = cos(-dt * V);
                    exp_v[i].i = sin(-dt * V);
                }
                current_up_down = 1;
            }
        } else {
            if (current_up_down != 2) {
                for (size_t i = 0; i < local_size; ++i) {
                    double V = ml_Vtot_down[i];
                    exp_v[i].r = cos(-dt * V);
                    exp_v[i].i = sin(-dt * V);
                }
                current_up_down = 2;
            }
        }
        mProductExpToV(ml_wave_set[sk].l_psi_set, exp_v.get());
    }

    watch.Record(81);


}


//[deprecated]正式採用版はv2となった
inline
void QUMASUN_SYMPLECTIC::EvolvePotentialNonlocal(int num_nuclei, double* delta_P, double dt) {
    //delta_P is not force, because dt is included in the follwing 2nd term
    // exp(iL)P = P + ihbar<psi| ...   exp(-i dt M) ...|psi>
    //
    size_t local_size = ml_grid.Size3D();

    for (int i = 0; i < m_num_nuclei * 3; ++i) {
        delta_P[i] = 0;
    }

#ifdef TEST_F_NONLOCAL
    auto F_test = std::make_unique<double[]>(num_nuclei * 3);
    for (int i = 0; i < m_num_nuclei * 3; ++i) {
        F_test[i] = 0;
    }
#endif

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

#ifdef TEST_F_NONLOCAL
            double pp_ene1 = m_pp_SvF.ForceNonlocal_bundle_withGradPsi(&F_test[0], &psi_and_derivatives_xyz[0], &(ml_wave_set[sk].occupancy[s]), min_bundle, ml_grid, m_nuclei, m_num_nuclei, sk, gx * coef, gy * coef, gz * coef);
#endif

            double pp_ene = m_pp_SvF.SymplecticIntegration_bundle(dt, delta_P, &ml_wave_set[sk].l_psi_set[s], &psi_and_derivatives_xyz[0], &(ml_wave_set[sk].occupancy[s]), min_bundle, ml_grid, m_nuclei, m_num_nuclei, sk, gx * coef, gy * coef, gz * coef);
            watch.Record(76);
            //EV += pp_ene;
        }
    }

    if (IsRoot(m_ddm_comm)) {
        //double tot_EV = 0.0;
        //MPI_Reduce(&EV, &tot_EV, 1, MPI_DOUBLE, MPI_SUM, m_root_id, m_same_ddm_place_comm);


        auto force_sum = std::make_unique<double[]>(m_num_nuclei * 3);
        MPI_Reduce(delta_P, &force_sum[0], m_num_nuclei * 3, MPI_DOUBLE, MPI_SUM, m_root_id, m_same_ddm_place_comm);

        if (IsRoot(m_same_ddm_place_comm)) {
            for (int i = 0; i < m_num_nuclei * 3; ++i) {
                delta_P[i] = force_sum[i];
            }
        }       


#ifdef TEST_F_NONLOCAL
        MPI_Reduce(&F_test[0], &force_sum[0], m_num_nuclei * 3, MPI_DOUBLE, MPI_SUM, m_root_id, m_same_ddm_place_comm);

        if (IsRoot(m_same_ddm_place_comm)) {
            for (int n = 0; n < num_nuclei; ++n) {
                F_test[n * 3] = force_sum[n * 3];
                F_test[n * 3 + 1] = force_sum[n * 3 + 1];
                F_test[n * 3 + 2] = force_sum[n * 3 + 2];
                printf("FNL_r = %.10f, %.10f, %.10f\n", F_test[n * 3], F_test[n * 3 + 1], F_test[n * 3 + 2]);
                printf("FNL_s = %.10f, %.10f, %.10f\n", delta_P[n * 3] / dt, delta_P[n * 3 + 1] / dt, delta_P[n * 3 + 2] / dt);

            }
        }
        /*
        for (int n = 0; n < num_nuclei * 3; ++n) {
            delta_P[n] = F_test[n] * dt;
        }
        */
#endif      

    }

}


/*
* Symplectic Integratorでの正式採用版
* 
* 
*/
inline
void QUMASUN_SYMPLECTIC::EvolvePotentialNonlocal_v2(int num_nuclei, double* delta_P, double dt, int forward_or_backward) {
    //delta_P is not force, because dt is included in the follwing 2nd term
    // exp(iL)P = P + ihbar<psi| ...   exp(-i dt M) ...|psi>
    //
    size_t local_size = ml_grid.Size3D();

    for (int i = 0; i < m_num_nuclei * 3; ++i) {
        delta_P[i] = 0;
    }

#ifdef TEST_F_NONLOCAL
    auto F_test = std::make_unique<double[]>(num_nuclei * 3);
    for (int i = 0; i < m_num_nuclei * 3; ++i) {
        F_test[i] = 0;
    }
#endif

    
#ifdef FORCE_DIFF_PROJ
    const int bundle_width = num_solution;
#else
    const int bundle_width = num_solution; // NL_FORCE_BANDLE_WIDTH;
    std::vector<SoAComplex> psi_and_derivatives_xyz(bundle_width * 4);
    double* p_work = m_work;
#endif


    double EV = 0.0;
    for (int sk = 0; sk < m_num_having_spin_kpoint; ++sk) {
        const double gx = (double)(ml_wave_set[sk].kpoint_x) * m_dkx;
        const double gy = (double)(ml_wave_set[sk].kpoint_y) * m_dky;
        const double gz = (double)(ml_wave_set[sk].kpoint_z) * m_dkz;


        for (int s = 0; s < num_solution; s += bundle_width) {
            const int min_bundle = std::min(bundle_width, num_solution - s);
            const double coef = 1.0;

#if 1 && defined(FORCE_DIFF_PROJ)
            //for bench: case that all nuclei are not overlap each other.
            {
                
                double pp_ene = m_pp_SvF.SymplecticIntegration_bundle2(dt, delta_P, &ml_wave_set[sk].l_psi_set[s], &ml_wave_set[sk].l_psi_set[s], &(ml_wave_set[sk].occupancy[s]), min_bundle, ml_grid, m_nuclei, num_nuclei, forward_or_backward, sk, gx * coef, gy * coef, gz * coef);
                watch.Record(76);
                //EV += pp_ene;
            }
#else
            for (int ni = 0; ni < num_nuclei; ++ni) {

#if defined(TEST_F_NONLOCAL) || !defined(FORCE_DIFF_PROJ)
                for (int b = 0; b < min_bundle; ++b) {
                    psi_and_derivatives_xyz[b * 4] = ml_wave_set[sk].l_psi_set[s + b];
                    psi_and_derivatives_xyz[b * 4 + 1].re = p_work + local_size * (6 * b);
                    psi_and_derivatives_xyz[b * 4 + 1].im = p_work + local_size * (6 * b + 1);
                    psi_and_derivatives_xyz[b * 4 + 2].re = p_work + local_size * (6 * b + 2);
                    psi_and_derivatives_xyz[b * 4 + 2].im = p_work + local_size * (6 * b + 3);
                    psi_and_derivatives_xyz[b * 4 + 3].re = p_work + local_size * (6 * b + 4);
                    psi_and_derivatives_xyz[b * 4 + 3].im = p_work + local_size * (6 * b + 5);

#if 0
                    GradientDDMFFT(ml_grid, m_global_grid, *m_ddm_exchanger, m_fft_x, m_fft_y, m_fft_z,
                        //p_work + local_size * 6 * min_bundle,
                        (double*)&psi_and_derivatives_xyz[b * 4], (double*)&psi_and_derivatives_xyz[b * 4 + 1], (double*)&psi_and_derivatives_xyz[b * 4 + 2], (double*)&psi_and_derivatives_xyz[b * 4 + 3],
                        m_dx, m_dy, m_dz, true);
#else
                    Gradient8th_ddm(ml_grid, psi_and_derivatives_xyz[b * 4 + 1].re, psi_and_derivatives_xyz[b * 4 + 2].re, psi_and_derivatives_xyz[b * 4 + 3].re, ml_wave_set[sk].l_psi_set[s + b].re, coef, m_dx, m_dy, m_dz);
                    Gradient8th_ddm(ml_grid, psi_and_derivatives_xyz[b * 4 + 1].im, psi_and_derivatives_xyz[b * 4 + 2].im, psi_and_derivatives_xyz[b * 4 + 3].im, ml_wave_set[sk].l_psi_set[s + b].im, coef, m_dx, m_dy, m_dz);
#endif
                }

                watch.Record(75);
#endif

#ifdef TEST_F_NONLOCAL
                if (ni == 0) {
                    double pp_ene1 = m_pp_SvF.ForceNonlocal_bundle(&F_test[0], &psi_and_derivatives_xyz[0], &(ml_wave_set[sk].occupancy[s]), min_bundle, ml_grid, m_nuclei, m_num_nuclei, sk, gx * coef, gy * coef, gz * coef);
                }
#endif
                double pp_ene = m_pp_SvF.SymplecticIntegration_bundle2(dt, delta_P, &ml_wave_set[sk].l_psi_set[s], &psi_and_derivatives_xyz[0], &(ml_wave_set[sk].occupancy[s]), min_bundle, ml_grid, m_nuclei, num_nuclei, forward_or_backward, sk, gx * coef, gy * coef, gz * coef);
                watch.Record(76);
                //EV += pp_ene;
            }
#endif
        }
    }

#if 1
    {

        auto force_sum = std::make_unique<double[]>(m_num_nuclei * 3);
        MPI_Reduce(delta_P, &force_sum[0], m_num_nuclei * 3, MPI_DOUBLE, MPI_SUM, m_root_id, m_mpi_comm);

        if (IsRoot(m_mpi_comm)) {
            for (int i = 0; i < m_num_nuclei * 3; ++i) {
                delta_P[i] = force_sum[i];
            }
        }
    }

#else
    if (IsRoot(m_ddm_comm)) {
        //double tot_EV = 0.0;
        //MPI_Reduce(&EV, &tot_EV, 1, MPI_DOUBLE, MPI_SUM, m_root_id, m_same_ddm_place_comm);


        auto force_sum = std::make_unique<double[]>(m_num_nuclei * 3);
        MPI_Reduce(delta_P, &force_sum[0], m_num_nuclei * 3, MPI_DOUBLE, MPI_SUM, m_root_id, m_same_ddm_place_comm);

        if (IsRoot(m_same_ddm_place_comm)) {
            for (int i = 0; i < m_num_nuclei * 3; ++i) {
                delta_P[i] = force_sum[i];
            }
        }


#ifdef TEST_F_NONLOCAL
        MPI_Reduce(&F_test[0], &force_sum[0], m_num_nuclei * 3, MPI_DOUBLE, MPI_SUM, m_root_id, m_same_ddm_place_comm);

        if (IsRoot(m_same_ddm_place_comm)) {
            for (int n = 0; n < num_nuclei; ++n) {
                F_test[n * 3] = force_sum[n * 3];
                F_test[n * 3 + 1] = force_sum[n * 3 + 1];
                F_test[n * 3 + 2] = force_sum[n * 3 + 2];
                printf("FNL_r = %.10f, %.10f, %.10f\n", F_test[n * 3], F_test[n * 3 + 1], F_test[n * 3 + 2]);
                printf("FNL_s = %.10f, %.10f, %.10f\n", delta_P[n * 3] / dt, delta_P[n * 3 + 1] / dt, delta_P[n * 3 + 2] / dt);

            }
        }
        /*
        for (int n = 0; n < num_nuclei * 3; ++n) {
            delta_P[n] = F_test[n] * dt;
        }
        */
#endif      

    }
#endif
}


//[deprecated]
inline
void QUMASUN_SYMPLECTIC::EvolvePotentialAll(int num_nuclei, double* nucl_F, double dt) {
    mSetDensityByPsi();
    watch.Record(3);

    //calculate potential in real space//
    mSetPotentialVhart();
    mSetPotentialVxc();
    mSetPotentialVtot();

    auto forces = std::make_unique<vec3d[]>(num_nuclei);
    auto forces_nn = std::make_unique<vec3d[]>(num_nuclei);
    auto forces_xc_pcc = std::make_unique<vec3d[]>(num_nuclei);

    mGetForceHartreeNuclRho_HR((double*)&forces[0]);
    mGetForceCoreCoreByVextNuclRho_HR((double*)&forces_nn[0]);
#ifdef USE_PCC_HR
    mGetForceXcPcc_HR((double*)&forces_xc_pcc[0]);
#else
    mGetForceXcPcc((double*)&forces_xc_pcc[0]);
#endif

    if (IsRoot(m_mpi_comm)) {
        for (int i = 0; i < num_nuclei; ++i) {
            forces[i] += forces_nn[i] + m_force_nn_TF[i] - m_force_nn_correction[i] + forces_xc_pcc[i];
        }

        for (int i = 0; i < num_nuclei; ++i) {
            nucl_F[i * 3] = forces[i].x;
            nucl_F[i * 3 + 1] = forces[i].y;
            nucl_F[i * 3 + 2] = forces[i].z;
        }
    }


    auto delta_P = std::make_unique<double[]>(num_nuclei * 3);
    for (int i = 0; i < m_num_nuclei * 3; ++i) {
        delta_P[i] = 0;
    }

#ifdef TEST_F_NONLOCAL
    auto F_test = std::make_unique<double[]>(num_nuclei * 3);
    for (int i = 0; i < m_num_nuclei * 3; ++i) {
        F_test[i] = 0;
    }
#endif


    const int bundle_width = NL_FORCE_BANDLE_WIDTH;
    auto psi_and_derivatives_xyz = std::make_unique<SoAComplex[]>(bundle_width * 4);
    double* p_work = m_work;

    const size_t local_size = ml_grid.Size3D();
    auto exp_v = std::make_unique<OneComplex[]>(local_size);
    int current_up_down = 0;

    for (int sk = 0; sk < m_num_having_spin_kpoint; ++sk) {

#ifndef IGNORE_NONLOCAL
        const double gx = (double)(ml_wave_set[sk].kpoint_x) * m_dkx;
        const double gy = (double)(ml_wave_set[sk].kpoint_y) * m_dky;
        const double gz = (double)(ml_wave_set[sk].kpoint_z) * m_dkz;


        for (int s = 0; s < num_solution; s += bundle_width) {
            const int min_bundle = std::min(bundle_width, num_solution - s);
            const double coef = 1.0;

            for (int b = 0; b < min_bundle; ++b) {
                psi_and_derivatives_xyz[b * 4].re = p_work + local_size * (8 * b);
                psi_and_derivatives_xyz[b * 4].im = p_work + local_size * (8 * b+1);
                psi_and_derivatives_xyz[b * 4 + 1].re = p_work + local_size * (8 * b+2);
                psi_and_derivatives_xyz[b * 4 + 1].im = p_work + local_size * (8 * b + 3);
                psi_and_derivatives_xyz[b * 4 + 2].re = p_work + local_size * (8 * b + 4);
                psi_and_derivatives_xyz[b * 4 + 2].im = p_work + local_size * (8 * b + 5);
                psi_and_derivatives_xyz[b * 4 + 3].re = p_work + local_size * (8 * b + 6);
                psi_and_derivatives_xyz[b * 4 + 3].im = p_work + local_size * (8 * b + 7);


                memcpy(psi_and_derivatives_xyz[b * 4].re, ml_wave_set[sk].l_psi_set[s + b].re, sizeof(double) * local_size);
                memcpy(psi_and_derivatives_xyz[b * 4].im, ml_wave_set[sk].l_psi_set[s + b].im, sizeof(double) * local_size);

                Gradient8th_ddm(ml_grid, psi_and_derivatives_xyz[b * 4 + 1].re, psi_and_derivatives_xyz[b * 4 + 2].re, psi_and_derivatives_xyz[b * 4 + 3].re, ml_wave_set[sk].l_psi_set[s + b].re, coef, m_dx, m_dy, m_dz);
                Gradient8th_ddm(ml_grid, psi_and_derivatives_xyz[b * 4 + 1].im, psi_and_derivatives_xyz[b * 4 + 2].im, psi_and_derivatives_xyz[b * 4 + 3].im, ml_wave_set[sk].l_psi_set[s + b].im, coef, m_dx, m_dy, m_dz);
            }

            watch.Record(75);

#ifdef TEST_F_NONLOCAL
            double pp_ene1 = m_pp_SvF.ForceNonlocal_bundle(&F_test[0], &psi_and_derivatives_xyz[0], &(ml_wave_set[sk].occupancy[s]), min_bundle, ml_grid, m_nuclei, m_num_nuclei, sk, gx * coef, gy * coef, gz * coef);
#endif

            double pp_ene = m_pp_SvF.SymplecticIntegration_bundle(dt, &delta_P[0], &(ml_wave_set[sk].l_psi_set[s]), &psi_and_derivatives_xyz[0], &(ml_wave_set[sk].occupancy[s]), min_bundle, ml_grid, m_nuclei, m_num_nuclei, sk, gx * coef, gy * coef, gz * coef);
            watch.Record(76);
            //EV += pp_ene;
        }
#endif   // IGNORE_NONLOCAL

        if (ml_wave_set[sk].spin == SPIN::UP) {

            if (current_up_down != 1) {
                for (size_t i = 0; i < local_size; ++i) {
                    double V = ml_Vtot[i];
                    exp_v[i].r = cos(-dt * V);
                    exp_v[i].i = sin(-dt * V);
                }
                current_up_down = 1;
            }
        } else {
            if (current_up_down != 2) {
                for (size_t i = 0; i < local_size; ++i) {
                    double V = ml_Vtot_down[i];
                    exp_v[i].r = cos(-dt * V);
                    exp_v[i].i = sin(-dt * V);
                }
                current_up_down = 2;
            }
        }
        mProductExpToV(ml_wave_set[sk].l_psi_set, exp_v.get());
    }

    watch.Record(81);

#ifndef IGNORE_NONLOCAL
    if (IsRoot(m_ddm_comm)) {
        //double tot_EV = 0.0;
        //MPI_Reduce(&EV, &tot_EV, 1, MPI_DOUBLE, MPI_SUM, m_root_id, m_same_ddm_place_comm);


        auto force_sum = std::make_unique<double[]>(m_num_nuclei * 3);
        MPI_Reduce(&delta_P[0], &force_sum[0], m_num_nuclei * 3, MPI_DOUBLE, MPI_SUM, m_root_id, m_same_ddm_place_comm);

        if (IsRoot(m_same_ddm_place_comm)) {
            for (int i = 0; i < m_num_nuclei * 3; ++i) {
                delta_P[i] = force_sum[i];
            }
        }


#ifdef TEST_F_NONLOCAL
        MPI_Reduce(&F_test[0], &force_sum[0], m_num_nuclei * 3, MPI_DOUBLE, MPI_SUM, m_root_id, m_same_ddm_place_comm);

        if (IsRoot(m_same_ddm_place_comm)) {
            for (int n = 0; n < num_nuclei; ++n) {
                F_test[n * 3] = force_sum[n * 3];
                F_test[n * 3 + 1] = force_sum[n * 3 + 1];
                F_test[n * 3 + 2] = force_sum[n * 3 + 2];
                printf("FNL_r = %.10f, %.10f, %.10f\n", F_test[n * 3], F_test[n * 3 + 1], F_test[n * 3 + 2]);
                printf("FNL_s = %.10f, %.10f, %.10f\n", delta_P[n * 3] / dt, delta_P[n * 3 + 1] / dt, delta_P[n * 3 + 2] / dt);

            }
        }
        /*
        for (int n = 0; n < num_nuclei * 3; ++n) {
            delta_P[n] = F_test[n] * dt;
        }
        */
#endif      

        if (IsRoot(m_same_ddm_place_comm)) {
            for (int i = 0; i < num_nuclei; ++i) {
                nucl_F[i * 3] += delta_P[i*3]/dt;
                nucl_F[i * 3 + 1] += delta_P[i * 3+1] / dt;
                nucl_F[i * 3 + 2] += delta_P[i * 3+2] / dt;
            }
        }

    }
#endif
}



inline
double QUMASUN_SYMPLECTIC::GetEnergy(bool use_kinetic_energy_evolveK) {
    mSetDensityByPsi();
    watch.Record(3);
    //calculate potential in real space//
    mSetPotentialVhart();
    mSetPotentialVxc();//GGAではここでエネルギーも計算するので必要//
    watch.Record(4);
    
    if (!use_kinetic_energy_evolveK) {
        //運動エネルギーの計算//
        //時間発展の初回ステップだけここを通る//
        //2ステップ目以降は運動エネルギー項の発展時に計算済み//
        m_Ekin = mGetEnergyKineticKspace();


    }   
    //どちらにせよ運動エネルギーは計算済みとなるので、
    //mGetTotalEnergy内では運動エネルギー項は計算しなくてよいので第一引数はfalse//
    double E_tot = mGetTotalEnergy(false, false);
    watch.Record(6);
    return E_tot;
}

//This is called only in test mode.
//In genuin symplectic integrator, this function should not be called.//
double QUMASUN_SYMPLECTIC::GetForce(vec3d* forces) {

    const bool is_root_global = IsRoot(m_mpi_comm);

    watch.Restart();


    //calculate potential in real space//
    mSetPotentialVhart();
    mSetPotentialVxc();//GGAではここでエネル
    watch.Record(4);
    double E_tot = mGetTotalEnergy(true, true);
    watch.Record(6);

    mGetForce(true);
    watch.Record(8);
    
    for (int i = 0; i < m_num_nuclei; ++i) {
        forces[i] = m_nucl_forces[i];
    }

    return E_tot;
}


#pragma once
//#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <memory>
#include "qumasun_base1.h"
#include "Vxc.h"
#include "xc_gga_pbe.h"
#include "GridGradient8th.h"
#include "nucleus.h"
#include "convertHR.h"
#include "check_field.h"

//#define USE_R2C_C2R

#ifdef DDM_FFT




/*
* V_hartを計算する
*
* V_extはmPrepareCoreで最初に一度だけ計算
* 
* Up-convertすることで、ml_hr_Vhartも取得する
* オプションによっては ml_hr_rhoも取得できる
*/
inline
void QUMASUN_BASE1::mSetPotentialVhart() {

    //gather rho in order to do FFT in root proc.
    if (IsRoot(m_same_ddm_place_comm)) {

        watch.Record(4);


#ifdef GRAD_VHART_FFT
#if 0  //def USE_PCC_HR
        mPoissonDDMFFT_Upconvert(ml_rho, ml_hr_Vhart, ml_hr_dVhart_dx, ml_hr_dVhart_dy, ml_hr_dVhart_dz, ml_hr_rho);
#else
        mPoissonDDMFFT_Upconvert(ml_rho, ml_hr_Vhart, ml_hr_dVhart_dx, ml_hr_dVhart_dy, ml_hr_dVhart_dz, nullptr);
#endif
#else
        mPoissonDDMFFT_Upconvert(ml_rho, ml_hr_Vhart, nullptr, nullptr, nullptr, nullptr);
#endif

#ifdef _DEBUG
        printf("[%d]after mPoissonDDMFFT_Upconvert:\n", GetProcessID(m_ddm_comm));
        fflush(stdout);
        MPI_Barrier(m_ddm_comm);
#endif

        auto& range = ml_grid;
        const int local_size_x = range.end_x - range.begin_x;
        const int local_size_y = range.end_y - range.begin_y;
        const int local_size_z = range.end_z - range.begin_z;

#ifdef TEST_DOWNCONVERT_POTENTIAL
        mDDMFFT_Downconvert(ml_hr_Vhart, ml_Vhart);
#else
        //元の電子密度がupconvertしたものだから、そもそもVhartも高周波数成分を持っておらず、これで問題ないと思われる//
        DownConvert_Realspace(ml_Vhart, local_size_x, local_size_y, local_size_z, ml_hr_Vhart, m_HR_ratio_x, m_HR_ratio_y, m_HR_ratio_z);
#endif
        watch.Record(57);

#ifdef _DEBUG
        printf("[%d]after DownConvert_Realspace:\n", GetProcessID(m_ddm_comm));
        fflush(stdout);
        MPI_Barrier(m_ddm_comm);
#endif


        /*
        {
            double diff = 0.0;
            for (int iz = 0; iz < m_size_z; ++iz) {
                for (int iy = 0; iy < m_size_y; ++iy) {
                    for (int ix = 0; ix < m_size_x; ++ix) {
                        int64_t i = ix + m_size_x * (iy + m_size_y * iz);
                        int64_t oi = m_HR_ratio_x * (ix + m_size_x * (m_HR_ratio_y * (iy + m_size_y * m_HR_ratio_z * iz)));
                        //int64_t oi = (ix + hr_ratio_x * size_x * (iy + hr_ratio_y * size_y * iz));
                        diff += (ml_hr_Vhart[oi] - ml_Vhart[i]) * (ml_hr_Vhart[oi] - ml_Vhart[i]);
                    }
                }
            }
            double diff_tot = 0.0;
            MPI_Reduce(&diff, &diff_tot, 1, MPI_DOUBLE, MPI_SUM, 0, m_ddm_comm);
            printf("diff_Vhart(HR) = %f\n", diff_tot * m_dx * m_dy * m_dz);
            watch.Record(57);
        }


#ifdef _DEBUG
        printf("[%d]after mSetPotentialVhart:\n", GetProcessID(m_ddm_comm));
        fflush(stdout);
        MPI_Barrier(m_ddm_comm);
#endif
*/
    }


}



#else
/*
* V_hartを計算する
* 
* V_extはmPrepareCoreで最初に一度だけ計算
*/
inline
void QUMASUN_BASE1::mSetPotentialVhart() {

    //gather rho in order to do FFT in root proc.
    if (IsRoot(m_same_ddm_place_comm)) {
        mGatherField(m_rho.Pointer(), ml_rho);
    }

	if (IsRoot(m_mpi_comm)) {

		if (m_hamiltonian_type == HAMILTONIAN::KohnSham_PP) {//DFT//
        
#if 1
            watch.Record(4);
            

            
            auto* rho_comp = m_fftw->GetBuffer();
            {
                const int64_t size_3d = m_size_3d;

                for (size_t i = 0; i < size_3d; ++i) {
                    rho_comp[i][0] = m_rho[i];
                    rho_comp[i][1] = 0.0;
                }
            }

            auto rhok = m_fftw->ForwardDirect(rho_comp);
            watch.Record(50);
            
            //calculate HR rho
            {
                //OneComplex* rhok = (OneComplex*)rhok;
                OneComplex* hr_rhok = (OneComplex*)(m_HR_fftw->GetBuffer());
                UpConvert_Kspace((OneComplex*)rhok, m_size_x, m_size_y, m_size_z, hr_rhok, m_HR_ratio_x, m_HR_ratio_y, m_HR_ratio_z);
                watch.Record(51);
                const size_t hr_size_3d = m_size_3d * m_HR_ratio_x * m_HR_ratio_y * m_HR_ratio_z;
                //OneComplex* hr_rho_c = hr_rhok + hr_size_3d;
                //IFFT_3D(hr_rho_c, hr_rhok, m_size_x * m_HR_ratio_x, m_size_y * m_HR_ratio_y, m_size_z * m_HR_ratio_z);
                auto* res = m_HR_fftw->BackwardDirect(hr_rhok);
                OneComplex* hr_rho_c = (OneComplex*)res;
                watch.Record(52);
                double sum_rho = 0.0;
                const double invN = 1.0 / (double)m_size_3d;
                for (size_t i = 0; i < hr_size_3d; ++i) {
                    m_hr_rho[i] = hr_rho_c[i].r * invN;
                    sum_rho += hr_rho_c[i].r * invN;
                }
                printf("Rho(HR) = %f\n", sum_rho * m_dx * m_dy * m_dz / (double)(m_HR_ratio_x * m_HR_ratio_y * m_HR_ratio_z));
                watch.Record(53);

                double diff = 0.0;
                for (int iz = 0; iz < m_size_z; ++iz) {
                    for (int iy = 0; iy < m_size_y; ++iy) {
                        for (int ix = 0; ix < m_size_x; ++ix) {
                            int64_t i = ix + m_size_x * (iy + m_size_y * iz);
                            int64_t oi = m_HR_ratio_x * (ix + m_size_x * (m_HR_ratio_y * (iy + m_size_y * m_HR_ratio_z * iz)));
                            //int64_t oi = (ix + hr_ratio_x * size_x * (iy + hr_ratio_y * size_y * iz));
                            diff += (m_hr_rho[oi] - m_rho[i]) * (m_hr_rho[oi] - m_rho[i]);
                        }
                    }
                }
                printf("diff_Rho(HR) = %f\n", diff * m_dx * m_dy * m_dz);
                watch.Record(54);
            }

            SolvePoissonInKspace(rhok, m_size_x, m_size_y, m_size_z, m_dx, m_dy, m_dz);
            watch.Record(50);

            //calculate HR Vhart
            {
                OneComplex* vhartk = (OneComplex*)(rhok);
                OneComplex* hr_vhartk = (OneComplex*)(m_HR_fftw->GetBuffer());
                UpConvert_Kspace(vhartk, m_size_x, m_size_y, m_size_z, hr_vhartk, m_HR_ratio_x, m_HR_ratio_y, m_HR_ratio_z);
                watch.Record(51);
                const size_t hr_size_3d = m_size_3d * m_HR_ratio_x * m_HR_ratio_y * m_HR_ratio_z;
                //OneComplex* hr_vhart_c = hr_vhartk + hr_size_3d;
                //IFFT_3D(hr_vhart_c, hr_vhartk, m_size_x * m_HR_ratio_x, m_size_y * m_HR_ratio_y, m_size_z * m_HR_ratio_z);
                auto* res = m_HR_fftw->BackwardDirect(hr_vhartk);
                OneComplex* hr_vhart_c = (OneComplex*)res;
                watch.Record(52);
                //double sum_rho = 0.0;
                //const double scaling_factor = (double)(m_HR_ratio_x * m_HR_ratio_y * m_HR_ratio_z);
                const double scaling_factor = 1.0 / (double)m_size_3d;
                for (size_t i = 0; i < hr_size_3d; ++i) {
                    m_hr_Vhart[i] = hr_vhart_c[i].r * scaling_factor;
                }
                watch.Record(53);

            }


#ifdef GRAD_VHART_FFT
            {
                const int64_t size_3d_c = (m_size_x) * m_size_y * m_size_z;
                const int64_t hr_size_3d = m_size_3d * m_HR_ratio_x * m_HR_ratio_y * m_HR_ratio_z;
                auto vhartk = std::make_unique<OneComplex[]>(size_3d_c);
                const double scaling_factor = 1.0 / (double)m_size_3d;
                for (int64_t i = 0; i < size_3d_c; ++i) {
                    vhartk[i].r = rhok[i][0] * scaling_factor;
                    vhartk[i].i = rhok[i][1] * scaling_factor;
                }

                auto dV_dxyz_k = std::make_unique<OneComplex[]>(size_3d_c);
                OneComplex* hr_dv_dxyz_k = (OneComplex*)(m_HR_fftw->GetBuffer());
                double* hr_dVhart_dx = m_work;
                double* hr_dVhart_dy = hr_dVhart_dx + hr_size_3d;
                double* hr_dVhart_dz = hr_dVhart_dx + hr_size_3d * 2;

                GradientXInKspace((fftw_complex*)&dV_dxyz_k[0], (fftw_complex*)&vhartk[0], m_size_x, m_size_y, m_size_z, m_dx, m_dy, m_dz);
                UpConvert_Kspace(&dV_dxyz_k[0], m_size_x, m_size_y, m_size_z, hr_dv_dxyz_k, m_HR_ratio_x, m_HR_ratio_y, m_HR_ratio_z);
                m_HR_fftw->BackwardDirect(hr_dv_dxyz_k);
                for (int i = 0; i < hr_size_3d; ++i) {
                    hr_dVhart_dx[i] = hr_dv_dxyz_k[i].r;
                }

                GradientYInKspace((fftw_complex*)&dV_dxyz_k[0], (fftw_complex*)&vhartk[0], m_size_x, m_size_y, m_size_z, m_dx, m_dy, m_dz);
                UpConvert_Kspace(&dV_dxyz_k[0], m_size_x, m_size_y, m_size_z, hr_dv_dxyz_k, m_HR_ratio_x, m_HR_ratio_y, m_HR_ratio_z);
                m_HR_fftw->BackwardDirect(hr_dv_dxyz_k);
                for (int i = 0; i < hr_size_3d; ++i) {
                    hr_dVhart_dy[i] = hr_dv_dxyz_k[i].r;
                }
                
                GradientZInKspace((fftw_complex*)&dV_dxyz_k[0], (fftw_complex*)&vhartk[0], m_size_x, m_size_y, m_size_z, m_dx, m_dy, m_dz);
                UpConvert_Kspace(&dV_dxyz_k[0], m_size_x, m_size_y, m_size_z, hr_dv_dxyz_k, m_HR_ratio_x, m_HR_ratio_y, m_HR_ratio_z);
                m_HR_fftw->BackwardDirect(hr_dv_dxyz_k);
                for (int i = 0; i < hr_size_3d; ++i) {
                    hr_dVhart_dz[i] = hr_dv_dxyz_k[i].r;
                }
                

            }
#endif

            auto Vh = m_fftw->BackwardDirect(rhok);
            const double invN = 1.0 / (double)(m_size_3d);
            for (size_t i = 0; i < m_size_3d; ++i) {
                m_Vhart[i] = Vh[i][0] * invN;
            }
            watch.Record(50);

            {
                double diff = 0.0;
                for (int iz = 0; iz < m_size_z; ++iz) {
                    for (int iy = 0; iy < m_size_y; ++iy) {
                        for (int ix = 0; ix < m_size_x; ++ix) {
                            int64_t i = ix + m_size_x * (iy + m_size_y * iz);
                            int64_t oi = m_HR_ratio_x * (ix + m_size_x * (m_HR_ratio_y * (iy + m_size_y * m_HR_ratio_z * iz)));
                            //int64_t oi = (ix + hr_ratio_x * size_x * (iy + hr_ratio_y * size_y * iz));
                            diff += (m_hr_Vhart[oi] - m_Vhart[i]) * (m_hr_Vhart[oi] - m_Vhart[i]);
                        }
                    }
                }
                printf("diff_Vhart(HR) = %f\n", diff * m_dx * m_dy * m_dz);
                watch.Record(54);
            }

#else
            watch.Record(4);
            SetPotentialByPoissonFFT_keepRhok(m_Vhart, m_rho, m_size_x, m_size_y, m_size_z, m_dx, m_dy, m_dz, m_work);
            watch.Record(50);
            //calculate HR rho
            {
                OneComplex* rhok = (OneComplex*)m_work;
                OneComplex* hr_rhok = (OneComplex*)(m_work + m_size_3d * 4);
                UpConvert_Kspace(rhok, m_size_x, m_size_y, m_size_z, hr_rhok, m_HR_ratio_x, m_HR_ratio_y, m_HR_ratio_z);
                watch.Record(51);
                const size_t hr_size_3d = m_size_3d * m_HR_ratio_x * m_HR_ratio_y * m_HR_ratio_z;
                OneComplex* hr_rho_c = hr_rhok + hr_size_3d;
                IFFT_3D(hr_rho_c, hr_rhok, m_size_x * m_HR_ratio_x, m_size_y * m_HR_ratio_y, m_size_z * m_HR_ratio_z);
                watch.Record(52);
                double sum_rho = 0.0;
                const double scaling_factor = (double)(m_HR_ratio_x * m_HR_ratio_y * m_HR_ratio_z);
                for (size_t i = 0; i < hr_size_3d; ++i) {
                    m_hr_rho[i] = hr_rho_c[i].r * scaling_factor;
                    sum_rho += hr_rho_c[i].r * scaling_factor;
                }
                printf("Rho(HR) = %f\n", sum_rho * m_dx * m_dy * m_dz / (double)(m_HR_ratio_x * m_HR_ratio_y * m_HR_ratio_z));
                watch.Record(53);

                double diff = 0.0;
                for (int iz = 0; iz < m_size_z; ++iz) {
                    for (int iy = 0; iy < m_size_y; ++iy) {
                        for (int ix = 0; ix < m_size_x; ++ix) {
                            int64_t i = ix + m_size_x * (iy + m_size_y * iz);
                            int64_t oi = m_HR_ratio_x * (ix + m_size_x * (m_HR_ratio_y * (iy + m_size_y * m_HR_ratio_z * iz)));
                            //int64_t oi = (ix + hr_ratio_x * size_x * (iy + hr_ratio_y * size_y * iz));
                            diff += (m_hr_rho[oi] - m_rho[i]) * (m_hr_rho[oi] - m_rho[i]);
                        }
                    }
                }
                printf("diff_Rho(HR) = %f\n", diff * m_dx * m_dy * m_dz);
                watch.Record(54);
            }


            //calculate HR Vhart
            {
                OneComplex* vhartk = (OneComplex*)(m_work + m_size_3d * 2);
                OneComplex* hr_vhartk = (OneComplex*)(m_work + m_size_3d * 4);
                UpConvert_Kspace(vhartk, m_size_x, m_size_y, m_size_z, hr_vhartk, m_HR_ratio_x, m_HR_ratio_y, m_HR_ratio_z);
                watch.Record(51);
                const size_t hr_size_3d = m_size_3d * m_HR_ratio_x * m_HR_ratio_y * m_HR_ratio_z;
                OneComplex* hr_vhart_c = hr_vhartk + hr_size_3d;
                IFFT_3D(hr_vhart_c, hr_vhartk, m_size_x * m_HR_ratio_x, m_size_y * m_HR_ratio_y, m_size_z * m_HR_ratio_z);
                watch.Record(52);
                //double sum_rho = 0.0;
                const double scaling_factor = (double)(m_HR_ratio_x * m_HR_ratio_y * m_HR_ratio_z);
                for (size_t i = 0; i < hr_size_3d; ++i) {
                    m_hr_Vhart[i] = hr_vhart_c[i].r * scaling_factor;                    
                }
                watch.Record(53);

                double diff = 0.0;
                for (int iz = 0; iz < m_size_z; ++iz) {
                    for (int iy = 0; iy < m_size_y; ++iy) {
                        for (int ix = 0; ix < m_size_x; ++ix) {
                            int64_t i = ix + m_size_x * (iy + m_size_y * iz);
                            int64_t oi = m_HR_ratio_x * (ix + m_size_x * (m_HR_ratio_y * (iy + m_size_y * m_HR_ratio_z * iz)));
                            //int64_t oi = (ix + hr_ratio_x * size_x * (iy + hr_ratio_y * size_y * iz));
                            diff += (m_hr_Vhart[oi] - m_Vhart[i]) * (m_hr_Vhart[oi] - m_Vhart[i]);
                        }
                    }
                }
                printf("diff_Vhart(HR) = %f\n", diff * m_dx * m_dy * m_dz);
                watch.Record(54);
            }
#endif

        

		} else {//KohnSham_AE//
           
		}
	}

    if (m_hamiltonian_type == HAMILTONIAN::KohnSham_PP) {//DFT//
        if (IsRoot(m_same_ddm_place_comm)) {
            
            mScatterField(ml_Vhart, m_Vhart);
            mScatterField_HR(ml_hr_rho, m_hr_rho);
            mScatterField_HR(ml_hr_Vhart, m_hr_Vhart);

#ifdef GRAD_VHART_FFT
            const int64_t hr_size_3d = m_size_3d * m_HR_ratio_x * m_HR_ratio_y * m_HR_ratio_z;
            double* hr_dVhart_dx = m_work;
            double* hr_dVhart_dy = hr_dVhart_dx + hr_size_3d;
            double* hr_dVhart_dz = hr_dVhart_dx + hr_size_3d * 2;
            mScatterField_HR(ml_hr_dVhart_dx, hr_dVhart_dx);
            mScatterField_HR(ml_hr_dVhart_dy, hr_dVhart_dy);
            mScatterField_HR(ml_hr_dVhart_dz, hr_dVhart_dz);
#endif
            watch.Record(57);

        }
    }

}
#endif


inline
void QUMASUN_BASE1::mSetPotentialVxc() {

    //Vxc by electronic density and PCC charge (depending on nuclei position)/////////////////////////////////////////////////
    if (IsRoot(m_same_ddm_place_comm)) {

        if (m_hamiltonian_type == HAMILTONIAN::KohnSham_PP) {//DFT//
            if (m_xc_type == XC_TYPE::GGA) {

                const int64_t local_size = ml_grid.Size3D();

                //E_xは加法則が成り立つ(E_x(up,down) = [E_x(2*up) + E_x(2*down)]/2.0
                auto Exchange_GGA = [](GridRangeMPI& l_grid, const double* l_rho_sigma, const double* drho_dx, const double* drho_dy, const double* drho_dz, double* l_Vx, double* work_buffer, double dx, double dy, double dz) {
                    const int64_t local_size = l_grid.Size3D();

                    double* sq_grad_rho = work_buffer;


                    for (int64_t i = 0; i < local_size; ++i) {
                        sq_grad_rho[i] = drho_dx[i] * drho_dx[i] + drho_dy[i] * drho_dy[i] + drho_dz[i] * drho_dz[i];
                    }

#ifdef XC_IMPLE_VER2

                    double* P_drho_dx = work_buffer + local_size * 1;
                    double* P_drho_dy = work_buffer + local_size * 2;
                    double* P_drho_dz = work_buffer + local_size * 3;
                    double* div_P_grad_rho = work_buffer + local_size * 4;

                    double E_x = 0.0;
                    for (int64_t i = 0; i < local_size; ++i) {
                        double vx_by_rho, P;
                        E_x += GGA_PBE::Potential_X_test3(l_rho_sigma[i], sq_grad_rho[i], &vx_by_rho, &P);
                        l_Vx[i] = vx_by_rho;
                        P_drho_dx[i] = P * drho_dx[i];
                        P_drho_dy[i] = P * drho_dy[i];
                        P_drho_dz[i] = P * drho_dz[i];

                    }
                    //printf("max_Vx = %g, %d\n", max_Vx, max_i);

                    GradientX_8th_ddm(l_grid, div_P_grad_rho, P_drho_dx, 1.0, dx);
                    for (int64_t i = 0; i < local_size; ++i) {
                        l_Vx[i] -= 2.0 * div_P_grad_rho[i];
                    }
                    GradientY_8th_ddm(l_grid, div_P_grad_rho, P_drho_dy, 1.0, dy);
                    for (int64_t i = 0; i < local_size; ++i) {
                        l_Vx[i] -= 2.0 * div_P_grad_rho[i];
                    }
                    GradientZ_8th_ddm(l_grid, div_P_grad_rho, P_drho_dz, 1.0, dz);
                    for (int64_t i = 0; i < local_size; ++i) {
                        l_Vx[i] -= 2.0 * div_P_grad_rho[i];
                    }
#else

                    double* rho_n_df_ds_drho_dx = out_head + local_size * 4;
                    double* rho_n_df_ds_drho_dy = out_head + local_size * 5;
                    double* rho_n_df_ds_drho_dz = out_head + local_size * 6;
                    double* div_rho_n_df_ds_grad_rho = out_head + local_size * 7;

                    double E_x = 0.0;
                    for (int64_t i = 0; i < local_size; ++i) {
                        double rho_n_df_ds;
                        E_x += l_rho_sigma[i] * GGA_PBE::Potential_X_test2(l_rho_sigma[i], sq_grad_rho[i], &l_Vx[i], &rho_n_df_ds);
                        rho_n_df_ds_drho_dx[i] = rho_n_df_ds * drho_dx[i];
                        rho_n_df_ds_drho_dy[i] = rho_n_df_ds * drho_dy[i];
                        rho_n_df_ds_drho_dz[i] = rho_n_df_ds * drho_dz[i];

                    }
                    //printf("max_Vx = %g, %d\n", max_Vx, max_i);

                    GradientX_8th_ddm(l_grid, div_rho_n_df_ds_grad_rho, rho_n_df_ds_drho_dx, 1.0, dx);
                    for (int64_t i = 0; i < local_size; ++i) {
                        l_Vx[i] += div_rho_n_df_ds_grad_rho[i];
                    }
                    GradientY_8th_ddm(l_grid, div_rho_n_df_ds_grad_rho, rho_n_df_ds_drho_dy, 1.0, dy);
                    for (int64_t i = 0; i < local_size; ++i) {
                        l_Vx[i] += div_rho_n_df_ds_grad_rho[i];
                    }
                    GradientZ_8th_ddm(l_grid, div_rho_n_df_ds_grad_rho, rho_n_df_ds_drho_dz, 1.0, dz);
                    for (int64_t i = 0; i < local_size; ++i) {
                        l_Vx[i] += div_rho_n_df_ds_grad_rho[i];
                    }
#endif
                    return E_x * dx * dy * dz;
                    };


#ifdef XC_IMPLE_VER2
                auto Correlation_GGA = [](GridRangeMPI& l_grid, const double* l_rho, const double* l_rho_diff, const double* drho_dx, const double* drho_dy, const double* drho_dz, double* l_Vc, double* l_Vc_down, double* work_buffer, double dx, double dy, double dz) {
                    const int64_t local_size = l_grid.Size3D();

                    double* sq_grad_rho = work_buffer;
                    for (int64_t i = 0; i < local_size; ++i) {
                        sq_grad_rho[i] = drho_dx[i] * drho_dx[i] + drho_dy[i] * drho_dy[i] + drho_dz[i] * drho_dz[i];
                    }

                    double* dE_ds_grad_rho_x = work_buffer + local_size * 1;
                    double* dE_ds_grad_rho_y = work_buffer + local_size * 2;
                    double* dE_ds_grad_rho_z = work_buffer + local_size * 3;
                    double* div_dE_ds_grad_rho = work_buffer + local_size * 4;

                    double E_c = 0.0;
                    if (l_Vc_down) {
                        for (int64_t i = 0; i < local_size; ++i) {
                            double vc_up, vc_down, P;
                            E_c += l_rho[i] * GGA_PBE::EnergyDensityPotential_C_test3(l_rho[i], l_rho_diff[i], sq_grad_rho[i], &vc_up, &vc_down, &P);
                            l_Vc[i] = vc_up;
                            l_Vc_down[i] = vc_down;
                            dE_ds_grad_rho_x[i] = P * drho_dx[i];
                            dE_ds_grad_rho_y[i] = P * drho_dy[i];
                            dE_ds_grad_rho_z[i] = P * drho_dz[i];
                        }
                    } else {
                        for (int64_t i = 0; i < local_size; ++i) {
                            double vc_up, vc_down, P;
                            E_c += l_rho[i] * GGA_PBE::EnergyDensityPotential_C_test3(l_rho[i], 0.0, sq_grad_rho[i], &vc_up, &vc_down, &P);
                            l_Vc[i] = vc_up;
                            dE_ds_grad_rho_x[i] = P * drho_dx[i];
                            dE_ds_grad_rho_y[i] = P * drho_dy[i];
                            dE_ds_grad_rho_z[i] = P * drho_dz[i];
                        }
                    }
                    //printf("max_Vx = %g, %d\n", max_Vc, max_i);

                    GradientX_8th_ddm(l_grid, div_dE_ds_grad_rho, dE_ds_grad_rho_x, 1.0, dx);
                    for (int64_t i = 0; i < local_size; ++i) {
                        l_Vc[i] -= 2.0 * div_dE_ds_grad_rho[i];
                    }

                    if (l_Vc_down) {
                        for (int64_t i = 0; i < local_size; ++i) {
                            l_Vc_down[i] -= 2.0 * div_dE_ds_grad_rho[i];
                        }
                    }
                    GradientY_8th_ddm(l_grid, div_dE_ds_grad_rho, dE_ds_grad_rho_y, 1.0, dy);
                    for (int64_t i = 0; i < local_size; ++i) {
                        l_Vc[i] -= 2.0 * div_dE_ds_grad_rho[i];
                    }
                    if (l_Vc_down) {
                        for (int64_t i = 0; i < local_size; ++i) {
                            l_Vc_down[i] -= 2.0 * div_dE_ds_grad_rho[i];
                        }
                    }

                    GradientZ_8th_ddm(l_grid, div_dE_ds_grad_rho, dE_ds_grad_rho_z, 1.0, dz);
                    for (int64_t i = 0; i < local_size; ++i) {
                        l_Vc[i] -= 2.0 * div_dE_ds_grad_rho[i];
                    }
                    if (l_Vc_down) {
                        for (int64_t i = 0; i < local_size; ++i) {
                            l_Vc_down[i] -= 2.0 * div_dE_ds_grad_rho[i];
                        }
                    }

                    return E_c * dx * dy * dz;
                    };
#else
                auto Correlation_GGA = [](GridRangeMPI& l_grid, double* l_rho_sigma, double* l_Vc, double* l_Vc_down, double* drho_dx, double dx, double dy, double dz) {
                    const int64_t local_size = l_grid.Size3D();

                    double* out_x = drho_dx;
                    double* out_y = out_x + local_size;
                    double* out_z = out_x + local_size * 2;
                    double* l_rho_sq_nabla = out_x + local_size * 3;

                    double* dE_ds_grad_rho_x = out_x + local_size * 4;
                    double* dE_ds_grad_rho_y = out_x + local_size * 5;
                    double* dE_ds_grad_rho_z = out_x + local_size * 6;
                    double* div_dE_ds_grad_rho = out_x + local_size * 7;

                    double E_c = 0.0;
                    if (l_Vc_down) {
                        for (int64_t i = 0; i < local_size; ++i) {
                            double dE_ds;
                            E_c += l_rho_sigma[i] * GGA_PBE::EnergyDensityPotential_C_test2(l_rho_sigma[i], 0.0, l_rho_sq_nabla[i], &l_Vc[i], &l_Vc_down[i], &dE_ds);
                            dE_ds_grad_rho_x[i] = dE_ds * out_x[i];
                            dE_ds_grad_rho_y[i] = dE_ds * out_y[i];
                            dE_ds_grad_rho_z[i] = dE_ds * out_z[i];
                        }
                    } else {
                        for (int64_t i = 0; i < local_size; ++i) {
                            double dummy;
                            double dE_ds;
                            E_c += l_rho_sigma[i] * GGA_PBE::EnergyDensityPotential_C_test2(l_rho_sigma[i], 0.0, l_rho_sq_nabla[i], &l_Vc[i], &dummy, &dE_ds);
                            dE_ds_grad_rho_x[i] = dE_ds * out_x[i];
                            dE_ds_grad_rho_y[i] = dE_ds * out_y[i];
                            dE_ds_grad_rho_z[i] = dE_ds * out_z[i];
                        }
                    }
                    //printf("max_Vx = %g, %d\n", max_Vc, max_i);

                    GradientX_8th_ddm(l_grid, div_dE_ds_grad_rho, dE_ds_grad_rho_x, 1.0, dx);
                    for (int64_t i = 0; i < local_size; ++i) {
                        l_Vc[i] += div_dE_ds_grad_rho[i];
                    }
                    GradientY_8th_ddm(l_grid, div_dE_ds_grad_rho, dE_ds_grad_rho_y, 1.0, dy);
                    for (int64_t i = 0; i < local_size; ++i) {
                        l_Vc[i] += div_dE_ds_grad_rho[i];
                    }
                    GradientZ_8th_ddm(l_grid, div_dE_ds_grad_rho, dE_ds_grad_rho_z, 1.0, dz);
                    for (int64_t i = 0; i < local_size; ++i) {
                        l_Vc[i] += div_dE_ds_grad_rho[i];
                    }

                    return E_c * dx * dy * dz;
                    };
#endif



                if (is_spin_on) {
                    // with Spin polarization//
                    
                    double* l_rho_sigma = m_work;
                    for (int64_t i = 0; i < local_size; ++i) {
                        l_rho_sigma[i] = (ml_rho[i] + ml_rho_diff[i] + ml_pcc_rho[i]) / 2.0;
                    }

                    double* drho_up_dx = m_work + local_size;
                    double* drho_up_dy = m_work + local_size * 2;
                    double* drho_up_dz = m_work + local_size * 3;
                    Gradient8th_ddm(ml_grid, drho_up_dx, drho_up_dy, drho_up_dz, l_rho_sigma, 1.0, m_dx, m_dy, m_dz);
                    double E_x_up = Exchange_GGA(ml_grid, l_rho_sigma, drho_up_dx, drho_up_dy, drho_up_dz, ml_Vx, m_work + local_size * 4, m_dx, m_dy, m_dz);


                    for (int64_t i = 0; i < local_size; ++i) {
                        l_rho_sigma[i] = (ml_rho[i] - ml_rho_diff[i] + ml_pcc_rho[i]) / 2.0;
                    }

                    double* drho_down_dx = m_work + local_size * 4;
                    double* drho_down_dy = m_work + local_size * 5;
                    double* drho_down_dz = m_work + local_size * 6;
                    Gradient8th_ddm(ml_grid, drho_down_dx, drho_down_dy, drho_down_dz, l_rho_sigma, 1.0, m_dx, m_dy, m_dz);
                    double E_x_down = Exchange_GGA(ml_grid, l_rho_sigma, drho_down_dx, drho_down_dy, drho_down_dz, ml_Vx_down, m_work + local_size * 7, m_dx, m_dy, m_dz);

                    m_E_x = E_x_up + E_x_down;


#ifdef IGNORE_GGA_PBE_CORRELATION_H 
                    double E_c = 0.0;
                    double dummy = 0.0;
                    for (int64_t i = 0; i < local_size; ++i) {
                        double rho_i = ml_rho[i] + ml_pcc_rho[i];
                        E_c += rho_i * LDA_PW92::EnergyDensityPotential_C(ml_rho[i] + ml_pcc_rho[i], ml_rho_diff[i], &ml_Vc[i], &ml_Vc_down[i]);
                    }
                    m_E_c = E_c * m_dx * m_dy * m_dz;
#elif defined(XC_IMPLE_VER2)
                    //double* l_rho_sigma = m_work;
                    for (int64_t i = 0; i < local_size; ++i) {
                        l_rho_sigma[i] = ml_rho[i] + ml_pcc_rho[i];
                        drho_up_dx[i] += drho_down_dx[i];
                        drho_up_dy[i] += drho_down_dy[i];
                        drho_up_dz[i] += drho_down_dz[i];
                    }
                    m_E_c = Correlation_GGA(ml_grid, l_rho_sigma, ml_rho_diff, drho_up_dx, drho_up_dy, drho_up_dz, ml_Vc, ml_Vc_down, m_work + local_size * 7, m_dx, m_dy, m_dz);

#else
                    double* l_rho_sigma = m_work;
                    double* sq_grad_rho_up = m_work + local_size * 4;
                    for (int64_t i = 0; i < local_size; ++i) {
                        l_rho_sigma[i] = ml_rho[i] + ml_pcc_rho[i];
                        drho_up_dx[i] += drho_down_dx[i];
                        drho_up_dy[i] += drho_down_dy[i];
                        drho_up_dz[i] += drho_down_dz[i];
                        sq_grad_rho_up[i] = drho_up_dx[i] * drho_up_dx[i] + drho_up_dy[i] * drho_up_dy[i] + drho_up_dz[i] * drho_up_dz[i];
                    }
                    m_E_c = Correlation_GGA(ml_grid, l_rho_sigma, ml_Vc, ml_Vc_down, drho_up_dx, m_dx, m_dy, m_dz);
#endif
                } else {
                    //without Spin polarization//
                    double* l_rho_sigma = m_work;
                    for (int64_t i = 0; i < local_size; ++i) {
                        l_rho_sigma[i] = (ml_rho[i] + ml_pcc_rho[i]) / 2.0;
                    }


                    double* drho_dx = m_work + local_size;
                    double* drho_dy = m_work + local_size * 2;
                    double* drho_dz = m_work + local_size * 3;
                    Gradient8th_ddm(ml_grid, drho_dx, drho_dy, drho_dz, l_rho_sigma, 1.0, m_dx, m_dy, m_dz);

                    //calculate E_x, and derivative of rho//
                    double E_x_up = Exchange_GGA(ml_grid, l_rho_sigma, drho_dx, drho_dy, drho_dz, ml_Vx, m_work + local_size * 7, m_dx, m_dy, m_dz);
                    m_E_x = E_x_up * 2.0;

#ifdef IGNORE_GGA_PBE_CORRELATION_H 
                    double E_c = 0.0;
                    double dummy = 0.0;
                    for (int64_t i = 0; i < local_size; ++i) {
                        double rho_i = ml_rho[i] + ml_pcc_rho[i];
                        E_c += rho_i * LDA_PW92::EnergyDensityPotential_C(ml_rho[i] + ml_pcc_rho[i], 0.0, &ml_Vc[i], &dummy);
                    }
                    m_E_c = E_c * m_dx * m_dy * m_dz;
#else
                    for (int64_t i = 0; i < local_size; ++i) {
                        l_rho_sigma[i] *= 2.0;
                        drho_dx[i] *= 2.0;
                        drho_dy[i] *= 2.0;
                        drho_dz[i] *= 2.0;
                    }
                    m_E_c = Correlation_GGA(ml_grid, l_rho_sigma, nullptr, drho_dx, drho_dy, drho_dz, ml_Vc, nullptr, m_work + local_size * 4, m_dx, m_dy, m_dz);
#endif
                }

                //printf("[%d] local E_x, E_c = %f, %f\n", GetProcessID(m_ddm_comm), m_E_x, m_E_c);

                double Exc_local[2]{ m_E_x, m_E_c };
                double Exc_total[2]{ 0.0,0.0 };
                MPI_Reduce(&Exc_local[0], &Exc_total[0], 2, MPI_DOUBLE, MPI_SUM, 0, m_ddm_comm);
                m_E_x = Exc_total[0];
                m_E_c = Exc_total[1];


            } else if (m_xc_type == XC_TYPE::LDA) {
#ifdef USE_PCC_HR
                auto& range = ml_grid;
                const int local_size_x = range.end_x - range.begin_x;
                const int local_size_y = range.end_y - range.begin_y;
                const int local_size_z = range.end_z - range.begin_z;
                auto hr_grid = ScaleRange(ml_grid, m_HR_ratio_x, m_HR_ratio_y, m_HR_ratio_z);
                mDDMFFT_Upconvert(ml_rho, ml_hr_rho);
                CheckMinMax(ml_grid, ml_rho, "rho  ");
                CheckMinMax(ScaleRange(ml_grid, m_HR_ratio_x, m_HR_ratio_y, m_HR_ratio_z), ml_hr_rho, "rhoHR");
                //test
                /*
                //mDDMFFT_Downconvert_SvF(ml_hr_rho, ml_rho);
                mDDMFFT_Downconvert(ml_hr_rho, m_work);
                {
                    double sum0 = 0.0;
                    double sum1 = 0.0;
                    double inner = 0.0;
                    for (int64_t i = 0; i < ml_grid.Size3D(); ++i) {
                        sum0 += ml_rho[i]* ml_rho[i];
                        sum1 += m_work[i] * m_work[i];
                        inner += ml_rho[i] * m_work[i];
                    }
                    inner /=  sqrt(sum0 * sum1);
                    sum0 *= m_dx * m_dy * m_dz;
                    sum1 *= m_dx * m_dy * m_dz;
                    
                    printf("sum=%.10f, %.10f, %.10f\n", sum0, sum1, inner);
                }
                */
                if (!is_spin_on) {
                    //Vxcの計算にのみPCC chargeを加味する//                    
                    mSetPotentialVxc_LDA_local(hr_grid, ml_hr_Vx, ml_hr_Vc, ml_hr_rho, ml_hr_pcc);

#if 0
                    mDDMFFT_Downconvert_SvF(ml_hr_Vx, ml_Vx);
                    mDDMFFT_Downconvert_SvF(ml_hr_Vc, ml_Vc);
#elif 1
                    //原理的にはRealspaceよりこちらの方が精度が出るはず
                    //DownConvert_Realspaceよりも精度が落ちる//
                    mDDMFFT_Downconvert(ml_hr_Vx, ml_Vx);
                    mDDMFFT_Downconvert(ml_hr_Vc, ml_Vc);
#else
                    DownConvert_Realspace(ml_Vx, local_size_x, local_size_y, local_size_z, ml_hr_Vx, m_HR_ratio_x, m_HR_ratio_y, m_HR_ratio_z);
                    DownConvert_Realspace(ml_Vc, local_size_x, local_size_y, local_size_z, ml_hr_Vc, m_HR_ratio_x, m_HR_ratio_y, m_HR_ratio_z);
#endif
                } else {
                    mDDMFFT_Upconvert(ml_rho_diff, ml_hr_rho_diff);
                    CheckMinMax(ml_grid, ml_pcc_rho, "pcc  ");
                    CheckMinMax(ScaleRange(ml_grid,m_HR_ratio_x, m_HR_ratio_y, m_HR_ratio_z), ml_hr_pcc, "pccHR");

                    mSetPotentialVxc_LSDA_local(hr_grid, ml_hr_Vx, ml_hr_Vc, ml_hr_Vx_down, ml_hr_Vc_down, ml_hr_rho, ml_hr_pcc, ml_hr_rho_diff);

#if 0
                    mDDMFFT_Downconvert_SvF(ml_hr_Vx, ml_Vx);
                    mDDMFFT_Downconvert_SvF(ml_hr_Vc, ml_Vc);
                    mDDMFFT_Downconvert_SvF(ml_hr_Vx_down, ml_Vx_down);
                    mDDMFFT_Downconvert_SvF(ml_hr_Vc_down, ml_Vc_down);
#elif 1
                    //原理的にはRealspaceよりこちらの方が精度が出るはず
                    //DownConvert_Realspaceよりも精度が落ちる//
                    mDDMFFT_Downconvert(ml_hr_Vx, ml_Vx);
                    //DownConvert_Realspace(ml_Vc, local_size_x, local_size_y, local_size_z, ml_hr_Vc, m_HR_ratio_x, m_HR_ratio_y, m_HR_ratio_z);
                    mDDMFFT_Downconvert(ml_hr_Vc, ml_Vc);
                    CheckMinMax(ml_grid, ml_Vc, "Vc  ");
                    CheckMinMax(ScaleRange(ml_grid, m_HR_ratio_x, m_HR_ratio_y, m_HR_ratio_z), ml_hr_Vc, "VcHR");
                    mDDMFFT_Downconvert(ml_hr_Vx_down, ml_Vx_down);
                    //DownConvert_Realspace(ml_Vc_down, local_size_x, local_size_y, local_size_z, ml_hr_Vc_down, m_HR_ratio_x, m_HR_ratio_y, m_HR_ratio_z);
                    mDDMFFT_Downconvert(ml_hr_Vc_down, ml_Vc_down);
#elif 1
                    //原理的にはRealspaceよりこちらの方が精度が出るはず
                    //DownConvert_Realspaceよりも精度が落ちる//
                    mDDMFFT_Downconvert(ml_hr_Vx, ml_Vx);
                    mDDMFFT_Downconvert(ml_hr_Vc, ml_Vc);
                    mDDMFFT_Downconvert(ml_hr_Vx_down, ml_Vx_down);
                    mDDMFFT_Downconvert(ml_hr_Vc_down, ml_Vc_down);
#else
                    DownConvert_Realspace(ml_Vx, local_size_x, local_size_y, local_size_z, ml_hr_Vx, m_HR_ratio_x, m_HR_ratio_y, m_HR_ratio_z);
                    DownConvert_Realspace(ml_Vc, local_size_x, local_size_y, local_size_z, ml_hr_Vc, m_HR_ratio_x, m_HR_ratio_y, m_HR_ratio_z);
                    DownConvert_Realspace(ml_Vx_down, local_size_x, local_size_y, local_size_z, ml_hr_Vx_down, m_HR_ratio_x, m_HR_ratio_y, m_HR_ratio_z);
                    DownConvert_Realspace(ml_Vc_down, local_size_x, local_size_y, local_size_z, ml_hr_Vc_down, m_HR_ratio_x, m_HR_ratio_y, m_HR_ratio_z);
                    
#endif

                }
#else
                if (!is_spin_on) {
                    //Vxcの計算にのみPCC chargeを加味する//                    
                    mSetPotentialVxc_LDA_local(ml_grid, ml_Vx, ml_Vc, ml_rho, ml_pcc_rho);
                } else {
                    //Vxcの計算にのみPCC chargeを加味する//		
                    mSetPotentialVxc_LSDA_local(ml_grid, ml_Vx, ml_Vc, ml_Vx_down, ml_Vc_down, ml_rho, ml_pcc_rho, ml_rho_diff);
                }
#endif
            }
        }
#ifdef IGNORE_XC
        {//cancell vx and vc to ignore XC
            m_E_x = 0.0;
            m_E_c = 0.0;
            const int64_t local_size = ml_grid.Size3D();
            memset(&ml_Vx[0], 0, sizeof(double)* local_size);
            memset(&ml_Vc[0], 0, sizeof(double)* local_size);
            if (is_spin_on) {
                memset(&ml_Vx_down[0], 0, sizeof(double) * local_size);
                memset(&ml_Vc_down[0], 0, sizeof(double) * local_size);
            }
        }
#endif
#ifdef IGNORE_EXCHANGE
        {//cancell vx and vc to ignore XC
            m_E_x = 0.0;
            const int64_t local_size = ml_grid.Size3D();
            memset(&ml_Vx[0], 0, sizeof(double) * local_size);
            if (is_spin_on) {
                memset(&ml_Vx_down[0], 0, sizeof(double) * local_size);
            }
        }
#endif
#ifdef IGNORE_CORRELATION
        {//cancell vx and vc to ignore XC
            m_E_c = 0.0;
            const int64_t local_size = ml_grid.Size3D();
            memset(&ml_Vc[0], 0, sizeof(double) * local_size);
            if (is_spin_on) {
                memset(&ml_Vc_down[0], 0, sizeof(double) * local_size);
            }
        }
#endif

    }
    watch.Record(55);
}

/*
* すでに計算済みのVext, Vhart, Vx, VcをVtotにまとめる
* Vextは原子核の移動時にリセットされ、一方でVhart, Vx, Vcは波動関数の変化時にリセットされる。
* よって、どちらか一方でも変わったときには、Vtotを変更すべき。
* もっと楽なのは、Vtotだけは使うたびリセットすること。
*/
inline
void QUMASUN_BASE1::mSetPotentialVtot() {

    //Vtot ///////////////////////////////////////////////// 
    if (IsRoot(m_same_ddm_place_comm)) {
        
        const size_t local_size = ml_grid.Size3D();
        for (size_t i = 0; i < local_size; ++i) {
            ml_Vtot[i] = ml_Vext[i] + ml_Vhart[i] + ml_Vx[i] + ml_Vc[i];
        }
        MPI_Bcast(ml_Vtot, (int)local_size, MPI_DOUBLE, 0, m_same_ddm_place_comm);


        if (is_spin_on) {
            for (size_t i = 0; i < local_size; ++i) {
                ml_Vtot_down[i] = ml_Vext[i] + ml_Vhart[i] + ml_Vx_down[i] + ml_Vc_down[i];
            }
            MPI_Bcast(ml_Vtot_down, (int)local_size, MPI_DOUBLE, 0, m_same_ddm_place_comm);

        }
    } else {

        const size_t local_size = ml_grid.Size3D();
        MPI_Bcast(ml_Vtot, (int)local_size, MPI_DOUBLE, 0, m_same_ddm_place_comm);
        if (is_spin_on) {
            MPI_Bcast(ml_Vtot_down, (int)local_size, MPI_DOUBLE, 0, m_same_ddm_place_comm);

        }

    }
    watch.Record(56);
}



inline
void QUMASUN_BASE1::mSetPotentialVxc_LDA(double* Vx, double* Vc, const double* rho, const double* pcc_rho) {

	for (size_t i = 0; i < m_size_3d; ++i) {
#ifdef XC_IMPLE_VER2
        auto ret = Calc_XC_LSDA_v2(rho[i] + pcc_rho[i], 0.0);
        Vx[i] = ret.V_x_up;
        Vc[i] = ret.V_c_up;
#else
        auto ret = Calc_XC_LDA(rho[i] + pcc_rho[i]);
		Vx[i] = ret.V_x;
		Vc[i] = ret.V_c;
#endif
	}
}

inline
void QUMASUN_BASE1::mSetPotentialVxc_LSDA(double* Vx_up, double* Vc_up, double* Vx_down, double* Vc_down, const double* rho, const double* pcc_rho, const double* rho_diff) {
	for (size_t i = 0; i < m_size_3d; ++i) {
#ifdef XC_IMPLE_VER2
        auto ret = Calc_XC_LSDA_v2(rho[i] + pcc_rho[i], rho_diff[i]);
#else
        auto ret = Calc_XC_LSDA(rho[i] + pcc_rho[i], rho_diff[i] / (rho[i] + pcc_rho[i]));
#endif
		Vx_up[i] = ret.V_x_up;
		Vc_up[i] = ret.V_c_up;
		Vx_down[i] = ret.V_x_down;
		Vc_down[i] = ret.V_c_down;
	}

}

inline
void QUMASUN_BASE1::mSetPotentialVxc_LDA_local(const GridRange& l_grid, double* Vx, double* Vc, const double* rho, const double* pcc_rho) {
    const size_t local_size = l_grid.Size3D();
    for (size_t i = 0; i < local_size; ++i) {
        auto ret = Calc_XC_LDA(rho[i] + pcc_rho[i]);
        Vx[i] = ret.V_x;
        Vc[i] = ret.V_c;
    }
}

inline
void QUMASUN_BASE1::mSetPotentialVxc_LSDA_local(const GridRange& l_grid, double* Vx_up, double* Vc_up, double* Vx_down, double* Vc_down, const double* rho, const double* pcc_rho, const double* rho_diff) {
    const size_t local_size = l_grid.Size3D();
    for (size_t i = 0; i < local_size; ++i) {
#ifdef XC_IMPLE_VER2
        auto ret = Calc_XC_LSDA_v2(rho[i] + pcc_rho[i], rho_diff[i]);
#else
        auto ret = Calc_XC_LSDA(rho[i] + pcc_rho[i], rho_diff[i] / (rho[i] + pcc_rho[i]));
#endif
        Vx_up[i] = ret.V_x_up;
        Vc_up[i] = ret.V_c_up;
        Vx_down[i] = ret.V_x_down;
        Vc_down[i] = ret.V_c_down;
    }

}


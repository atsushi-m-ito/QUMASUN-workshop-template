#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include "qumasun_base1.h"
#include "poisson_fft.h"
#include "Vxc.h"
#include "nucleus.h"
#include "convertHR.h"
#include "GaussLegendre.h"
#include "LagrangeInterpolation.h"
#include "CubicHermiteSpline.h"


#ifdef DDM_FFT

/*
* Perform the following calculations
* - Vext due to nuclei
* - Vlocal potential due to nuclei
* - Vnonlocal potential due to nuclei
* - correction of Core-Core repulsion
* where they only need to be calculated once before the SCF calculation
*/


/***************************************
* Note: Coreの補正項(内殻電子密度とVlocalの楕円積分)を補間する方法として
* Lagrange補間、CubicHermiteSpline補間、5次精度Spline補間(両端の2階微分まで使用)
* の三種類を比較した。
* さらに、CubicHermiteSpline補間では、両端の1階微分値として、
* 楕円積分から直接計算する方法と二次精度中心差分で代替する方法を試した
* 結果として、
* Lagrange補間: 
*   点数は24から48点がよい. それ以下でもそれ以上でも精度が落ちる.
*   ただし、偶関数として0の両側をミラー対称で補完させているので実際の点数は半分に相当する
*   ONCV擬ポテンシャルなら十分な静度が出る.
*   MBK(ADPACK)擬ポテンシャルでもCubicHermiteSplineよりは精度が出る.
* 
* CubicHermiteSpline補間:
*   点数は16程度がよい. 100以上だと精度が悪くなる
*   両端の1階微分値は楕円積分から直接計算する方が良い
*   ONCV擬ポテンシャルなら十分な静度が出るが, Lagrange補間にやや劣る
*   MBK(ADPACK)擬ポテンシャルでは悩ましい精度
*   
* 5次精度Spline補間:
*   精度が良くない。微分値がを差分で設定しても波打ってしまう。
* 
* qumasun_base1.hのCOMPARISON_INTERPOLATION_MODELを有効にすると、上記の比較を行ってファイルを出力
*****************************************/

inline
void QUMASUN_BASE1::mPrepareCore() {
    const bool is_root = IsRoot(m_mpi_comm);


    auto& m_pp_integrator = m_pp_SvF;

    if (IsRoot(m_same_ddm_place_comm)) {
        if (m_hamiltonian_type == HAMILTONIAN::KohnSham_PP) {//DFT//

            //Vext calculation on HR (high resolution) grid///////////////////////////////////////////////////////
            watch.Restart();
            m_pp_integrator.SetChargeVlocal_HR(ml_hr_nucl_rho, ml_grid, m_nuclei, m_num_nuclei);
            //m_pp_integrator.SetPccCharge_v2(ml_hr_nucl_rho, ml_grid, m_nuclei, m_num_nuclei);
            watch.Record(31);
            MPI_Barrier(m_ddm_comm);
            watch.Record(37);

            //printf("TEST: %s: %d\n", __FILE__, __LINE__);

            //call here//
#ifdef GRAD_VEXT_FFT
            mPoissonDDMFFT_HR(ml_hr_nucl_rho, ml_hr_Vext, ml_hr_dVext_dx, ml_hr_dVext_dy, ml_hr_dVext_dz);
#else
            mPoissonDDMFFT_HR(ml_hr_nucl_rho, ml_hr_Vext, nullptr, nullptr, nullptr);
#endif

            //printf("TEST: %s: %d\n", __FILE__, __LINE__);

            auto& range = ml_grid;
            const int local_size_x = range.end_x - range.begin_x;
            const int local_size_y = range.end_y - range.begin_y;
            const int local_size_z = range.end_z - range.begin_z;

#ifdef DDM_FFT_DOWNCONVERT_CORE
            //そもそもmPoissonDDMFFT_HRで同時にDownConvertできるのでは？
            mDDMFFT_Downconvert(ml_hr_Vext, ml_Vext);
#else
            DownConvert_Realspace(ml_Vext, local_size_x, local_size_y, local_size_z, ml_hr_Vext, m_HR_ratio_x, m_HR_ratio_y, m_HR_ratio_z);
#endif
            watch.Record(36);
            //printf("TEST: %s: %d\n", __FILE__, __LINE__);

#ifdef USE_PCC_HR
            m_pp_integrator.SetPccCharge_HR(ml_hr_pcc, ml_grid, m_nuclei, m_num_nuclei);
            watch.Record(26);

#if 0
            mDDMFFT_Downconvert_SvF(ml_hr_pcc, ml_pcc_rho);
#elif 1
            //DownConvert_Realspaceよりも精度が落ちる//
            mDDMFFT_Downconvert(ml_hr_pcc, ml_pcc_rho);
#else
            DownConvert_Realspace(ml_pcc_rho, local_size_x, local_size_y, local_size_z, ml_hr_pcc, m_HR_ratio_x, m_HR_ratio_y, m_HR_ratio_z);
#endif
            watch.Record(36);

#else
            m_pp_integrator.SetPccCharge_v2(ml_pcc_rho, ml_grid, m_nuclei, m_num_nuclei);
            watch.Record(26);
            //printf("TEST: %s: %d\n", __FILE__, __LINE__);
#endif            
        }
        
    }



    m_Ecore_self_HR = m_pp_SvF.GetEnergySelfCore(m_nuclei, m_num_nuclei) / 2.0;

    //printf("TEST: %s: %d\n", __FILE__, __LINE__);
    mCoreCoreCorrection();

    //printf("TEST: %s: %d\n", __FILE__, __LINE__);
}

#else
/*
* Perform the following calculations
* - Vext due to nuclei
* - Vlocal potential due to nuclei
* - Vnonlocal potential due to nuclei
* - correction of Core-Core repulsion 
* where they only need to be calculated once before the SCF calculation
*/
inline
void QUMASUN_BASE1::mPrepareCore() {
    const bool is_root = IsRoot(m_mpi_comm);



    auto& m_pp_integrator = m_pp_SvF;

    if (IsRoot(m_same_ddm_place_comm)) {
        if (m_hamiltonian_type == HAMILTONIAN::KohnSham_PP) {//DFT//

            //Vext calculation on HR (high resolution) grid///////////////////////////////////////////////////////

            m_pp_integrator.SetChargeVlocal_HR(ml_hr_nucl_rho, ml_grid, m_nuclei, m_num_nuclei);
            watch.Record(31);

            const size_t hr_size_3d = (m_size_x * m_HR_ratio_x) * (m_size_y * m_HR_ratio_y) * (m_size_z * m_HR_ratio_z);
            const size_t hr_size_3d_c = hr_size_3d * 2;

            double* hr_nucl_rho = m_work;

            double* hr_Vext_k = m_work + hr_size_3d_c;


            if (IsRoot(ml_grid.mpi_comm)) {

                mGatherField_HR(hr_nucl_rho, ml_hr_nucl_rho);

                double sumrho = 0.0;
                const size_t hr_size_3d = m_size_3d * m_HR_ratio_x * m_HR_ratio_y * m_HR_ratio_z;
                for (size_t i = 0; i < hr_size_3d; ++i) {
                    sumrho += hr_nucl_rho[i];
                }
                sumrho *= m_dx * m_dy * m_dz / (double)(m_HR_ratio_x * m_HR_ratio_y * m_HR_ratio_z);
                printf("ChargeForVlocal(HR)=%f\n", sumrho);
                watch.Record(33);
#if 1
                SetPotentialByPoissonFFT_v2(m_HR_fftw, m_hr_Vext, hr_nucl_rho, hr_Vext_k, m_size_x * m_HR_ratio_x, m_size_y * m_HR_ratio_y, m_size_z * m_HR_ratio_z, m_dx / (double)m_HR_ratio_x, m_dy / (double)m_HR_ratio_y, m_dz / (double)m_HR_ratio_z);
#else
                SetPotentialByPoissonFFT(m_hr_Vext, hr_nucl_rho, m_size_x * m_HR_ratio_x, m_size_y * m_HR_ratio_y, m_size_z * m_HR_ratio_z, m_dx / (double)m_HR_ratio_x, m_dy / (double)m_HR_ratio_y, m_dz / (double)m_HR_ratio_z, m_work + hr_size_3d);
#endif
                watch.Record(34);
            } else {
                mGatherField_HR(nullptr, ml_hr_nucl_rho);
            }

            mScatterField_HR(ml_hr_Vext, m_hr_Vext);
            //printf("[%d]mGatherField\n", proc_id); fflush(stdout);
            watch.Record(35);


            //Vext calculation on low resolution grid///////////////////////////////////////////////////////


            if (IsRoot(ml_grid.mpi_comm)) {
                DownConvert_Realspace(m_Vext.Pointer(), m_size_x, m_size_y, m_size_z, m_hr_Vext.Pointer(), m_HR_ratio_x, m_HR_ratio_y, m_HR_ratio_z);
                watch.Record(36);
            }
            mScatterField(ml_Vext, m_Vext);
            watch.Record(25);

#ifdef GRAD_VEXT_FFT
            if (IsRoot(ml_grid.mpi_comm)) {
                const size_t hr_size_3d = m_size_3d * m_HR_ratio_x * m_HR_ratio_y * m_HR_ratio_z;

                const double invN = 1.0 / (double)(hr_size_3d);
                for (size_t i = 0; i < hr_size_3d_c; ++i) {
                    hr_Vext_k[i] *= invN;
                }


                double* hr_dV_dxyz = m_work + hr_size_3d_c * 2;

                //grad_x /////////////
                fftw_complex* hr_dV_dxyz_k = m_HR_fftw->GetBuffer();
                GradientXInKspace(hr_dV_dxyz_k, (fftw_complex*)hr_Vext_k, m_size_x * m_HR_ratio_x, m_size_y * m_HR_ratio_y, m_size_z * m_HR_ratio_z, m_dx / (double)m_HR_ratio_x, m_dy / (double)m_HR_ratio_y, m_dz / (double)m_HR_ratio_z);
                watch.Record(37);
                m_HR_fftw->BackwardDirect(hr_dV_dxyz_k);
                for (int i = 0; i < hr_size_3d; ++i) {
                    hr_dV_dxyz[i] = hr_dV_dxyz_k[i][0];
                }
                watch.Record(38);
                mScatterField_HR(ml_hr_dVext_dx, hr_dV_dxyz);
                watch.Record(35);

                //grad_y /////////////
                GradientYInKspace(hr_dV_dxyz_k, (fftw_complex*)hr_Vext_k, m_size_x * m_HR_ratio_x, m_size_y * m_HR_ratio_y, m_size_z * m_HR_ratio_z, m_dx / (double)m_HR_ratio_x, m_dy / (double)m_HR_ratio_y, m_dz / (double)m_HR_ratio_z);
                watch.Record(37);
                m_HR_fftw->BackwardDirect(hr_dV_dxyz_k);
                for (int i = 0; i < hr_size_3d; ++i) {
                    hr_dV_dxyz[i] = hr_dV_dxyz_k[i][0];
                }
                watch.Record(38);
                mScatterField_HR(ml_hr_dVext_dy, hr_dV_dxyz);
                watch.Record(35);

                //grad_z /////////////
                GradientZInKspace(hr_dV_dxyz_k, (fftw_complex*)hr_Vext_k, m_size_x * m_HR_ratio_x, m_size_y * m_HR_ratio_y, m_size_z * m_HR_ratio_z, m_dx / (double)m_HR_ratio_x, m_dy / (double)m_HR_ratio_y, m_dz / (double)m_HR_ratio_z);
                watch.Record(37);
                m_HR_fftw->BackwardDirect(hr_dV_dxyz_k);
                for (int i = 0; i < hr_size_3d; ++i) {
                    hr_dV_dxyz[i] = hr_dV_dxyz_k[i][0];
                }
                watch.Record(38);
                mScatterField_HR(ml_hr_dVext_dz, hr_dV_dxyz);
                watch.Record(35);

            } else {
                //slave process//
                mScatterField_HR(ml_hr_dVext_dx, nullptr);
                mScatterField_HR(ml_hr_dVext_dy, nullptr);
                mScatterField_HR(ml_hr_dVext_dz, nullptr);
                watch.Record(35);
            }


#endif




            m_pp_integrator.SetPccCharge_v2(ml_pcc_rho, ml_grid, m_nuclei, m_num_nuclei);
            watch.Record(26);
            /*
            mGatherField(m_pcc_rho, ml_pcc_rho);
            watch.Record(27);
            */
        } else {
            if (IsRoot(ml_grid.mpi_comm)) {
                mSetPotentialVext(m_Vext, m_nuclei, m_num_nuclei);
            }
        }

        /* move to QUMASUN_BASE1::mInitializeBuffer
        //Valence Electron/////////////////////////////////////////
        {

            for (int i = 0; i < m_num_nuclei; ++i) {
                m_nuclei_valence_elecron[i] = m_pp_integrator.NumValenceElectron(m_nuclei[i].Z);
            }

            watch.Record(28);
        }
        */
    }



    m_Ecore_self_HR = m_pp_SvF.GetEnergySelfCore(m_nuclei, m_num_nuclei) / 2.0;

    mCoreCoreCorrection();
}
#endif

inline
void QUMASUN_BASE1::mCoreCoreCorrection() {
    const bool is_root = IsRoot(m_mpi_comm);
	if (is_root) {
        
        //(v2)E_core_core and PP内側の補正(VlocalとZZ/rの差)//////////////////////////////////////////////
        {
            //カットオフの記憶//
            auto cutoff_vlocal = std::make_unique<double[]>(m_num_nuclei);
            int prev_Z = 0;
            double prev_cut = 0.0;
            for (int i = 0; i < m_num_nuclei; ++i) {
                if (m_nuclei[i].Z == prev_Z) {
                    cutoff_vlocal[i] = prev_cut;
                } else {

                    prev_cut = m_pp_SvF.GetCutoffVlocal(m_nuclei[i].Z);
                    cutoff_vlocal[i] = prev_cut;
                    prev_Z = m_nuclei[i].Z;
                }
            }

            //force clear//
            m_force_nn_correction = std::make_unique<vec3d[]>(m_num_nuclei);
            m_force_nn_TF = std::make_unique<vec3d[]>(m_num_nuclei);
            for (int i = 0; i < m_num_nuclei; ++i) {
                m_force_nn_correction[i] = { 0.0,0.0,0.0 };
                m_force_nn_TF[i] = { 0.0,0.0,0.0 };
            }

            double EV1 = 0.0;
            double EV2 = 0.0;
            double EV_cutoff_12 = 0.0;

            auto Folding = [](double x, double box_width) {
                return x - floor(x / box_width) * box_width;
                };

            auto Distance = [](double x, double box_width) {
                return (fabs(x) < box_width * 0.5) ? x : -copysign(box_width - fabs(x), x);
                };

            auto E_TF = [](double R, double Z_a, double Z_b, double Q_a, double Q_b, double a0, double b0) {
                return (Q_a + (Z_a - Q_a) * exp(-R / a0)) * (Q_b + (Z_b - Q_b) * exp(-R / b0)) / R;
                };

            auto Force_TF = [](double R, double Z_a, double Z_b, double Q_a, double Q_b, double a0, double b0) {
                double F = (Q_a + (Z_a - Q_a) * exp(-R / a0)) * (Q_b + (Z_b - Q_b) * exp(-R / b0)) / (R);
                F += ((Z_a - Q_a) * exp(-R / a0)) * (Q_b + (Z_b - Q_b) * exp(-R / b0)) / (a0);
                F += (Q_a + (Z_a - Q_a) * exp(-R / a0)) * ((Z_b - Q_b) * exp(-R / b0)) / (b0);
                return F / R;
                };





            //note: Energy, Forceともに最後に1/2を乗じる.
            //よって、計算のループにおいてはi<kおよびi>kの場合を合わせてダブルカウントする//

            for (int i = 0; i < m_num_nuclei; ++i) {
                const double Qi = m_nuclei_valence_elecron[i];
                const double Z_a = m_nuclei[i].Z;
                const double a0 = cutoff_vlocal[i] / (2.0 * Z_a / Qi) / 2.0; //2.0で割るのは補正(20250411).カットオフが長いと格子定数が合わなくなる//
                double xi = Folding(m_nuclei[i].Rx, m_box_x);
                double yi = Folding(m_nuclei[i].Ry, m_box_y);
                double zi = Folding(m_nuclei[i].Rz, m_box_z);


                //printf("Folding[%d]: %.10f, %.10f, %.10f\n", i, xi, yi, zi);
                

                //近接の原子核との相互作用のCoulombとVlocalの差//
                //擬ポテンシャルのカットオフ長よりも内側におけるVlocalと理想のZZ/rとの差を補正//
                for (int k = 0; k < m_num_nuclei; ++k) {
                    const double cutoff_length_PP = (cutoff_vlocal[i] + cutoff_vlocal[k])*2.0;// PPのVlocalとZZ/rがズレ始める距離
                    const double Qk = m_nuclei_valence_elecron[k];
                    const double Z_b = m_nuclei[k].Z;
                    const double b0 = cutoff_vlocal[k] / (2.0 * Z_b / Qk) / 2.0;
                    double xk = Folding(m_nuclei[k].Rx, m_box_x);
                    double yk = Folding(m_nuclei[k].Ry, m_box_y);
                    double zk = Folding(m_nuclei[k].Rz, m_box_z);

#if 0
                    const double over_ratio = 1.05;
                    auto convert_x_to_r = [&](double x) {
                        return  ((x * over_ratio + 1.0) / 2.0) * cutoff_length_PP;
                        };
                    auto inverse_r_to_x = [&](double r) {
                        return ((r * 2.0 / (cutoff_length_PP)) - 1.0) / over_ratio;
                        };
                    const double scale_x_r = 2.0 / (cutoff_length_PP * over_ratio);
#else
                    const double over_ratio = 1.05;
                    auto convert_x_to_r = [&](double x) {
                        return  x * over_ratio * cutoff_length_PP;
                        };
                    auto inverse_r_to_x = [&](double r) {
                        return r  / (cutoff_length_PP * over_ratio);
                        };
                    const double scale_x_r = 1.0 / (cutoff_length_PP * over_ratio);
#endif
                    //TFポテンシャルと擬ポテンシャルのコアコア相互作用のカットオフ長でのエネルギー差を第二の補正として加える必要がある//
                    double E_correction_TF_cutoff;
                    double E_correction_PP_cutoff;
                    double optimal_cutoff_r;
                    const auto pair_key = (m_nuclei[i].Z <= m_nuclei[k].Z) ? std::make_pair(m_nuclei[i].Z, m_nuclei[k].Z) : std::make_pair(m_nuclei[k].Z, m_nuclei[i].Z);
                    auto pair_itr = m_const_Enn_close_pairlist.find(pair_key);
                    if (pair_itr == m_const_Enn_close_pairlist.end()){
                        //この元素pairを始めて計算する場合は計算する//

                        // 補正データセットの初期化                        
                        auto res = m_const_Enn_close_pairlist.emplace(pair_key, EnnCorrect{ });
                        pair_itr = res.first;
                        auto& correct_info = res.first->second;
                        
                        correct_info.scale_x_r = scale_x_r;

#ifndef IGNORE_CORECORE_CORRECTION                        
#ifdef INTERPOLATION_CORECORE_CORRECTION
                        //Enn_correctionの楕円積分を初回に数点のみ行い、以降はLagrange補間する//
                        auto& mEnn_corrector = correct_info.E_nn_interpolator;

                        auto F = [&](double x) {
                            double r = fabs(convert_x_to_r(x));

                            double force_nn_dummy = 0.0;
                            double force_nn_dummy2 = 0.0;

                            double Eik = m_pp_SvF.CoreCoreEnergyCorrection(r, m_nuclei[i].Z, m_nuclei[k].Z, &force_nn_dummy);
                            double Eki = m_pp_SvF.CoreCoreEnergyCorrection(r, m_nuclei[k].Z, m_nuclei[i].Z, &force_nn_dummy2);
                            return (Eik+ Eki)/2.0;

                        };

                        const int N = mEnn_corrector.size;                        
#if 0
                        mEnn_corrector.Initialize(Quadrature::Gauss_Legendre::Points<N>(), F);
#else
                        auto points = std::make_unique<double[]>(N);
                        ChebyshevNodes::SetPoints(N, &points[0]);
                        mEnn_corrector.InitializeEven(&points[0], F);
#endif
                        
#ifdef COMPARISON_INTERPOLATION_MODEL
                        auto F2 = [&](double r) {
                            r = fabs(r);
                            double force_nn_dummy = 0.0;
                            double force_nn_dummy2 = 0.0;

                            double Eik = m_pp_SvF.CoreCoreEnergyCorrection(r, m_nuclei[i].Z, m_nuclei[k].Z, &force_nn_dummy);
                            double Eki = m_pp_SvF.CoreCoreEnergyCorrection(r, m_nuclei[k].Z, m_nuclei[i].Z, &force_nn_dummy2);
                            return (Eik + Eki) / 2.0;

                            };

                        auto FwithD = [&](double r, double* dv) {
                            

                            double dv1, dv2;
                            double Eik = m_pp_SvF.CoreCoreEnergyCorrection(fabs(r), m_nuclei[i].Z, m_nuclei[k].Z, &dv1);
                            double Eki = m_pp_SvF.CoreCoreEnergyCorrection(fabs(r), m_nuclei[k].Z, m_nuclei[i].Z, &dv2);
                            if (r > 0.0) {
                                *dv = -(dv1+dv2)/2.0;
                            } else {
                                *dv = (dv1 + dv2) / 2.0;
                            }
                            return (Eik + Eki) / 2.0;

                            };

                        auto& mEnn_corrector_cubic = correct_info.E_nn_cubic_spline;
                        {
                            constexpr int N_CUBIC_SPLINE = 16;
                            double dx = cutoff_length_PP / (double)(N_CUBIC_SPLINE - 4);
                            mEnn_corrector_cubic.Initialize(N_CUBIC_SPLINE, -1.5*dx, dx * ((double)(N_CUBIC_SPLINE-1)-1.5), CubicHermiteSpline::EDGE_DIFF_TYPE::ALWAYS_ZERO, FwithD);
                        }

                        auto& mEnn_corrector_cubic2 = correct_info.E_nn_cubic_spline2;
                        {
                            constexpr int N_CUBIC_SPLINE = 16;
                            double dx = cutoff_length_PP / (double)(N_CUBIC_SPLINE - 4);
                            mEnn_corrector_cubic2.Initialize(N_CUBIC_SPLINE, -1.5 * dx, dx * ((double)(N_CUBIC_SPLINE - 1) - 1.5), CubicHermiteSpline_diff2nd::EDGE_DIFF_TYPE::ALWAYS_ZERO, F2);
                        }

                        auto& mEnn_corrector_fifth = correct_info.E_nn_fifth_spline;
                        {
                            constexpr int N_CUBIC_SPLINE = 100;
                            double dx = cutoff_length_PP / (double)(N_CUBIC_SPLINE - 4);
                            mEnn_corrector_fifth.Initialize(N_CUBIC_SPLINE, -1.5 * dx, dx * ((double)(N_CUBIC_SPLINE - 1) - 1.5), FifthSpline::EDGE_DIFF_TYPE::ALWAYS_ZERO, F2);
                        }



                        std::string corr_name("Enn_correction_");
                        if (m_nuclei[i].Z <= m_nuclei[k].Z) {
                            corr_name += std::to_string(m_nuclei[i].Z) + "_" + std::to_string(m_nuclei[k].Z) + ".txt";
                        } else {
                            corr_name += std::to_string(m_nuclei[k].Z) + "_" + std::to_string(m_nuclei[i].Z) + ".txt";
                        }
                        FILE* fp = fopen(corr_name.c_str(), "w");

                        fprintf(fp, "#id\tdistance\tEnn_corr_int\tFnn_corr_int\tE_nn_direct\tF_nn_direct\n");
                        int ix_end = 10000;
                        for (int ix = 0; ix < ix_end; ++ix) {

                            double r = (double)(ix + 1) / (double)ix_end * cutoff_length_PP;

                            double force = 0.0;

                            double E_nn_direct = m_pp_SvF.CoreCoreEnergyCorrection(r, m_nuclei[i].Z, m_nuclei[k].Z, &force);
                            E_nn_direct += m_pp_SvF.CoreCoreEnergyCorrection(r, m_nuclei[k].Z, m_nuclei[i].Z, &force);
                            E_nn_direct /= 2.0;

                            double force_cubic;
                            double E_nn_cubic = mEnn_corrector_cubic.Interpolate(r, &force_cubic);
                            force_cubic = -force_cubic;

                            double force_fifth;
                            double E_nn_fifth = mEnn_corrector_fifth.Interpolate(r, &force_fifth);
                            force_fifth = -force_fifth;

                            const double Enn_tf = E_TF(r, Z_a, Z_b, Qi, Qk, a0, b0);
                            const double Fnn_tf = Force_TF(r, Z_a, Z_b, Qi, Qk, a0, b0);

                            fprintf(fp, "%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", ix, r,
                                mEnn_corrector.LagrangeInterpolation(inverse_r_to_x(r)),
                                -correct_info.scale_x_r * mEnn_corrector.LagrangeInterpolationDerivative(inverse_r_to_x(r)),
                                E_nn_direct, force,
                                E_nn_cubic, force_cubic,
                                E_nn_fifth, force_fifth,
                                Enn_tf, Fnn_tf, Qi* Qk / r, Qi * Qk / (r*r));
                        }
                        fclose(fp);
#endif //COMPARISON_INTERPOLATION_MODEL


                        double opt_r;
                        {//最適なカットオフ長の算出//
                        //Forceの差がzeroになるところをカットオフとする
                            double Fnn_corr, Fnn_tf;
                            const double dr = 0.05;
                            bool through_plus = false;
                            for (opt_r = cutoff_length_PP / 4.0; opt_r <= cutoff_length_PP; opt_r += dr) {

                                Fnn_corr = -correct_info.scale_x_r * mEnn_corrector.LagrangeInterpolationDerivative(inverse_r_to_x(opt_r));
                                Fnn_tf = Force_TF(opt_r, Z_a, Z_b, Qi, Qk, a0, b0);

                                double diff_F = Fnn_tf - Fnn_corr;
                                if (diff_F > 0.0) {
                                    through_plus = true;
                                }
                                if ((diff_F < 0.0) && through_plus) {
                                    double min_r = opt_r - dr;
                                    double max_r = opt_r;
                                    opt_r = (max_r + min_r) / 2.0;

                                    while (fabs(diff_F) > 1.0e-8) {
                                        
                                        Fnn_corr = -correct_info.scale_x_r * mEnn_corrector.LagrangeInterpolationDerivative(inverse_r_to_x(opt_r));
                                        Fnn_tf = Force_TF(opt_r, Z_a, Z_b, Qi, Qk, a0, b0);
                                        diff_F = Fnn_tf - Fnn_corr;
                                        if (diff_F > 0.0) {
                                            min_r = opt_r;
                                        } else {
                                            max_r = opt_r;
                                        }
                                        opt_r = (max_r + min_r) / 2.0;
                                        if (fabs(max_r - min_r) < 1.0e-8) break;
                                    }
                                    break;
                                }

                            }


                            printf("optimal_cutoff[%d,%d,r=%f]Fnn_tf - Fnn_cor = %f\n", std::min(m_nuclei[i].Z,m_nuclei[k].Z), std::max(m_nuclei[i].Z, m_nuclei[k].Z), opt_r, Fnn_tf - Fnn_corr);
                        }
                        const double Enn_correct_cutoff = mEnn_corrector.LagrangeInterpolation(inverse_r_to_x(opt_r));

#else

                        double force_nn_dummy = 0.0;


                        const double Enn_correct_cutoff = m_pp_SvF.CoreCoreEnergyCorrection(cutoff_length_PP, m_nuclei[i].Z, m_nuclei[k].Z, &force_nn_dummy);
#endif
#else
                        const double Enn_correct_cutoff = 0.0;
#endif

#ifndef IGNORE_TF_CORRECTION
                        const double Enn_tf = E_TF(opt_r, Z_a, Z_b, Qi, Qk, a0, b0);
                        correct_info.optimal_cutoff_r = opt_r;
                        optimal_cutoff_r = opt_r;
#else
                        const double Enn_tf = 0.0;
                        correct_info.optimal_cutoff_r = cutoff_length_PP/2.0;
                        optimal_cutoff_r = cutoff_length_PP / 2.0;
#endif
                        //カットオフ長のところで内殻電荷密度の積分とTFでの差が0になるように補正するための値//
                        E_correction_PP_cutoff = Enn_correct_cutoff;
                        correct_info.E_nn_cutoff = Enn_correct_cutoff;

                        E_correction_TF_cutoff = Enn_tf;                        
                        correct_info.E_tf_cutoff= Enn_tf;
                    } else {
                        //カットオフ長のところで内殻電荷密度の積分とTFでの差が0になるように補正するための値//
                        E_correction_TF_cutoff = pair_itr->second.E_tf_cutoff;
                        E_correction_PP_cutoff = pair_itr->second.E_nn_cutoff;
                        optimal_cutoff_r = pair_itr->second.optimal_cutoff_r;
                    }

                    const int w_x = (int)ceil(optimal_cutoff_r / m_box_x);
                    const int w_y = (int)ceil(optimal_cutoff_r / m_box_y);
                    const int w_z = (int)ceil(optimal_cutoff_r / m_box_z);

                    //周期境界も加味してミラー粒子とも当たり判定を行う//
                    for (int mirror_iz = -w_z; mirror_iz <= w_z; ++mirror_iz) {
                        for (int mirror_iy = -w_y; mirror_iy <= w_y; ++mirror_iy) {
                            for (int mirror_ix = -w_x; mirror_ix <= w_x; ++mirror_ix) {
                                if ((i == k) && (mirror_ix == 0) && (mirror_iy == 0) && (mirror_iz == 0)) continue;

                                double dx = xk + (double)mirror_ix * m_box_x - xi;
                                double dy = yk + (double)mirror_iy * m_box_y - yi;
                                double dz = zk + (double)mirror_iz * m_box_z - zi;
                                double r = sqrt(dx * dx + dy * dy + dz * dz);
                                if (r < optimal_cutoff_r) {

#ifndef IGNORE_CORECORE_CORRECTION

                                    
#ifdef INTERPOLATION_CORECORE_CORRECTION
                                    double x_r = inverse_r_to_x(r);
                                    auto& E_nn_interpolator = pair_itr->second.E_nn_interpolator;
                                    const double Enn_correct = E_nn_interpolator.LagrangeInterpolation(x_r);
                                    const double force_nn = -(pair_itr->second.scale_x_r) * E_nn_interpolator.LagrangeInterpolationDerivative(x_r);

#else
                                    double force_nn = 0.0;

                                    const double Enn_correct = m_pp_SvF.CoreCoreEnergyCorrection(r, m_nuclei[i].Z, m_nuclei[k].Z, &force_nn);
#endif


                                    EV1 += Enn_correct;
                                    {
                                        const double fx = force_nn * dx / (r * 2.0);
                                        const double fy = force_nn * dy / (r * 2.0);
                                        const double fz = force_nn * dz / (r * 2.0);
                                        m_force_nn_correction[i].x -= fx;
                                        m_force_nn_correction[i].y -= fy;
                                        m_force_nn_correction[i].z -= fz;
                                        m_force_nn_correction[k].x += fx;
                                        m_force_nn_correction[k].y += fy;
                                        m_force_nn_correction[k].z += fz;

                                        //printf("NNcorr[%d,%d]: %.10f, %.10f, %.10f, %.10f\n", i, k, r, fx, fy, fz);
                                    }
#endif
#ifndef IGNORE_TF_CORRECTION
                                    double E_TF_ik = E_TF(r, Z_a, Z_b, Qi, Qk, a0, b0);
                                    EV2 += E_TF_ik;
                                    double force_tf = Force_TF(r, Z_a, Z_b, Qi, Qk, a0, b0);
                                    {
                                        const double fx = force_tf * dx / (r * 2.0);
                                        const double fy = force_tf * dy / (r * 2.0);
                                        const double fz = force_tf * dz / (r * 2.0);
                                        m_force_nn_TF[i].x -= fx;
                                        m_force_nn_TF[i].y -= fy;
                                        m_force_nn_TF[i].z -= fz;
                                        m_force_nn_TF[k].x += fx;
                                        m_force_nn_TF[k].y += fy;
                                        m_force_nn_TF[k].z += fz;

                                        //printf("NN_tf_[%d,%d]: %.10f, %.10f, %.10f, %.10f\n", i, k, r, fx, fy, fz);
                                    }

                                    

#endif
                                    EV_cutoff_12 += E_correction_TF_cutoff - E_correction_PP_cutoff;
                                }//end of r cutoff
                            }
                        }
                    }
                }
            }
            /*
            {
                int i = 6;
                printf("NN_tf_[%d]: %.10f, %.10f, %.10f\n", 6, m_force_nn_correction[i].x, m_force_nn_correction[i].y, m_force_nn_correction[i].z);
            }
            */
            
            m_Enn_close_correction = (EV2 - EV1 - EV_cutoff_12) / 2.0;

            watch.Record(29);
        }

	}
}


inline
void QUMASUN_BASE1::mSetPotentialVext(double* V, const Nucleus* nuclei, int num_nuclei) {
    auto Length = [](double x, double box_w) { return (x > box_w * 0.5) ? x - box_w : (x < -box_w * 0.5) ? x + box_w : x; };
    auto SQ = [](double x) { return x * x; };

	for (size_t i = 0; i < m_size_3d; ++i) {
		V[i] = 0.0;
	}

	for (int ni = 0; ni < num_nuclei; ++ni) {
		const double Qe = m_nuclei_valence_elecron[ni];
		const double R0_x = nuclei[ni].Rx;
		const double R0_y = nuclei[ni].Ry;
		const double R0_z = nuclei[ni].Rz;
		for (int iz = 0; iz < m_size_z; ++iz) {
			const double rz2 = SQ(Length(m_dz * (double)iz - R0_z, m_box_z));
			for (int iy = 0; iy < m_size_y; ++iy) {
				const double ry2 = SQ(Length(m_dy * (double)iy - R0_y, m_box_y));
				for (int ix = 0; ix < m_size_x; ++ix) {
					const double rx2 = SQ(Length(m_dx * (double)ix - R0_x, m_box_x));
					const size_t i = ix + m_size_x * (iy + (m_size_y * iz));

					V[i] += std::max<double>(-Qe / sqrt(rx2 + ry2 + rz2), -1.0e4);

				}
			}
		}
	}

}

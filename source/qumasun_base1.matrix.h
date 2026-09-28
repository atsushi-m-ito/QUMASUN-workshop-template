#pragma once
//#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "qumasun_base1.h"
#include "GridDifference2nd.h"
#include "GridDifference4th.h"
#include "GridDifference6th.h"
#include "GridDifference8th.h"
#include "GridDifferenceKpoint2nd.h"
#include "GridDifferenceKpoint6th.h"
#include "GridDifferenceKpoint8th.h"
#include "actual_kpoint.h"
#include "lobpcg_z_multi_mpi.h"
#include "LaplacianDDMFFT.h"
#include "gyield/gyield.h"
#include "gyield/gy_for.h"


#define KINETIC_8TH


namespace QUMASUN {
    /*
    * 時間発展を行う。
    * 戻り値として時間発展後の運動エネルギーを返す
    * [重要]時間発展では位相回転するだけであり、
    * 時間発展の前後で運動エネルギーは変化しない
    */
    inline
    void OperateKineticInKspace(OneComplex* psi_k, int size_x, int size_y, int size_z, double dx, double dy, double dz, double gx, double gy, double gz)
    {


        const double g2_2 = (gx * gx + gy * gy + gz * gz) / 2.0;

        //NOTE: 
        //  V = c rho, where c = 4pi / (kx^2 + ky^2 + kz^2) //
        //    = 1.0/ ((pi*kx^2/box_x^2) + (pi*ky^2/box_y^2) + (pi*kz^2/box_z^2)) 

        const double coef1_x = (2.0 * M_PI / (dx * (double)size_x));
        const double coef1_y = (2.0 * M_PI / (dy * (double)size_y));
        const double coef1_z = (2.0 * M_PI / (dz * (double)size_z));

        
#ifdef GY_WITH_CUDA_OR_HIP
        gy::For(size_x, size_y, size_z, GY_LAMBDA(int kx, int ky, int kz) {
            const double kz1 = (coef1_z * (double)(kz * 2 > size_z ? kz - size_z : kz));
            const double ky1 = (coef1_y * (double)(ky * 2 > size_y ? ky - size_y : ky));
            const double kx1 = (coef1_x * (double)(kx * 2 > size_x ? kx - size_x : kx));
            const size_t i = kx + size_x * (ky + (size_y * kz));
            const double k2_2 = (kx1 * kx1 + ky1 * ky1 + kz1 * kz1) / 2.0;
            const double kg = (kx1 * gx + ky1 * gy + kz1 * gz);
            psi_k[i].r *= (k2_2 + g2_2 + kg);
            psi_k[i].i *= (k2_2 + g2_2 + kg);
        });
#else
        for (int kz = 0; kz < size_z; ++kz) {
            const double kz1 = (coef1_z * (double)(kz * 2 > size_z ? kz - size_z : kz));
            for (int ky = 0; ky < size_y; ++ky) {
                const double ky1 = (coef1_y * (double)(ky * 2 > size_y ? ky - size_y : ky));

                for (int kx = 0; kx < size_x; ++kx) {
                    const double kx1 = (coef1_x * (double)(kx * 2 > size_x ? kx - size_x : kx));
                    const size_t i = kx + size_x * (ky + (size_y * kz));

                    const double k2_2 = (kx1 * kx1 + ky1 * ky1 + kz1 * kz1) / 2.0;
                    const double kg = (kx1 * gx + ky1 * gy + kz1 * gz);
                    psi_k[i].r *= (k2_2 + g2_2 + kg);
                    psi_k[i].i *= (k2_2 + g2_2 + kg);

                }

            }
        }
#endif
        
    }



#if 0//def GY_WITH_CUDA_OR_HIP
    inline
        void OperateKineticInKspace_bundle(int num_bundle, OneComplex* psi_k, int size_x, int size_y, int size_z, double dx, double dy, double dz, double gx, double gy, double gz) {
        const int64_t global_size = size_x * size_y * size_z;

        for (int s = 0; s < num_bundle; ++s) {
            auto* psi_whole_complx = psi_k + s * global_size;
            QUMASUN::OperateKineticInKspace(psi_whole_complx, size_x, size_y, size_z, dx, dy, dz, gx, gy, gz);
        }
    }

#else
    inline
    void OperateKineticInKspace_bundle(int num_bundle, OneComplex* psi_k, int size_x, int size_y, int size_z, double dx, double dy, double dz, double gx, double gy, double gz) {
        const int64_t global_size = size_x * size_y * size_z;


#if 1
        

        //const double g2_2 = (gx * gx + gy * gy + gz * gz) / 2.0;

        //NOTE: 
        //  V = c rho, where c = 4pi / (kx^2 + ky^2 + kz^2) //
        //    = 1.0/ ((pi*kx^2/box_x^2) + (pi*ky^2/box_y^2) + (pi*kz^2/box_z^2)) 

        const double coef1_x = (2.0 * M_PI / (dx * (double)size_x));
        const double coef1_y = (2.0 * M_PI / (dy * (double)size_y));
        const double coef1_z = (2.0 * M_PI / (dz * (double)size_z));
        gy::For_3d_bundle(size_x, size_y, size_z, num_bundle,
            GY_LAMBDA(int kx, int ky, int kz, int n){
                const double kz1 = (coef1_z * (double)(kz * 2 > size_z ? kz - size_z : kz)) + gz;
                const double ky1 = (coef1_y * (double)(ky * 2 > size_y ? ky - size_y : ky)) + gy;
                const double kx1 = (coef1_x * (double)(kx * 2 > size_x ? kx - size_x : kx)) + gx;
                const size_t i = global_size * n + kx + size_x * (ky + (size_y * kz));
                const double k2_2 = (kx1 * kx1 + ky1 * ky1 + kz1 * kz1) / 2.0;
                //const double kg = (kx1 * gx + ky1 * gy + kz1 * gz);
                //psi_k[i].r *= (k2_2 + g2_2 + kg);
                //psi_k[i].i *= (k2_2 + g2_2 + kg);
                psi_k[i].r *= k2_2;
                psi_k[i].i *= k2_2;
            }
        );
#else
        for (int s = 0; s < num_bundle; ++s) {
            auto* psi_whole_complx = psi_k + s * global_size;
            QUMASUN::OperateKineticInKspace(psi_whole_complx, size_x, size_y, size_z, dx, dy, dz, gx, gy, gz);
        }
#endif
    }
#endif
}


/*
calculate qk = H' pk for K point
where H' = -(1/2) (\Delta +2ik \Nabla - k^2) + V
*/
inline
void QUMASUN_BASE1::mHamiltonianMatrix_ddm_bundle(SoAComplex* l_Hp, const double* l_Vtot, const SoAComplex* l_phi,
        int num_bundle, int kpoint_x, int kpoint_y, int kpoint_z, int id_spin_kpoint) {
/*#define ZERO_CLEAR
#ifdef ZERO_CLEAR
    SoAC::SetZero(l_Hp, ml_grid.Size3D());
#endif
*/



    if(m_algorithm_Laplacian==2){
        for (int n = 0; n < num_bundle; ++n) {
            //calculate q = V p , where V is effective potential////////////////
            mPotentialMatrix_ddm(l_Hp[n].re, l_Vtot, l_phi[n].re);
            mPotentialMatrix_ddm(l_Hp[n].im, l_Vtot, l_phi[n].im);
        }
        watch.Record(12);
        

        LaplacianDDMFFT( ml_grid, m_global_grid,
            *m_ddm_exchanger, 
#ifndef USE_FFT_MANY
            m_fft_x, m_fft_y, m_fft_z,
#endif
            l_Hp, l_phi, num_bundle,
            m_dx, m_dy, m_dz,
            (double)kpoint_x * m_dkx, (double)kpoint_y * m_dky, (double)kpoint_z * m_dkz, watch);
        watch.Record(11);
        
    }else { //(m_algorithm_Laplacian==1)
        double* work = m_work;// +LOBPCG::WorkSize_z_multi_mpi(ml_grid.Size3D(), num_solution);

        DDMGatherScatter exchanger;
        gyCheckError(gyGetLastError(), (std::string("DEV-ERROR") + std::to_string(__LINE__)).c_str());

        const auto range = exchanger.Estimate(m_global_grid, ml_grid, num_bundle);
        exchanger.GatherExchangeD2Z(range, m_global_grid, work, ml_grid, l_phi[0].re, num_bundle, work + 2 * m_size_3d * range.num_bundle);

        gyCheckError(gyGetLastError(), (std::string("DEV-ERROR") + std::to_string(__LINE__)).c_str());
        watch.Record(83);
        const int64_t global_size = m_global_grid.Size3D();
        const double invV = 1.0 / (double)global_size;


        if (m_fft_many_3d.GetNumBundle() != range.num_bundle) {
            m_fft_many_3d.Initialize(m_size_x, m_size_y, m_size_z, range.num_bundle, (OneComplex*)&work[0]);

            gyCheckError(gyGetLastError(), (std::string("DEV-ERROR") + std::to_string(__LINE__)).c_str());
        }
        m_fft_many_3d.ForwardDirect((OneComplex*)&work[0]);
        gyCheckError(gyGetLastError(), (std::string("DEV-ERROR") + std::to_string(__LINE__)).c_str());
        watch.Record(87);


        QUMASUN::OperateKineticInKspace_bundle(range.num_bundle, (OneComplex*)work, m_size_x, m_size_y, m_size_z, m_dx, m_dy, m_dz, (double)kpoint_x * m_dkx, (double)kpoint_y * m_dky, (double)kpoint_z * m_dkz);
        gyCheckError(gyGetLastError(), (std::string("DEV-ERROR") + std::to_string(__LINE__)).c_str());
        gy::Synchronize();
        watch.Record(11);

        m_fft_many_3d.BackwardDirect((OneComplex*)&work[0]);
        gyCheckError(gyGetLastError(), (std::string("DEV-ERROR") + std::to_string(__LINE__)).c_str());
        watch.Record(88);

        //このスケーリングを逆FFTの前に持ってくると、数値誤差が乗って収束しなくなる. FFTのバグかも.//        
        gy::For_1d_bundle<int64_t>(global_size, 2 * range.num_bundle, GY_LAMBDA(int64_t i, int n){
            work[i + n * (int64_t)(global_size)] *= invV;
        });
        gyCheckError(gyGetLastError(), (std::string("DEV-ERROR") + std::to_string(__LINE__)).c_str());
        gy::Synchronize();
        watch.Record(11);



        exchanger.ScatterExchangeZ2D(range, ml_grid, l_Hp[0].re, m_global_grid, work, num_bundle, work + 2 * m_size_3d * range.num_bundle);
        gyCheckError(gyGetLastError(), (std::string("DEV-ERROR") + std::to_string(__LINE__)).c_str());
        watch.Record(85);

        //exchanger.MergeTimer(82, watch);

    

        mPotentialMatrixAdd_ddm_bundle(l_Hp[0].re, l_Vtot, l_phi[0].re, num_bundle);
        gyCheckError(gyGetLastError(), (std::string("DEV-ERROR") + std::to_string(__LINE__)).c_str());
        gy::Synchronize();

        watch.Record(12);
    }



    if (m_hamiltonian_type == HAMILTONIAN::KohnSham_PP) {

#ifndef IGNORE_NONLOCAL


#ifdef GY_WITH_CUDA_OR_HIP
        const int bundle_width = num_bundle;
#else
        const int bundle_width = 16;
#endif

        for (int b = 0; b < num_bundle; b += bundle_width) {
            const int min_bundle = std::min(bundle_width, num_bundle - b);
            m_pp_SvF.ProjectionPP_bundle(&l_Hp[b], &l_phi[b], 0, min_bundle, ml_grid, m_nuclei, m_num_nuclei, id_spin_kpoint, (std::byte*)m_work);
        }


#endif
        gyCheckError(gyGetLastError(), (std::string("DEV-ERROR") + std::to_string(__LINE__)).c_str());
        gy::Synchronize();
        watch.Record(13);
    }
}

/*
* [deprecated]
* bundle以降使われない
calculate qk = H' pk for K point
where H' = -(1/2) (\Delta +2ik \Nabla - k^2) + V
*/
inline
void QUMASUN_BASE1::mHamiltonianMatrix_ddm(SoAComplex& l_Hp, const double* l_Vtot, const SoAComplex& l_phi, int kpoint_x, int kpoint_y, int kpoint_z, int id_spin_kpoint) {
#define ZERO_CLEAR
#ifdef ZERO_CLEAR
	SoAC::SetZero(l_Hp, ml_grid.Size3D());
#endif
	
	//calculate q = V p , where V is effective potential////////////////
	if (m_hamiltonian_type == HAMILTONIAN::KohnSham_PP) {
		mPotentialMatrix_ddm(l_Hp.re, l_Vtot, l_phi.re);
		mPotentialMatrix_ddm(l_Hp.im, l_Vtot, l_phi.im);
		watch.Record(12);


        //set bundle size is 1
        m_pp_SvF.ProjectionPP_bundle(&l_Hp, &l_phi, 0,1, ml_grid, m_nuclei, m_num_nuclei, id_spin_kpoint, (std::byte*)m_work);


        watch.Record(13);
	} else {
		mPotentialMatrix_ddm(l_Hp.re, l_Vtot, l_phi.re);
		mPotentialMatrix_ddm(l_Hp.im, l_Vtot, l_phi.im);
		watch.Record(12);
	}
	
	//calculate qk += K pk , where K is kinetic energy operator. //////////
	mKineticMatrixAdd_ddm_kpint(l_Hp, l_phi, kpoint_x, kpoint_y, kpoint_z);
	watch.Record(11);
	///////complete to create qk = Hpk //
}




//運動エネルギー演算子の作用を差分で行うもの//
//FFTは使わない//
inline
void QUMASUN_BASE1::mKineticMatrixAdd_ddm_kpint(SoAComplex& Kp, const SoAComplex& p, int kpoint_x, int kpoint_y, int kpoint_z) {
	
	
	const double kpx = (double)kpoint_x;
	const double kpy = (double)kpoint_y;
	const double kpz = (double)kpoint_z;

	

#ifdef KINETIC_8TH
	LaplacianKpoint8th_ddm(ml_grid, Kp, p, -1.0 / 2.0, m_dx, m_dy, m_dz, kpx * m_dkx, kpy * m_dky, kpz * m_dkz);
#elif defined( KINETIC_6TH)
	LaplacianKpoint6th_ddm(ml_grid, Kp, p, -1.0 / 2.0, m_dx, m_dy, m_dz, kpx * m_dkx, kpy * m_dky, kpz * m_dkz);
#elif defined( KINETIC_4TH)
	LaplacianKpoint4th_ddm(ml_grid, Kp, p, -1.0 / 2.0, m_dx, m_dy, m_dz, kpx * m_dkx, kpy * m_dky, kpz * m_dkz);
#else
	LaplacianKpoint2nd_ddm(ml_grid, Kp, p, -1.0 / 2.0, m_dx, m_dy, m_dz, kpx * m_dkx, kpy * m_dky, kpz * m_dkz);
#endif
}

// Vpr = V(r) * pr, operate V operator in real space//
inline
void QUMASUN_BASE1::mPotentialMatrix_ddm(double* Vp, const double* V, const double* p) {
	const int64_t size_3d = ml_grid.Size3D();
	for (int64_t i = 0; i < size_3d; ++i) {
		Vp[i] = V[i] * p[i];
	}
}

//calculate q = V p , where V is effective potential////////////////
inline
void QUMASUN_BASE1::mPotentialMatrixAdd_ddm_bundle(double* Vp, const double* V, const double* p, int num_bundle) {
    const int64_t size_3d = ml_grid.Size3D();


    gy::For_1d_bundle(size_3d, num_bundle * 2, GY_LAMBDA(int64_t i, int n){
        Vp[i + n * size_3d] += V[i] * p[i + n * size_3d];
    });
}
